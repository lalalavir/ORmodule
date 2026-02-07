# ORmodule 通用 GPU 调度求解器框架（实现蓝图）

## 1. 目标

在 `ORmodule` 内建立一个可扩展的 **Engine + Problem 插件**框架，先兼容 JSP，再扩展到 FJSP/PFSP。

核心目标：
- 复用通用搜索管线（种群、精英、多样性、重启、日志）。
- 问题特定逻辑插件化（编码、评估、邻域、修复、交叉）。
- 保持 GPU 友好（SoA 布局、按解并行、批处理扩展）。

---

## 2. 目录结构（建议）

```text
ORmodule/
  include/or/
    core/
      cuda_check.h
      device_buffer.h
      rng_state.h
      reduction.h
      sort_topk.h
    engine/
      run_config.h
      run_stats.h
      solver_engine.h
      archive.h
      diversity.h
    problem/
      problem_ops.h
      problem_traits.h
    sched/
      jsp/
        jsp_instance.h
        jsp_solution.h
        jsp_ops.cuh
      fjsp/
        fjsp_instance.h
        fjsp_solution.h
        fjsp_ops.cuh
  src/
    engine/
      solver_engine.cu
      archive.cu
      diversity.cu
    sched/
      jsp/
        jsp_ops.cu
      fjsp/
        fjsp_ops.cu
  examples/
    jsp_demo.cu
    fjsp_demo.cu
```

---

## 3. 分层职责

### 3.1 Core（问题无关）
- CUDA 错误检查、设备内存 RAII、随机数初始化。
- 通用归约（best/avg/var）、排序/Top-K、统计采样。
- 不出现任何 JSP/FJSP 语义。

### 3.2 Engine（弱问题相关）
- 统一迭代管线：`init -> evaluate -> variation -> local search -> repair -> selection -> archive -> restart`。
- 统一控制配置：时间限制、迭代次数、种群规模、算子开关。
- 统一输出：最优值、平均值、收敛轨迹、可行率。

### 3.3 Problem（强问题相关）
- 实例数据布局（host/device）。
- 解表示（染色体/调度序列/机器分配）。
- `evaluate`（核心约束与目标计算）。
- `propose_move / crossover / repair / distance` 等问题算子。

---

## 4. 统一接口（关键）

```cpp
template <class Problem>
struct ProblemOps {
  using Instance = typename Problem::Instance;
  using Population = typename Problem::Population;
  using Scratch = typename Problem::Scratch;

  static void random_init(const Instance&, Population&, curandState* rng);
  static void evaluate(const Instance&, Population&, Scratch&);
  static void propose_move(const Instance&, Population&, Scratch&, curandState* rng);
  static void apply_move(Population&, Scratch&);
  static void repair(const Instance&, Population&, Scratch&);          // 可选
  static void crossover(const Instance&, const Population&, Population&, curandState* rng); // 可选
  static void distance(const Population&, DeviceBuffer<int>& out_dist); // 可选
};
```

说明：
- `Engine` 只依赖这个接口，不依赖 JSP/FJSP 实现细节。
- 若某问题不支持某算子，可在配置中关闭并走默认分支。

---

## 5. 引擎主循环（伪代码）

```cpp
initialize_rng();
ProblemOps::random_init(instance, pop, rng);
ProblemOps::evaluate(instance, pop, scratch);
archive.update(pop);

for (gen = 0; gen < max_gen && !timeout; ++gen) {
  if (cfg.enable_crossover) {
    ProblemOps::crossover(instance, pop, offspring, rng);
  } else {
    offspring.clone_from(pop);
  }

  for (int t = 0; t < cfg.local_search_iters; ++t) {
    ProblemOps::propose_move(instance, offspring, scratch, rng);
    ProblemOps::apply_move(offspring, scratch);
    ProblemOps::evaluate(instance, offspring, scratch);
  }

  if (cfg.enable_repair) {
    ProblemOps::repair(instance, offspring, scratch);
    ProblemOps::evaluate(instance, offspring, scratch);
  }

  engine_survival_select(pop, offspring, archive, cfg); // 质量+多样性
  archive.update(pop);

  if (engine_stagnated()) {
    engine_restart_or_inject(pop, archive, rng);
  }

  stats.collect(pop, archive);
}
```

---

## 6. 与现有 JSP 代码的映射

可直接复用并迁移：
- `kernel_evaluate` -> `ProblemOps<JSP>::evaluate`
- `kernel_proposal_move` -> `ProblemOps<JSP>::propose_move`
- `kernel_update_best` -> `engine::archive.update`
- `kernel_compute_scores_counted` + `kernel_gather_top_k` -> `engine::survival_select`
- `kernel_conditional_restart` -> `engine::restart_or_inject`

这意味着：
- 你当前 `JSP` 的算法强度可以保留。
- 只是把“调度细节”和“搜索管线”拆开，形成通用骨架。

---

## 7. Evaluate 设计原则（针对你当前讨论）

- JSP/FJSP 主线继续用 **拓扑/解码式 DP evaluate**（稀疏 DAG 下复杂度更优）。
- 热带矩阵乘保留为实验后端，不作为默认主后端。
- 对大规模实例优先做：
  - 按解并行（block-per-solution）
  - 批处理流式评估（batch + streams）
  - 增量评估（局部移动后只重算受影响部分）

---

## 8. 里程碑（落地顺序）

### M1（最小可运行）
- 建立 `core/engine/problem` 基础头文件。
- 跑通 `Engine + JSP Problem`：初始化、评估、生存选择、日志。

### M2（算法能力回迁）
- 把现有 JSP 的 tabu/elite/restart/hash 全部迁入 engine 或 strategy。
- 保持结果质量不退化。

### M3（多问题扩展）
- 新增 `FJSP Problem`（OS+MS）。
- 使用同一 `Engine` 直接运行并对比收敛曲线。

### M4（性能增强）
- 批处理 + 多 stream。
- 可选多 GPU（岛模型）。
- profile 驱动优化（shared memory、occupancy、memory coalescing）。

---

## 9. 完成判据

当满足以下条件，可认为“通用框架”阶段完成：
- 同一 `solver_engine` 不改代码，能跑 `JSP` 与 `FJSP` 两个插件。
- 通用模块中无 JSP/FJSP 业务字段。
- `evaluate` 与问题算子都在 Problem 插件层。
- 统计/日志/重启/精英池由 Engine 统一负责。

