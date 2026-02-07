# ORmodule

中文 | [English](#english)

`ORmodule` 是一个面向 GPU 的通用调度求解框架子项目，采用 **Engine + Problem 插件** 思路，目标是支持 JSP/FJSP 等问题在同一求解管线下运行。

## 中文

### 这个仓库（子目录）包含什么

- `ORmodule.vcxproj`：Visual Studio CUDA 工程文件
- `main.cpp`：当前工程入口（可作为后续框架集成入口）
- `generic_gpu_solver_framework.md`：通用框架实现蓝图（核心设计文档）

> 说明：这是“子目录发布版”README，只描述当前子目录内容。

### 设计目标

- 建立问题无关的 GPU 搜索引擎（初始化、评估、选择、重启、统计）
- 通过 `ProblemOps` 风格接口接入不同调度问题
- 支持后续从 JSP 扩展到 FJSP（OS+MS）

### 架构概要

- **Core**：CUDA 基础设施（内存、RNG、归约/排序工具）
- **Engine**：统一求解管线（迭代控制、生存选择、精英维护）
- **Problem**：问题插件层（编码、`evaluate`、邻域、修复、交叉）

详细设计见：`generic_gpu_solver_framework.md`

### 构建说明（Windows）

推荐环境：
- Visual Studio 2022
- CUDA Toolkit（版本与本机 VS 工具链匹配）

构建步骤：
1. 打开 `ORmodule.vcxproj`（或放入你的解决方案）
2. 选择 `x64` + `Release`
3. 构建并运行

### 当前状态

- 子项目结构已建立
- 通用框架蓝图已完成
- 下一步是落地 `core/engine/problem` 最小可编译骨架

### 路线图

- [ ] 抽象 `ProblemOps` 接口
- [ ] 实现 `SolverEngine` 最小闭环
- [ ] 增加 JSP 插件适配
- [ ] 增加 FJSP 插件适配

---

## English

### What this sub-repo contains

- `ORmodule.vcxproj`: Visual Studio CUDA project file
- `main.cpp`: current project entry point (future integration entry)
- `generic_gpu_solver_framework.md`: implementation blueprint for the generic framework

> Note: this README is tailored for **subdirectory-only publishing**.

### Goals

- Build a problem-agnostic GPU search engine (init, evaluate, selection, restart, stats)
- Plug in different scheduling problems via a `ProblemOps`-style interface
- Extend from JSP to FJSP (OS+MS)

### Architecture

- **Core**: CUDA infrastructure (memory, RNG, reduction/sorting helpers)
- **Engine**: unified solving pipeline (iteration control, survival, archive)
- **Problem**: problem plugins (encoding, `evaluate`, neighborhood, repair, crossover)

See details in: `generic_gpu_solver_framework.md`

### Build (Windows)

Recommended environment:
- Visual Studio 2022
- CUDA Toolkit (compatible with local VS toolchain)

Steps:
1. Open `ORmodule.vcxproj` (or include it in your solution)
2. Select `x64` + `Release`
3. Build and run

### Current status

- Subproject skeleton is in place
- Generic framework blueprint is documented
- Next step is implementing a minimal compilable `core/engine/problem` skeleton

### Roadmap

- [ ] Define `ProblemOps` abstraction
- [ ] Implement minimal `SolverEngine` loop
- [ ] Add JSP plugin adapter
- [ ] Add FJSP plugin adapter

