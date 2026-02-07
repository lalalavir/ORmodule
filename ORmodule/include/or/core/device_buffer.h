#pragma once

#include "or/core/cuda_check.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace orcore
{

// RAII wrapper for a typed CUDA device allocation.
template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    // Construct and allocate `count` elements.
    explicit DeviceBuffer(size_t count)
    {
        allocate(count);
    }

    // Automatically releases device memory on scope exit.
    ~DeviceBuffer()
    {
        release();
    }

    // Non-copyable: a device allocation has a single owner.
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Move support: transfer allocation ownership.
    DeviceBuffer(DeviceBuffer&& other) noexcept
    {
        move_from(other);
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            move_from(other);
        }
        return *this;
    }

    // Ensure capacity for `count` elements.
    // Reallocates only when size changes or buffer is null.
    void allocate(size_t count)
    {
        if (count == count_ && data_ != nullptr)
        {
            return;
        }

        release();

        if (count == 0)
        {
            return;
        }

        OR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
        count_ = count;
    }

    // Free the owned CUDA buffer (safe to call repeatedly).
    void release()
    {
        if (data_ != nullptr)
        {
            cudaFree(data_);
            data_ = nullptr;
            count_ = 0;
        }
    }

    // Async memset to zero on the selected stream.
    void memset_zero(cudaStream_t stream = 0)
    {
        if (data_ == nullptr || count_ == 0)
        {
            return;
        }

        OR_CUDA_CHECK(cudaMemsetAsync(data_, 0, bytes(), stream));
    }

    // Async host-to-device copy.
    // Auto-grows buffer when `count > size()`.
    void copy_from_host(const T* host_ptr, size_t count, cudaStream_t stream = 0)
    {
        if (count > count_)
        {
            allocate(count);
        }

        OR_CUDA_CHECK(cudaMemcpyAsync(data_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    // Async device-to-host copy.
    // Caller is responsible for stream synchronization if needed.
    void copy_to_host(T* host_ptr, size_t count, cudaStream_t stream = 0) const
    {
        OR_CUDA_CHECK(cudaMemcpyAsync(host_ptr, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    // Convenience wrapper around `copy_from_host`.
    void copy_from_vector(const std::vector<T>& host_values, cudaStream_t stream = 0)
    {
        copy_from_host(host_values.data(), host_values.size(), stream);
    }

    // Convenience wrapper that returns a host vector.
    // This method synchronizes the provided stream before returning.
    std::vector<T> copy_to_vector(cudaStream_t stream = 0) const
    {
        std::vector<T> host_values(count_);
        copy_to_host(host_values.data(), count_, stream);
        OR_CUDA_CHECK(cudaStreamSynchronize(stream));
        return host_values;
    }

    T* data()
    {
        return data_;
    }

    const T* data() const
    {
        return data_;
    }

    size_t size() const
    {
        return count_;
    }

    size_t bytes() const
    {
        return count_ * sizeof(T);
    }

private:
    // Internal helper used by move ctor/assignment.
    void move_from(DeviceBuffer& other)
    {
        data_ = other.data_;
        count_ = other.count_;
        other.data_ = nullptr;
        other.count_ = 0;
    }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

}  // namespace orcore
