#ifndef OPENMVG_TENSORRT_HPP
#define OPENMVG_TENSORRT_HPP

#include "openMVG/system/logger.hpp"
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

namespace openMVG
{
  namespace TensorRT
  {
    class HostAllocator
    {
    public:
      void *ptr{nullptr};
      uint64_t current_size{0};

      HostAllocator() = default;
      ~HostAllocator()
      {
        if (this->ptr)
        {
          cudaFreeHost(this->ptr);
        }
      }
      void *reallocate(uint64_t desire_size)
      {
        if (desire_size > this->current_size)
        {
          cudaError_t status = cudaSuccess;
          if (this->ptr)
          {
            status = cudaFreeHost(this->ptr);
            if (status != cudaSuccess)
            {
              OPENMVG_LOG_ERROR << "Failed to free memory" << cudaGetErrorString(status);
            }
            this->ptr = nullptr;
            this->current_size = 0;
          }
          if ((status = cudaMallocHost(&this->ptr, desire_size)) == cudaSuccess)
          {
            this->current_size = desire_size;
          }
          else
          {
            OPENMVG_LOG_ERROR << "Failed to allocate memory" << cudaGetErrorString(status);
          }
        }
        return this->ptr;
      }
    };
    class GPUAllocator
    {
    private:
      cudaStream_t stream;

    public:
      void *ptr{nullptr};
      uint64_t current_size{0};

      GPUAllocator() = default;
      GPUAllocator(cudaStream_t stream) : stream(stream) {}
      ~GPUAllocator()
      {
        cudaFree(this->ptr);
      }
      void *reallocate(uint64_t desire_size)
      {
        if (desire_size > this->current_size)
        {
          cudaError_t status = cudaSuccess;
          if (this->ptr)
          {
            status = cudaFreeAsync(this->ptr, this->stream);
            if (status != cudaSuccess)
            {
              OPENMVG_LOG_ERROR << "Failed to free memory" << cudaGetErrorString(status);
            }
            this->ptr = nullptr;
            this->current_size = 0;
          }
          if ((status = cudaMallocAsync(&this->ptr, desire_size, this->stream)) == cudaSuccess)
          {
            this->current_size = desire_size;
          }
          else
          {
            OPENMVG_LOG_ERROR << "Failed to allocate memory" << cudaGetErrorString(status);
          }
        }
        return this->ptr;
      }
    };
    class OutputAllocator : public nvinfer1::IOutputAllocator
    {
    public:
      GPUAllocator allocator;
      nvinfer1::Dims output_dims{};

      OutputAllocator() = default;
      OutputAllocator(cudaStream_t stream)
      {
        this->allocator = GPUAllocator(stream);
      }

      void *reallocateOutput(
          char const *tensor_name, void *current_memory,
          uint64_t desire_size, uint64_t alignment) noexcept override
      {
        return this->allocator.reallocate(desire_size);
      }

      void notifyShape(char const *tensor_name, nvinfer1::Dims const &dims) noexcept override
      {
        output_dims = dims;
      }
    };
  }
}

#endif