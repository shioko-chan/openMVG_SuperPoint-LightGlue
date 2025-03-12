#ifndef OPENMVG_CUDA_HPP
#define OPENMVG_CUDA_HPP

#include "openMVG/system/logger.hpp"

#include <cuda_runtime_api.h>

namespace openMVG
{
  namespace CUDA
  {
    namespace
    {
      class MemInterface
      {
      public:
        virtual int free(void *ptr) noexcept = 0;
        virtual int alloc(void **ptr, uint64_t desire_size) noexcept = 0;

        virtual ~MemInterface() = default;
      };

      template <typename T>
      class AllocatorImpl
      {
      private:
        T *ptr{nullptr};
        uint64_t current_size{0};
        std::unique_ptr<MemInterface> interface;

        inline bool free_() noexcept
        {
          if (ptr && interface->free(static_cast<void *>(ptr)) != 0)
          {
            OPENMVG_LOG_ERROR << "Failed to free memory: " << cudaGetLastError();
            return false;
          }
          return true;
        }
        inline bool alloc_(uint64_t desire_size) noexcept
        {
          if (interface->alloc(static_cast<void **>(&ptr), desire_size * sizeof(T)) != 0)
          {
            OPENMVG_LOG_ERROR << "Failed to allocate memory: " << cudaGetLastError();
            return false;
          }
          return true;
        }

      public:
        AllocatorImpl() = delete;
        AllocatorImpl(std::unique_ptr<MemInterface> interface) : interface(std::move(interface)) {}

        ~AllocatorImpl() noexcept
        {
          this->release();
        }

        inline T *data() noexcept
        {
          return this->ptr;
        }

        inline void release() noexcept
        {
          this->free_();
          ptr = nullptr;
          current_size = 0;
        }

        inline void *reallocate(uint64_t desire_size) noexcept
        {
          if (desire_size > current_size)
          {
            if (ptr)
            {
              this->release();
            }
            if (this->alloc_(desire_size))
            {
              current_size = desire_size;
            }
            else
            {
              return nullptr;
            }
          }
          return this->ptr;
        }
      };

      class HostMemInterface : public MemInterface
      {
      public:
        inline int free(void *ptr) noexcept override
        {
          return static_cast<int>(cudaFreeHost(ptr));
        }
        inline int alloc(void **ptr, uint64_t desire_size) noexcept override
        {
          return static_cast<int>(cudaMallocHost(ptr, desire_size));
        }
      };

      class GPUMemInterface : public MemInterface
      {
      public:
        inline int free(void *ptr) noexcept override
        {
          return static_cast<int>(cudaFree(ptr));
        }
        inline int alloc(void **ptr, uint64_t desire_size) noexcept override
        {
          return static_cast<int>(cudaMalloc(ptr, desire_size));
        }
      };

      class GPUStreamMemInterface : public MemInterface
      {
      private:
        cudaStream_t stream;

      public:
        inline int free(void *ptr) noexcept override
        {
          return static_cast<int>(cudaFreeAsync(ptr, stream));
        }
        inline int alloc(void **ptr, uint64_t desire_size) noexcept override
        {
          return static_cast<int>(cudaMallocAsync(ptr, desire_size, stream));
        }

        GPUStreamMemInterface() = delete;
        GPUStreamMemInterface(cudaStream_t stream) : stream(stream) {}
      };

    }

    template <typename T>
    class Allocator
    {
    private:
      std::unique_ptr<AllocatorImpl<T>> allocator;

    public:
      Allocator() = delete;
      Allocator(std::unique_ptr<MemInterface> interface) : allocator(std::move(interface)) {}

      inline T *reallocate(uint64_t desire_size) noexcept override
      {
        return allocator->reallocate(desire_size);
      }

      inline T *data() noexcept override
      {
        return allocator->data();
      }

      inline void release() noexcept override
      {
        allocator->release();
      }
    };

    template <typename T>
    class HostAllocator : public Allocator<T>
    {
    public:
      HostAllocator() : Allocator<T>(std::make_unique<HostMemInterface>()) {}
    };

    template <typename T>
    class GPUAllocator : public Allocator<T>
    {
    public:
      GPUAllocator() : Allocator<T>(std::make_unique<GPUMemInterface>()) {}
    };

    template <typename T>
    class GPUStreamAllocator : public Allocator<T>
    {
    public:
      GPUStreamAllocator() = delete;
      GPUStreamAllocator(cudaStream_t stream) : Allocator<T>(std::make_unique<GPUStreamMemInterface>(stream)) {}
    };

  }
}

#endif