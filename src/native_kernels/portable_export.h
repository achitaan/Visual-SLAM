#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32) || defined(_MSC_VER)
  #define VSLAM_EXPORT __declspec(dllexport)
  #define VSLAM_FORCE_INLINE __forceinline
  #define VSLAM_RESTRICT __restrict
  #include <malloc.h>
  #if defined(_MSC_VER)
    #include <intrin.h>
  #endif
  inline void* vslam_aligned_alloc(size_t align, size_t size) {
      return _aligned_malloc(size, align);
  }
  inline void vslam_aligned_free(void* p) {
      _aligned_free(p);
  }
  inline int32_t vslam_ctz32(uint32_t mask) {
  #if defined(_MSC_VER)
      unsigned long idx = 0;
      _BitScanForward(&idx, mask);
      return static_cast<int32_t>(idx);
  #else
      return __builtin_ctz(mask);
  #endif
  }
#else
  #define VSLAM_EXPORT __attribute__((visibility("default")))
  #define VSLAM_FORCE_INLINE inline __attribute__((always_inline))
  #define VSLAM_RESTRICT __restrict
  #include <cstdlib>
  inline void* vslam_aligned_alloc(size_t align, size_t size) {
      void* ptr = nullptr;
      if (posix_memalign(&ptr, align, size) != 0) {
          return nullptr;
      }
      return ptr;
  }
  inline void vslam_aligned_free(void* p) {
      std::free(p);
  }
  inline int32_t vslam_ctz32(uint32_t mask) {
      return __builtin_ctz(mask);
  }
#endif
