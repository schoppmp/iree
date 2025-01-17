// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/local/elf/platform.h"

#if defined(IREE_PLATFORM_APPLE)

// NOTE: because Apple there's some hoop-jumping to get executable code.
// https://developer.apple.com/documentation/apple-silicon/porting-just-in-time-compilers-to-apple-silicon
// https://keith.github.io/xcode-man-pages/pthread_jit_write_protect_np.3.html

#include <errno.h>
#include <libkern/OSCacheControl.h>
#include <mach/vm_statistics.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

// MAP_JIT and related utilities are only available on MacOS 11.0+.
#if defined(MAC_OS_VERSION_11_0) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_11_0
#define IREE_APPLE_IF_AT_LEAST_MAC_OS_11_0(expr) \
  if (__builtin_available(macOS 11.0, *)) {      \
    expr                                         \
  }
#else
#define IREE_APPLE_IF_AT_LEAST_MAC_OS_11_0(expr)
#endif  // MAC_OS_VERSION_11_0

//==============================================================================
// Memory subsystem information and control
//==============================================================================

void iree_memory_query_info(iree_memory_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));

  int page_size = sysconf(_SC_PAGESIZE);
  out_info->normal_page_size = page_size;
  out_info->normal_page_granularity = page_size;
  out_info->large_page_granularity = (2 * 1024 * 1024);  // What V8 uses.

  out_info->can_allocate_executable_pages = true;
}

void iree_memory_jit_context_begin(void) {
  IREE_APPLE_IF_AT_LEAST_MAC_OS_11_0({
    if (pthread_jit_write_protect_supported_np()) {
      pthread_jit_write_protect_np(0);
    }
  });
}

void iree_memory_jit_context_end(void) {
  IREE_APPLE_IF_AT_LEAST_MAC_OS_11_0({
    if (pthread_jit_write_protect_supported_np()) {
      pthread_jit_write_protect_np(1);
    }
  });
}

//==============================================================================
// Virtual address space manipulation
//==============================================================================

// This user tag makes it easier to find our pages in vmmap dumps.
#define IREE_MEMORY_MMAP_FD VM_MAKE_TAG(255)

static int iree_memory_access_to_prot(iree_memory_access_t access) {
  int prot = 0;
  if (access & IREE_MEMORY_ACCESS_READ) prot |= PROT_READ;
  if (access & IREE_MEMORY_ACCESS_WRITE) prot |= PROT_WRITE;
  if (access & IREE_MEMORY_ACCESS_EXECUTE) prot |= PROT_EXEC;
  return prot;
}

iree_status_t iree_memory_view_reserve(iree_memory_view_flags_t flags,
                                       iree_host_size_t total_length,
                                       void** out_base_address) {
  *out_base_address = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  int mmap_prot = PROT_NONE;
  int mmap_flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
  IREE_APPLE_IF_AT_LEAST_MAC_OS_11_0({
    if (flags & IREE_MEMORY_VIEW_FLAG_MAY_EXECUTE) {
      mmap_flags |= MAP_JIT;
    }
  });

  iree_status_t status = iree_ok_status();
  void* base_address =
      mmap(NULL, total_length, mmap_prot, mmap_flags, IREE_MEMORY_MMAP_FD, 0);
  if (base_address == MAP_FAILED) {
    status = iree_make_status(iree_status_code_from_errno(errno),
                              "mmap reservation failed");
  }

  *out_base_address = base_address;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_memory_view_release(void* base_address,
                              iree_host_size_t total_length) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: return value ignored as this is a shutdown path.
  munmap(base_address, total_length);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_memory_view_commit_ranges(
    void* base_address, iree_host_size_t range_count,
    const iree_byte_range_t* ranges, iree_memory_access_t initial_access) {
  IREE_TRACE_ZONE_BEGIN(z0);

  int mmap_prot = iree_memory_access_to_prot(initial_access);
  int mmap_flags = MAP_PRIVATE | MAP_ANON | MAP_FIXED;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    void* range_start = (void*)iree_page_align_start(
        (uintptr_t)base_address + ranges[i].offset, getpagesize());
    void* result = mmap(range_start, ranges[i].length, mmap_prot, mmap_flags,
                        IREE_MEMORY_MMAP_FD, 0);
    if (result == MAP_FAILED) {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "mmap commit failed");
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_memory_view_protect_ranges(void* base_address,
                                              iree_host_size_t range_count,
                                              const iree_byte_range_t* ranges,
                                              iree_memory_access_t new_access) {
  IREE_TRACE_ZONE_BEGIN(z0);

  int mmap_prot = iree_memory_access_to_prot(new_access);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    void* range_start = (void*)iree_page_align_start(
        (uintptr_t)base_address + ranges[i].offset, getpagesize());
    int ret = mprotect(range_start, ranges[i].length, mmap_prot);
    if (ret != 0) {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "mprotect failed");
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void sys_icache_invalidate(void* start, size_t len);

void iree_memory_view_flush_icache(void* base_address,
                                   iree_host_size_t length) {
  sys_icache_invalidate(base_address, length);
}

#endif  // IREE_PLATFORM_APPLE
