cmake_minimum_required(VERSION 3.24)
project(ninfer_prefix_disk_cpu LANGUAGES CXX)

# Standalone correctness gate: no CUDA toolkit, driver, model artifact or Python required.
get_filename_component(NINFER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
find_package(Threads REQUIRED)
enable_testing()
option(NINFER_PREFIX_DISK_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

add_library(ninfer_prefix_disk_cpu_support STATIC
  "${NINFER_ROOT}/src/core/disk_kv_store.cpp"
  "${NINFER_ROOT}/src/runtime/prefix_cache/block_hash.cpp"
  "${NINFER_ROOT}/src/runtime/prefix_cache/prefix_index.cpp"
  "${NINFER_ROOT}/src/runtime/prefix_cache/tap_planner.cpp"
  "${NINFER_ROOT}/src/runtime/prefix_cache/disk_catalog.cpp"
  "${NINFER_ROOT}/src/runtime/prefix_cache/disk_worker.cpp"
  "${NINFER_ROOT}/src/models/qwen3_5/program/prefix/hybrid_disk.cpp")
target_compile_features(ninfer_prefix_disk_cpu_support PUBLIC cxx_std_20)
set_target_properties(ninfer_prefix_disk_cpu_support PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(ninfer_prefix_disk_cpu_support PUBLIC
  "${NINFER_ROOT}/include" "${NINFER_ROOT}/src")
target_link_libraries(ninfer_prefix_disk_cpu_support PUBLIC Threads::Threads)
if(MSVC)
  target_compile_options(ninfer_prefix_disk_cpu_support PUBLIC /W4)
else()
  target_compile_options(ninfer_prefix_disk_cpu_support PUBLIC -Wall -Wextra)
endif()
if(NINFER_PREFIX_DISK_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "The prefix disk sanitizer gate requires GCC or Clang")
  endif()
  target_compile_options(ninfer_prefix_disk_cpu_support PUBLIC
    -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(ninfer_prefix_disk_cpu_support PUBLIC -fsanitize=address,undefined)
endif()

foreach(test IN ITEMS prefix_cache_index prefix_disk_catalog prefix_disk_worker
                      hybrid_disk_cpu prefix_disk_crash prefix_disk_reference)
  add_executable(ninfer_${test}_test "${NINFER_ROOT}/tests/test_${test}.cpp")
  target_link_libraries(ninfer_${test}_test PRIVATE ninfer_prefix_disk_cpu_support)
  add_test(NAME ninfer_${test}_test COMMAND ninfer_${test}_test)
  set_tests_properties(ninfer_${test}_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endforeach()
