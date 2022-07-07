cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(subtools_jit VERSION 0.1)


include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/kaldi  # for not changing c++ header names in kaldi source files
)
set(CMAKE_VERBOSE_MAKEFILE on)


include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

if(NOT MSVC)
  # Keep the same with openfst, -fPIC or -fpic
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
endif()

# third_party: gflags
FetchContent_Declare(gflags
  URL      https://github.com/gflags/gflags/archive/v2.2.1.zip
  URL_HASH SHA256=4e44b69e709c826734dbbbd5208f61888a2faf63f239d73d8ba0011b2dccc97a
)
FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include)

# third_party: glog
FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/v0.4.0.zip
  URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})


# third_party: boost
FetchContent_Declare(boost
  # URL      https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz
  URL      /tsdata/wenet/runtime/server/x86/boost_1_75_0.tar.gz
  URL_HASH SHA256=aeb26f80e80945e82ee93e5939baebdca47b9dee80a07d3144be1e1a6a66dd6a
)
FetchContent_MakeAvailable(boost)
include_directories(${boost_SOURCE_DIR})

# third_party: yaml
FetchContent_Declare(yaml
  URL      https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.6.3.zip
  URL_HASH SHA256=7c0ddc08a99655508ae110ba48726c67e4a10b290c214aed866ce4bbcbe3e84c
)
FetchContent_MakeAvailable(yaml)
include_directories(${yaml_SOURCE_DIR}/include)

# third_party: gtest
FetchContent_Declare(googletest
  URL      https://github.com/google/googletest/archive/release-1.10.0.zip
  URL_HASH SHA256=94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91
)
FetchContent_MakeAvailable(googletest)

# third_party: libtorch use FetchContent_Declare to download, and
# use find_package to find since libtorch is not a standard cmake project
set(PYTORCH_VERSION "1.9.0")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
    set(URL_HASH "SHA256=d7043b7d7bdb5463e5027c896ac21b83257c32c533427d4d0d7b251548db8f4b")
    set(CMAKE_BUILD_TYPE "Release")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIBTORCH_URL "/tsdata/wenet/runtime/server/x86/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}+cpu.zip")
    # set(URL_HASH "SHA256=6d7be1073d1bd76f6563572b2aa5548ad51d5bc241d6895e3181b7dc25554426")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${PYTORCH_VERSION}.zip")
    set(URL_HASH "SHA256=07cac2c36c34f13065cb9559ad5270109ecbb468252fb0aeccfd89322322a2b5")
else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux' or 'Darwin')")
endif()

FetchContent_Declare(libtorch
  URL      ${LIBTORCH_URL}
  # URL_HASH ${URL_HASH}
)
set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
FetchContent_MakeAvailable(libtorch)

find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")

add_subdirectory(kaldifeat/csrc)

# utils
add_library(utils STATIC
  utils/utils.cc utils/options.cc utils/string.cc
)
target_link_libraries(utils PUBLIC gflags glog yaml-cpp)


# frontend
add_library(frontend STATIC
  frontend/feature_pipeline.cc frontend/features.cc
)
target_link_libraries(frontend PUBLIC utils kaldifeat_core)

# extractor
add_library(extractor STATIC
  extractor/torch_asv_extractor.cc
  extractor/torch_asv_model.cc
)
target_link_libraries(extractor PUBLIC ${TORCH_LIBRARIES} utils)

# binary
add_executable(extractor_main bin/extractor_main.cc)
target_link_libraries(extractor_main PUBLIC utils frontend extractor)





