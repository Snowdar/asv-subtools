cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(subtools_jit VERSION 0.1)


set(CMAKE_VERBOSE_MAKEFILE on)
option(CXX11_ABI "whether to use CXX11_ABI libtorch" OFF)


include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})



list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")




include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
)


include(libtorch)
include(yaml)
# include(gtest)
include(gflags)
include(glog)


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





