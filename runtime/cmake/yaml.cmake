# third_party: yaml

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "build test")
FetchContent_Declare(yaml
  URL      https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.6.3.zip
  URL_HASH SHA256=7c0ddc08a99655508ae110ba48726c67e4a10b290c214aed866ce4bbcbe3e84c
)
FetchContent_MakeAvailable(yaml)
include_directories(${yaml_SOURCE_DIR}/include ${yaml_BINARY_DIR})

# FetchContent_GetProperties(yaml)
# if(NOT yaml-cpp_POPULATED)
#   message(STATUS "Fetching yaml-cpp...")
#   FetchContent_Populate(yaml)
#   add_subdirectory(${yaml_SOURCE_DIR} ${yaml_BINARY_DIR})
# endif()
# include_directories(${yaml_SOURCE_DIR}/include)