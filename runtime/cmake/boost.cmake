# third_party: boost
FetchContent_Declare(boost
  # URL      https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz
  URL      /work/kaldi/egs/ldx/voxceleb/require_run/boost_1_75_0.tar.gz
  URL_HASH SHA256=aeb26f80e80945e82ee93e5939baebdca47b9dee80a07d3144be1e1a6a66dd6a
)
FetchContent_MakeAvailable(boost)
include_directories(${boost_SOURCE_DIR})