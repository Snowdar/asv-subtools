# Copyright (c)  2021  Xiaomi Corporation (author: Fangjun Kuang)
project(kaldifeat)
set(kaldifeat_srcs
  feature-fbank.cc
  feature-functions.cc
  feature-mfcc.cc
  feature-plp.cc
  feature-spectrogram.cc
  feature-window.cc
  matrix-functions.cc
  mel-computations.cc
)


add_library(kaldifeat_core STATIC ${kaldifeat_srcs})

target_link_libraries(kaldifeat_core PUBLIC ${TORCH_LIBRARIES})

add_executable(test_kaldifeat test_kaldifeat.cc)
target_link_libraries(test_kaldifeat PRIVATE kaldifeat_core)

function(kaldifeat_add_test source)
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} "${source}")
  target_link_libraries(${name}
    PRIVATE
      kaldifeat_core
      gtest
      gtest_main
  )

  add_test(NAME "Test.${name}"
    COMMAND
    $<TARGET_FILE:${name}>
  )
endfunction()

if(kaldifeat_BUILD_TESTS)
  # please sort the source files alphabetically
  set(test_srcs
    feature-window-test.cc
  )

  foreach(source IN LISTS test_srcs)
    kaldifeat_add_test(${source})
  endforeach()
endif()
