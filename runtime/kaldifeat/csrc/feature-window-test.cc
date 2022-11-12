// kaldifeat/csrc/feature-window-test.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

#include "kaldifeat/csrc/feature-window.h"

#include "gtest/gtest.h"

namespace kaldifeat {

TEST(FeatureWindow, NumFrames) {
  FrameExtractionOptions opts;
  opts.samp_freq = 1000;
  opts.frame_length_ms = 4;
  opts.frame_shift_ms = 2;

  int32_t frame_length = opts.samp_freq / 1000 * opts.frame_length_ms;
  int32_t frame_shift = opts.samp_freq / 1000 * opts.frame_shift_ms;

  for (int32_t num_samples = 10; num_samples < 1000; ++num_samples) {
    opts.snip_edges = true;
    int32_t num_frames = NumFrames(num_samples, opts);
    int32_t expected_num_frames =
        (num_samples - frame_length) / frame_shift + 1;
    ASSERT_EQ(num_frames, expected_num_frames);

    opts.snip_edges = false;
    num_frames = NumFrames(num_samples, opts);
    expected_num_frames = (num_samples + frame_shift / 2) / frame_shift;
    ASSERT_EQ(num_frames, expected_num_frames);
  }
}

TEST(FeatureWindow, FirstSampleOfFrame) {
  FrameExtractionOptions opts;
  opts.samp_freq = 1000;
  opts.frame_length_ms = 4;
  opts.frame_shift_ms = 2;

  // samples: [a, b, c, d, e, f]
  int32_t num_samples = 6;
  opts.snip_edges = true;
  ASSERT_EQ(NumFrames(num_samples, opts), 2);
  EXPECT_EQ(FirstSampleOfFrame(0, opts), 0);
  EXPECT_EQ(FirstSampleOfFrame(1, opts), 2);

  // now for snip edges if false
  opts.snip_edges = false;
  ASSERT_EQ(NumFrames(num_samples, opts), 3);
  EXPECT_EQ(FirstSampleOfFrame(0, opts), -1);
  EXPECT_EQ(FirstSampleOfFrame(1, opts), 1);
  EXPECT_EQ(FirstSampleOfFrame(2, opts), 3);
}

TEST(FeatureWindow, GetStrided) {
  FrameExtractionOptions opts;
  opts.samp_freq = 1000;
  opts.frame_length_ms = 4;
  opts.frame_shift_ms = 2;

  // [0 1 2 3 4 5]
  torch::Tensor samples = torch::arange(0, 6).to(torch::kFloat);
  opts.snip_edges = true;
  auto frames = GetStrided(samples, opts);
  // 0 1 2 3
  // 2 3 4 5
  std::vector<float> v = {0, 1, 2, 3, 2, 3, 4, 5};
  torch::Tensor expected =
      torch::from_blob(v.data(), {int64_t(v.size())}, torch::kFloat32);
  EXPECT_TRUE(frames.flatten().allclose(expected));

  // 0 0 1 2
  // 1 2 3 4
  // 3 4 5 5
  opts.snip_edges = false;
  frames = GetStrided(samples, opts);
  v = {0, 0, 1, 2, 1, 2, 3, 4, 3, 4, 5, 5};
  expected = torch::from_blob(v.data(), {int64_t(v.size())}, torch::kFloat32);
  EXPECT_TRUE(frames.flatten().allclose(expected));
}

}  // namespace kaldifeat
