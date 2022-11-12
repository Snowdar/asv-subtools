// kaldifeat/csrc/test_kaldifeat.cc
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

#include "torch/all.h"
#include "torch/script.h"

static void TestPreemph() {
  torch::Tensor a = torch::arange(0, 12).reshape({3, 4}).to(torch::kFloat);

  torch::Tensor b =
      a.index({"...", torch::indexing::Slice(1, torch::indexing::None,
                                             torch::indexing::None)});

  torch::Tensor c = a.index({"...", torch::indexing::Slice(0, -1, 1)});

  a.index({"...", torch::indexing::Slice(1, torch::indexing::None,
                                         torch::indexing::None)}) =
      b - 0.97 * c;

  a.index({"...", 0}) *= 0.97;

  std::cout << a << "\n";
  std::cout << b << "\n";
  std::cout << "c: \n" << c << "\n";
  torch::Tensor d = b - 0.97 * c;
  std::cout << d << "\n";
}

static void TestPad() {
  torch::Tensor a = torch::arange(0, 6).reshape({2, 3}).to(torch::kFloat);
  torch::Tensor b = torch::nn::functional::pad(
      a, torch::nn::functional::PadFuncOptions({0, 3})
             .mode(torch::kConstant)
             .value(0));
  std::cout << a << "\n";
  std::cout << b << "\n";
}

static void TestGetStrided() {
  // 0 1 2 3 4 5
  //
  //
  // 0 1 2 3
  // 2 3 4 5

  torch::Tensor a = torch::arange(0, 6).to(torch::kFloat);
  torch::Tensor b = a.as_strided({2, 4}, {2, 1});
  // b = b.clone();
  std::cout << a << "\n";
  std::cout << b << "\n";
  std::cout << b.mean(1).unsqueeze(1) << "\n";
  b = b - b.mean(1).unsqueeze(1);
  std::cout << a << "\n";
  std::cout << b << "\n";
}

static void TestDither() {
  torch::Tensor a = torch::arange(0, 6).reshape({2, 3}).to(torch::kFloat);
  torch::Tensor b = torch::arange(0, 6).reshape({2, 3}).to(torch::kFloat) * 0.1;
  std::cout << a << "\n";
  std::cout << b << "\n";
  std::cout << (a + b * 2) << "\n";
}

static void TestCat() {
  torch::Tensor a = torch::arange(0, 6).reshape({2, 3}).to(torch::kFloat);
  torch::Tensor b = torch::arange(0, 2).reshape({2, 1}).to(torch::kFloat) * 0.1;
  torch::Tensor c = torch::cat({a, b}, 1);
  torch::Tensor d = torch::cat({b, a}, 1);
  torch::Tensor e = torch::cat({a, a}, 0);
  std::cout << a << "\n";
  std::cout << b << "\n";
  std::cout << c << "\n";
  std::cout << d << "\n";
  std::cout << e << "\n";
}

int main() {
  TestCat();
  return 0;
}
