// kaldifeat/csrc/log.h
//
// Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)

#ifndef KALDIFEAT_CSRC_LOG_H_
#define KALDIFEAT_CSRC_LOG_H_

#include <cstdlib>
#include <iostream>
#include <sstream>

namespace kaldifeat {

enum class LogLevel {
  kInfo = 0,
  kWarn = 1,
  kError = 2,  // abort the program
};

class Logger {
 public:
  Logger(const char *filename, const char *func_name, uint32_t line_num,
         LogLevel level)
      : filename_(filename),
        func_name_(func_name),
        line_num_(line_num),
        level_(level) {
    os_ << filename << ":" << func_name << ":" << line_num << "\n";
    switch (level_) {
      case LogLevel::kInfo:
        os_ << "[I] ";
        break;
      case LogLevel::kWarn:
        os_ << "[W] ";
        break;
      case LogLevel::kError:
        os_ << "[E] ";
        break;
    }
  }

  template <typename T>
  Logger &operator<<(const T &val) {
    os_ << val;
    return *this;
  }

  ~Logger() {
    std::cerr << os_.str() << "\n";
    if (level_ == LogLevel::kError) abort();
  }

 private:
  std::ostringstream os_;
  const char *filename_;
  const char *func_name_;
  uint32_t line_num_;
  LogLevel level_;
};

class Voidifier {
 public:
  void operator&(const Logger &)const {}
};

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__PRETTY_FUNCTION__)
// for clang and GCC
#define KALDIFEAT_FUNC __PRETTY_FUNCTION__
#else
// for other compilers
#define KALDIFEAT_FUNC __func__
#endif

#define KALDIFEAT_LOG                                   \
  kaldifeat::Logger(__FILE__, KALDIFEAT_FUNC, __LINE__, \
                    kaldifeat::LogLevel::kInfo)

#define KALDIFEAT_WARN                                  \
  kaldifeat::Logger(__FILE__, KALDIFEAT_FUNC, __LINE__, \
                    kaldifeat::LogLevel::kWarn)

#define KALDIFEAT_ERR                                   \
  kaldifeat::Logger(__FILE__, KALDIFEAT_FUNC, __LINE__, \
                    kaldifeat::LogLevel::kError)

#define KALDIFEAT_ASSERT(x)                                         \
  (x) ? (void)0                                                     \
      : kaldifeat::Voidifier() & KALDIFEAT_ERR << "Check failed!\n" \
                                               << "x: " << #x

}  // namespace kaldifeat

#endif  // KALDIFEAT_CSRC_LOG_H_
