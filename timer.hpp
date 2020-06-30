#ifndef TIMER_HPP
#define TIMER_HPP

#include <cstdint>
#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>

class Timer {
public:
    void start() {
        start_ = std::chrono::steady_clock::now();
    }
    int64_t elapsedSec() {
        auto n = std::chrono::steady_clock::now();
        auto e = std::chrono::duration_cast<std::chrono::seconds>(n - start_);
        return e.count();
    }
    float elapsedHour() {
        return elapsedSec() / 3600.0;
    }
    std::string elapsedTimeStr() {
        std::ostringstream oss;
        int64_t sec = elapsedSec();
        oss << std::setfill('0') << std::setw(3) << sec / 3600 << ":";
        sec %= 3600;
        oss << std::setfill('0') << std::setw(2) << sec / 60 << ":";
        sec %= 60;
        oss << std::setfill('0') << std::setw(2) << sec;
        return oss.str();
    }
private:
    std::chrono::steady_clock::time_point start_;
};

#endif //TIMER_HPP