#ifndef SLAM_NATIVE_ORIGIN_COUNTER_H
#define SLAM_NATIVE_ORIGIN_COUNTER_H
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
// One process-wide counter, instantiated only in Map.cc; zero is unavailable.
class OnlineOriginCounter {
    std::atomic<std::uint64_t> next_;
public:
    explicit OnlineOriginCounter(std::uint64_t first=1):next_(first) {}
    std::uint64_t allocate() {
        auto value=next_.load(std::memory_order_relaxed);
        for(;;) {
            if(value==0 || value==std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("native coordinate generation exhausted");
            if(next_.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) return value;
        }
    }
};
#endif
