#ifndef NATIVE_ONLINE_SEED_H
#define NATIVE_ONLINE_SEED_H
#include <opencv2/core.hpp>
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <string>
#include <stdexcept>
#include <climits>
inline int online_parse_seed(const std::string& text) {
    if(text.empty())throw std::invalid_argument("seed must be decimal 0..INT_MAX");
    int value=0;
    for(char ch:text){
        if(ch<'0'||ch>'9')throw std::invalid_argument("seed must be decimal 0..INT_MAX");
        const int digit=ch-'0';
        if(value>(INT_MAX-digit)/10)throw std::invalid_argument("seed exceeds INT_MAX");
        value=value*10+digit;
    }
    return value;
}
inline void online_initialize_seed(int seed) {
    if(seed<0)throw std::invalid_argument("negative seed");
    cv::setRNGSeed(seed);
    DUtils::Random::SeedRandOnce(seed);
}
#endif
