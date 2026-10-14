#ifndef SLAM_NATIVE_ONLINE_AUDIT_H
#define SLAM_NATIVE_ONLINE_AUDIT_H
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
namespace ORB_SLAM3 { namespace OnlineAudit {
inline int64_t NowNs() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct State {
    std::atomic<uint64_t> admitted{0},ba_started{0},ba_finished{0};
    std::mutex mutex;
    FILE* events=nullptr;
};
inline State& Shared(){ static State s; return s; }
inline unsigned& PoseCalls(){ static thread_local unsigned n=0;return n; }
inline uint64_t& PoseFrame(){ static thread_local uint64_t id=0;return id; }
inline void Event(const char* kind,uint64_t frame,int64_t map,int iterations=-1) {
    State& s=Shared(); std::lock_guard<std::mutex> lock(s.mutex);
    if(s.events) {
        if(std::fprintf(s.events,"%lld,%s,%llu,%llu,%lld,%d\n",(long long)NowNs(),kind,
            (unsigned long long)s.admitted.load(),(unsigned long long)frame,(long long)map,iterations)<0 || std::fflush(s.events))
            std::terminate(); // Never continue a run with a broken causal evidence stream.
    }
}
inline void Open(const char* path) {
    State& s=Shared();std::lock_guard<std::mutex> lock(s.mutex);
    if(s.events) throw std::runtime_error("audit already open");
    s.events=std::fopen(path,"wx");
    if(!s.events) throw std::runtime_error("audit file must be new");
    std::fputs("monotonic_ns,event,admitted_original_id,native_frame_id,map_id,iterations\n",s.events);std::fflush(s.events);
}
inline void Begin(uint64_t id){ Shared().admitted.store(id);PoseCalls()=0; }
inline void Pose(uint64_t frame){ ++PoseCalls();PoseFrame()=frame;Event("pose_begin",frame,-1); }
struct LocalScope {
    uint64_t frame;int64_t map;
    LocalScope(uint64_t f,int64_t m):frame(f),map(m){ ++Shared().ba_started;Event("local_ba_begin",frame,map); }
    ~LocalScope(){ Event("local_ba_end",frame,map);++Shared().ba_finished; }
};
struct Snapshot {
    int tracking_state=-1;int64_t map_id=-1;uint64_t map_generation=0,native_frame_id=0;
    unsigned pose_calls=0;uint64_t pose_frame_id=0;
    uint64_t ba_started=0,ba_finished=0;int mapping_queue=0;
};
} }
#endif
