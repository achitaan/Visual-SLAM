#ifndef SLAM_NATIVE_MAP_SNAPSHOT_H
#define SLAM_NATIVE_MAP_SNAPSHOT_H
#include "native_snapshot_mailbox.h"
#include <array>
#include <cstdint>
#include <vector>
namespace ORB_SLAM3 {
struct OnlineMapContext {
    uint64_t original_id;
    uint64_t native_frame_id;
    int64_t map_id;
    uint64_t map_generation;
    int64_t request_ns;
};
struct OnlineMapKeyframe {
    uint64_t id,native_frame_id;
    double timestamp;
    std::array<double,16> pose_T_wc;
};
struct OnlineMapPoint { uint64_t id; std::array<double,3> world; };
struct OnlineMapObservation { uint64_t keyframe,point; int feature; double u,v; };
struct OnlineMapSnapshot {
    OnlineMapContext request;
    int64_t capture_ns,completion_ns;
    int map_revision;
    uint64_t observations_outside_snapshot=0;
    std::vector<OnlineMapKeyframe> keyframes;
    std::vector<OnlineMapPoint> points;
    std::vector<OnlineMapObservation> observations;
};
using OnlineMapMailbox=OnlineSnapshotMailbox<OnlineMapContext,OnlineMapSnapshot>;
void WriteOnlineMapSnapshot(const OnlineMapSnapshot&,const char* path);
}
#endif
