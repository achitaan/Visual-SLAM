// Owned online map copy. Invoked only by LocalMapping at a normal safe boundary.
#include "System.h"
#include "native_map_snapshot.h"
#include <unordered_set>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace ORB_SLAM3 {
bool LocalMapping::RequestOnlineMap(const OnlineMapContext& context,double seconds,OnlineMapSnapshot& output) {
    if(mbMonocular || mbInertial || !std::isfinite(seconds) || seconds<=0 || seconds>5)
        throw std::invalid_argument("snapshot requires bounded visual stereo request");
    auto request=mOnlineMapMailbox.submit(context);
    return mOnlineMapMailbox.wait(request,std::chrono::duration<double>(seconds),output);
}
void LocalMapping::CloseOnlineSnapshots(){mOnlineMapMailbox.close();}

void LocalMapping::ServiceOnlineSnapshot() {
    mOnlineMapMailbox.service([this](const OnlineMapContext& context) {
        Map* map=mpAtlas->GetCurrentMap();
        if(!map || context.map_id<0 || static_cast<int64_t>(map->GetId())!=context.map_id)
            throw std::runtime_error("snapshot primary map changed");
        // Caller awaits after TrackStereo and before next image/reset. Mapper is
        // this thread, loops are disabled, and IMU/GBA paths are unsupported.
        std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);
        if(context.map_generation==0 || map->GetOriginGeneration()!=context.map_generation)
            throw std::runtime_error("snapshot coordinate generation changed");
        OnlineMapSnapshot out;out.request=context;out.capture_ns=OnlineAudit::NowNs();
        out.map_revision=map->GetMapChangeIndex();
        std::unordered_set<KeyFrame*> included;
        for(KeyFrame* k:map->GetAllKeyFrames()) {
            if(!k || k->isBad())continue;
            if(k->GetMap()!=map || k->NLeft!=-1 || k->mpCamera2)
                throw std::runtime_error("unsupported snapshot keyframe camera/map");
            OnlineMapKeyframe value;value.id=k->mnId;value.native_frame_id=k->mnFrameId;value.timestamp=k->mTimeStamp;
            const Eigen::Matrix4f twc=k->GetPoseInverse().matrix(); // owned value
            if(!twc.allFinite() || !std::isfinite(value.timestamp))throw std::runtime_error("nonfinite keyframe");
            for(int r=0;r<4;r++)for(int c=0;c<4;c++)value.pose_T_wc[4*r+c]=twc(r,c);
            out.keyframes.push_back(value);included.insert(k);
        }
        for(MapPoint* point:map->GetAllMapPoints()) {
            if(!point || point->isBad())continue;
            if(point->GetMap()!=map)throw std::runtime_error("point belongs to another map");
            const Eigen::Vector3f position=point->GetWorldPos();
            if(!position.allFinite())throw std::runtime_error("nonfinite landmark");
            OnlineMapPoint value;value.id=point->mnId;
            for(int j=0;j<3;j++)value.world[j]=position[j];
            out.points.push_back(value);
            for(const auto& observation:point->GetObservations()) {
                KeyFrame* k=observation.first;
                if(!included.count(k)){++out.observations_outside_snapshot;continue;}
                const int left=std::get<0>(observation.second),right=std::get<1>(observation.second);
                if(left<0 || right!=-1 || static_cast<size_t>(left)>=k->mvKeysUn.size() || k->GetMapPoint(left)!=point)
                    throw std::runtime_error("inconsistent reciprocal observation");
                const cv::Point2f pixel=k->mvKeysUn[left].pt;
                if(!std::isfinite(pixel.x) || !std::isfinite(pixel.y))throw std::runtime_error("nonfinite observation");
                out.observations.push_back({k->mnId,point->mnId,left,pixel.x,pixel.y});
            }
        }
        if(out.keyframes.empty() || out.points.empty() || out.observations.empty())
            throw std::runtime_error("no initialized connected snapshot");
        out.completion_ns=OnlineAudit::NowNs();
        OnlineAudit::Event("map_snapshot_copy",context.native_frame_id,context.map_id);
        return out;
    });
}

bool System::CaptureOnlineMap(uint64_t original_id,int64_t primary_map,uint64_t primary_generation,double seconds,OnlineMapSnapshot& out) {
    if(mSensor!=STEREO || mpViewer || !mpLoopCloser->OnlineSnapshotsAllowed())
        throw std::runtime_error("snapshot supports headless visual stereo with loops off only");
    OnlineMapContext request{original_id,mpTracker->mCurrentFrame.mnId,primary_map,primary_generation,OnlineAudit::NowNs()};
    return mpLocalMapper->RequestOnlineMap(request,seconds,out);
}

void WriteOnlineMapSnapshot(const OnlineMapSnapshot& s,const char* path) {
    FILE* f=std::fopen(path,"wx");if(!f)throw std::runtime_error("snapshot output must be new");
    std::fprintf(f,"{\"version\":2,\"original_id\":%llu,\"map_id\":%lld,\"map_generation\":%llu,\"request_ns\":%lld,\"capture_ns\":%lld,\"completion_ns\":%lld,\"map_revision\":%d,\"observations_outside_snapshot\":%llu,\"keyframes\":[",
        (unsigned long long)s.request.original_id,(long long)s.request.map_id,(unsigned long long)s.request.map_generation,(long long)s.request.request_ns,(long long)s.capture_ns,(long long)s.completion_ns,s.map_revision,(unsigned long long)s.observations_outside_snapshot);
    bool first=true;
    for(const auto& k:s.keyframes) {
        std::fprintf(f,"%s{\"id\":%llu,\"native_frame_id\":%llu,\"timestamp\":%.17g,\"pose_T_wc\":[",first?"":",",(unsigned long long)k.id,(unsigned long long)k.native_frame_id,k.timestamp);first=false;
        for(int j=0;j<16;j++)std::fprintf(f,"%s%.17g",j?",":"",k.pose_T_wc[j]);std::fprintf(f,"]}");
    }
    std::fprintf(f,"],\"points\":[");first=true;
    for(const auto& p:s.points){std::fprintf(f,"%s[%llu,%.17g,%.17g,%.17g]",first?"":",",(unsigned long long)p.id,p.world[0],p.world[1],p.world[2]);first=false;}
    std::fprintf(f,"],\"observations\":[");first=true;
    for(const auto& o:s.observations){std::fprintf(f,"%s[%llu,%llu,%d,%.17g,%.17g]",first?"":",",(unsigned long long)o.keyframe,(unsigned long long)o.point,o.feature,o.u,o.v);first=false;}
    std::fprintf(f,"]}\n");const bool bad=std::ferror(f);const int closed=std::fclose(f);
    if(bad || closed)throw std::runtime_error("snapshot output failed");
}
}
