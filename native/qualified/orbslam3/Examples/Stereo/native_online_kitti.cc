// Immutable online stereo runner for pinned ORB-SLAM3. No GT or retrospective export.
#include "System.h"
#include "native_online_emission.h"
#include "native_seed.h"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static FILE* new_file(const std::string& path) {
    FILE* f=std::fopen(path.c_str(),"wx");
    if(!f) throw std::runtime_error("output must be new: "+path);
    return f;
}
int main(int argc,char** argv) {
    if(argc!=8) { std::fprintf(stderr,"usage: native_online_kitti VOC CONFIG SEQUENCE EXISTING_EMPTY_OUTPUT_DIR FRAME_COUNT SEED SNAPSHOT_INTERVAL\n");return 2; }
    try {
        const int seed=online_parse_seed(argv[6]);
        const int snapshot_interval=online_parse_seed(argv[7]);if(snapshot_interval<1)throw std::runtime_error("positive snapshot interval required");
        const int count=std::stoi(argv[5]);if(count<1) throw std::runtime_error("positive count required");
        std::ifstream times(std::string(argv[3])+"/times.txt");
        if(!times) throw std::runtime_error("timestamps missing");
        std::vector<double> stamp;std::string line;
        while(std::getline(times,line) && (int)stamp.size()<count) {
            std::istringstream row(line);double t;std::string extra;
            if(!(row>>t) || (row>>extra) || !std::isfinite(t) || (!stamp.empty() && t<=stamp.back())) throw std::runtime_error("invalid/nonmonotonic timestamp");
            stamp.push_back(t);
        }
        if((int)stamp.size()!=count) throw std::runtime_error("incomplete timestamps");
        cv::FileStorage config(argv[2],cv::FileStorage::READ);
        if(!config.isOpened() || config["loopClosing"].empty() || (int)config["loopClosing"]!=0 || !config["System.SaveAtlasToFile"].empty() || !config["System.LoadAtlasFromFile"].empty()) throw std::runtime_error("require loops off and no atlas persistence");
        FILE* poses=new_file(std::string(argv[4])+"/online_poses.txt");
        FILE* rows=new_file(std::string(argv[4])+"/online_frames.csv");
        FILE* end=new_file(std::string(argv[4])+"/lifecycle.csv");
        std::fprintf(rows,"original_id,timestamp,start_ns,completion_ns,state,map_id,outcome,pose_calls,pose_frame_id,native_frame_id,ba_started,ba_finished,mapping_queue,map_generation,anchor_generation\n");
        cv::setNumThreads(1);online_initialize_seed(seed);
        FILE* seed_log=new_file(std::string(argv[4])+"/seed.csv");
        std::fprintf(seed_log,"requested_seed,opencv_state_before_system,native_seed_once\n%d,%llu,1\n",seed,(unsigned long long)cv::theRNG().state);
        if(std::fclose(seed_log)) throw std::runtime_error("seed record write failure");
        ORB_SLAM3::System slam(argv[1],argv[2],ORB_SLAM3::System::STEREO,false);
        if(slam.GetImageScale()!=1.0f) throw std::runtime_error("input resizing prohibited");
        slam.OpenOnlineAudit((std::string(argv[4])+"/optimizer_events.csv").c_str());
        OnlineLedger ledger=online_ledger_init();bool input_ok=true,observed=false;
        for(int id=0;id<count;id++) {
            const auto start=ORB_SLAM3::OnlineAudit::NowNs();
            std::ostringstream filename;filename<<std::setfill('0')<<std::setw(6)<<id<<".png";
            cv::Mat left=cv::imread(std::string(argv[3])+"/image_0/"+filename.str(),cv::IMREAD_UNCHANGED);
            cv::Mat right=cv::imread(std::string(argv[3])+"/image_1/"+filename.str(),cv::IMREAD_UNCHANGED);
            slam.BeginOnlineFrame(id);
            double tcw[12]={1,0,0,0,0,1,0,0,0,0,1,0};
            bool valid=!left.empty() && !right.empty() && left.size()==right.size() && left.type()==CV_8UC1 && right.type()==CV_8UC1;
            if(valid) {
                const auto current=slam.TrackStereo(left,right,stamp[id]).matrix();
                for(int r=0;r<3;r++) for(int c=0;c<4;c++) tcw[r*4+c]=current(r,c);
                observed=true;
            } else input_ok=false;
            auto s=observed?slam.GetOnlineSnapshot():ORB_SLAM3::OnlineAudit::Snapshot{};
            const auto completion=ORB_SLAM3::OnlineAudit::NowNs();
            if(!valid) s.tracking_state=-2;
            double twc[12];int outcome=online_emit(&ledger,id,s.tracking_state,s.map_id,s.map_generation,tcw,twc);
            if(outcome<0) throw std::runtime_error("original-ID coverage failure");
            for(int k=0;k<12;k++) std::fprintf(poses,k==11?"%.17g\n":"%.17g ",twc[k]);
            std::fprintf(rows,"%d,%.17g,%lld,%lld,%d,%lld,%d,%u,%llu,%llu,%llu,%llu,%d,%llu,%llu\n",id,stamp[id],(long long)start,(long long)completion,s.tracking_state,(long long)s.map_id,outcome,s.pose_calls,(unsigned long long)s.pose_frame_id,(unsigned long long)s.native_frame_id,(unsigned long long)s.ba_started,(unsigned long long)s.ba_finished,s.mapping_queue,(unsigned long long)s.map_generation,(unsigned long long)ledger.map_generation);
            if(std::fflush(poses)||std::fflush(rows)||std::ferror(poses)||std::ferror(rows)) throw std::runtime_error("emission write failure");
            if((id>0 && id%snapshot_interval==0) || id==count-1) {
                ORB_SLAM3::OnlineMapSnapshot snapshot;
                if(!slam.CaptureOnlineMap(id,ledger.map_id,ledger.map_generation,2.0,snapshot))throw std::runtime_error("snapshot timeout or closed worker");
                const std::string path=std::string(argv[4])+"/snapshot-"+std::to_string(id)+".json";
                ORB_SLAM3::WriteOnlineMapSnapshot(snapshot,path.c_str());
            }
        }
        // Close immutable outputs BEFORE shutdown. Later worker completion cannot rewrite them.
        if(std::fclose(poses)||std::fclose(rows)) throw std::runtime_error("emission close failure");
        const auto before=observed?slam.GetOnlineSnapshot():ORB_SLAM3::OnlineAudit::Snapshot{};
        const auto shutdown_start=ORB_SLAM3::OnlineAudit::NowNs();
        bool joined=slam.ShutdownOnline(30.0);
        const auto after=observed?slam.GetOnlineSnapshot():ORB_SLAM3::OnlineAudit::Snapshot{};
        std::fprintf(end,"rows,input_complete,workers_joined,shutdown_start_ns,shutdown_end_ns,ba_finished_before,ba_finished_after,queue_before,queue_after\n%llu,%d,%d,%lld,%lld,%llu,%llu,%d,%d\n",(unsigned long long)ledger.rows,(int)input_ok,(int)joined,(long long)shutdown_start,(long long)ORB_SLAM3::OnlineAudit::NowNs(),(unsigned long long)before.ba_finished,(unsigned long long)after.ba_finished,before.mapping_queue,after.mapping_queue);
        std::fflush(end);std::fclose(end);
        if(!joined) std::_Exit(3); // bounded failed lifecycle, not acceptance
        return input_ok?0:4;
    } catch(const std::exception& e) { std::fprintf(stderr,"incomplete run: %s\n",e.what());return 5; }
}
