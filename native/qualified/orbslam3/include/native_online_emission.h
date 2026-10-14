#ifndef SLAM_NATIVE_ONLINE_EMISSION_H
#define SLAM_NATIVE_ONLINE_EMISSION_H
#include <math.h>
#include <stdint.h>
enum OnlineOutcome { ONLINE_ID_ERROR=-1, ONLINE_INITIALIZED=0, ONLINE_TRACKED=1,
    ONLINE_HELD_STATE=2, ONLINE_HELD_MAP=3, ONLINE_HELD_INVALID=4 };
typedef struct {
    uint64_t rows;
    int initialized;
    int64_t map_id;
    uint64_t map_generation;
    double last_twc[12];
} OnlineLedger;
static inline OnlineLedger online_ledger_init(void) {
    OnlineLedger l={0,0,-1,0,{1,0,0,0,0,1,0,0,0,0,1,0}};
    return l;
}
/* Output convention: 3x4 row-major camera-to-world. No scale fitting or GT. */
static inline int online_rigid_inverse(const double t[12], double inverse[12]) {
    for(int i=0;i<12;i++) if(!isfinite(t[i])) return 0;
    for(int r=0;r<3;r++) for(int c=0;c<3;c++) {
        double dot=0;
        for(int k=0;k<3;k++) dot+=t[r*4+k]*t[c*4+k];
        if(fabs(dot-(r==c?1.0:0.0))>1e-4) return 0;
    }
    double det=t[0]*(t[5]*t[10]-t[6]*t[9])-t[1]*(t[4]*t[10]-t[6]*t[8])+t[2]*(t[4]*t[9]-t[5]*t[8]);
    if(fabs(det-1)>1e-4) return 0;
    for(int r=0;r<3;r++) {
        for(int c=0;c<3;c++) inverse[r*4+c]=t[c*4+r];
        inverse[r*4+3]=-(t[r]*t[3]+t[4+r]*t[7]+t[8+r]*t[11]);
        if(!isfinite(inverse[r*4+3])) return 0;
    }
    return 1;
}
/* ORB-SLAM3's pinned enum uses OK=2. All other states are explicit holds.
   Atlas resets cannot silently establish a new world origin. */
static inline int online_emit(OnlineLedger* l,uint64_t id,int state,int64_t map,uint64_t generation,
                              const double tcw[12],double twc[12]) {
    if(id!=l->rows) return ONLINE_ID_ERROR;
    int outcome=ONLINE_HELD_STATE;
    double candidate[12];
    if(state==2) {
        if(map<0 || generation==0 || (l->initialized && (map!=l->map_id || generation!=l->map_generation))) outcome=ONLINE_HELD_MAP;
        else if(!online_rigid_inverse(tcw,candidate)) outcome=ONLINE_HELD_INVALID;
        else {
            outcome=l->initialized?ONLINE_TRACKED:ONLINE_INITIALIZED;
            l->initialized=1;l->map_id=map;l->map_generation=generation;
            for(int i=0;i<12;i++) l->last_twc[i]=candidate[i];
        }
    }
    for(int i=0;i<12;i++) twc[i]=l->last_twc[i];
    ++l->rows;
    return outcome;
}
#endif
