#ifndef SLAM_NATIVE_SNAPSHOT_MAILBOX_H
#define SLAM_NATIVE_SNAPSHOT_MAILBOX_H
#include <chrono>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

// One owning application thread submits/waits; the mapper services between
// iterations. The owner must join that worker before destroying this mailbox.
// Timeout bounds waiting, not an already executing copy. Such a late copy owns
// its storage, cannot publish a cancelled result, and retains the single slot
// until it finishes. This class never stops/interrupts a solver or worker.
template<class Context,class Result> class OnlineSnapshotMailbox {
    struct Job {
        explicit Job(Context c):context(std::move(c)){}
        const Context context;
        std::mutex mutex;
        std::condition_variable ready;
        bool started=false,done=false,cancelled=false;
        std::unique_ptr<Result> result;
        std::exception_ptr error;
    };
public:
    using Request=std::shared_ptr<Job>;
    OnlineSnapshotMailbox()=default;
    OnlineSnapshotMailbox(const OnlineSnapshotMailbox&)=delete;
    OnlineSnapshotMailbox& operator=(const OnlineSnapshotMailbox&)=delete;
    ~OnlineSnapshotMailbox(){close();}

    Request submit(Context context) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(closed_ || slot_)return {};
        slot_=std::make_shared<Job>(std::move(context));return slot_;
    }

    template<class Rep,class Period>
    bool wait(const Request& request,std::chrono::duration<Rep,Period> duration,Result& output) {
        if(duration<duration.zero())throw std::invalid_argument("negative snapshot deadline");
        if(!request)return false;
        std::unique_lock<std::mutex> lock(request->mutex);
        if(!request->ready.wait_for(lock,duration,[&]{return request->done || request->cancelled;})) {
            request->cancelled=true;return false;
        }
        if(request->cancelled)return false;
        if(request->error)std::rethrow_exception(request->error);
        if(!request->done || !request->result)throw std::logic_error("incomplete snapshot result");
        output=*request->result;return true;
    }

    template<class Copy> bool service(Copy copy) {
        Request request;
        {
            // Lock order is always mailbox then job; completion releases the
            // job lock before taking the mailbox lock again.
            std::lock_guard<std::mutex> lock(mutex_);
            request=slot_;if(!request)return false;
            std::lock_guard<std::mutex> job_lock(request->mutex);
            if(request->started)return false;
            if(request->cancelled || closed_) {slot_.reset();return false;}
            request->started=true;
        }
        std::unique_ptr<Result> result;std::exception_ptr error;
        try {result.reset(new Result(copy(request->context)));}
        catch(...) {error=std::current_exception();}
        {
            std::lock_guard<std::mutex> lock(request->mutex);
            if(!request->cancelled) {request->result=std::move(result);request->error=error;}
            request->done=true;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if(slot_==request)slot_.reset();
        }
        request->ready.notify_all();return true;
    }

    void close() {
        Request request;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_=true;request=slot_;
            if(request) {
                std::lock_guard<std::mutex> job_lock(request->mutex);
                if(!request->done)request->cancelled=true;
            }
        }
        if(request)request->ready.notify_all();
    }
private:
    std::mutex mutex_;
    Request slot_;
    bool closed_=false;
};
#endif
