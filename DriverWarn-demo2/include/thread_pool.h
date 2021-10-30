#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using namespace std;

#define DEFAULT_THREAD_NUM 5


namespace threadpool
{
#ifdef __cplusplus
extern "C" {
#endif

typedef enum _ThreadState {
    INIT,
    WAITING,
    IDEL,
    RUNNing
}ThreadState;

typedef enum _ThreadType {
    SOURCE_THREAD,
    CONVERT_THREAD,
    AI_THREAD
} ThreadType;

#ifdef __cplusplus
}
#endif

typedef struct _PoolConfig {
    int core_threads = 5;
    int max_threads = 16;
    int max_task_size = 16;
    std::chrono::seconds time_out = std::chrono::seconds(5);
    ThreadType thread_type =  ThreadType::CONVERT_THREAD;
} PoolConfig;

enum class ThreadFlag {kCore = 1, kCache = 2 };

using ThreadId = std::atomic<int>;
using ThreadPtr = std::shared_ptr<std::thread>;
using ThreadFlagAtomic = std::atomic<ThreadFlag>;
using ThreadStateAtomic = std::atomic<ThreadState>;


struct ExThread {
    std::chrono::system_clock::time_point start_times;
    bool core_thread;
    ThreadId id;
    ThreadPtr ptr;
    ThreadStateAtomic state;
    ThreadFlagAtomic flags;

    ExThread() {
        ptr = nullptr;
        id = 0;
        state = ThreadState::INIT;
    }
};

using ExThreadPtr = std::shared_ptr<ExThread>;

class ThreadPool {
public:
    ThreadPool(PoolConfig _config);
    template<class F, class... Args>
    auto AddTask(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();

private:
    PoolConfig config;
    // need to keep track of threads so we can join them
    std::vector< ExThreadPtr > threads;
    // the task queue
    std::queue< std::function<void()> > tasks;

    std::atomic<int> current_thread_num;
    std::atomic<int> current_tasks_num;
    ThreadId next_thread_id;
    
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    void AddThread(int , ThreadFlag);
    int GetNextThreadID() {
        return this->next_thread_id++;
    }
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(PoolConfig _config) : stop(false)
{
    this->config = _config;
    for (int i = 0; i < this->config.core_threads; i++)
        AddThread(i, ThreadFlag::kCore);        
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::AddTask(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;
    if(this->current_tasks_num.load() > this->config.max_threads){
        AddThread(this->GetNextThreadID(),ThreadFlag::kCache);
    }

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
        this->current_tasks_num++;
    }
    condition.notify_one();
    return res;
}

inline void ThreadPool::AddThread(int _id, ThreadFlag _core_thread)
{
    ExThreadPtr thread_ptr = std::make_shared<ExThread>();
    thread_ptr->id.store( _id);
    thread_ptr->state.store(ThreadState::INIT);
    thread_ptr->flags.store(_core_thread);
    thread_ptr->start_times = std::chrono::system_clock::now();
    auto func = [this,thread_ptr] (){
        for(;;)
        {
            std::function<void()> task;
            {
                //using code block for auto unlock when code finished.
                std::unique_lock<std::mutex> lock(this->queue_mutex);

                thread_ptr->state.store(ThreadState::WAITING);
                this->current_thread_num++;

                if (ThreadFlag::kCore == thread_ptr->flags.load()) {
                    //To core thread, allow block long time.
                    this->condition.wait(lock,
                        [this,thread_ptr]{ return this->stop || !this->tasks.empty() || 
                            thread_ptr->state.load() == ThreadState::IDEL; });
                } else {
                    //To no core thread,closed when time out.
                    this->condition.wait_for(lock,this->config.time_out,
                        [this,thread_ptr]{ return this->stop || !this->tasks.empty() || 
                            thread_ptr->state.load() == ThreadState::IDEL; });
                }
                thread_ptr->state.store(ThreadState::RUNNing);
                //no core thread auto kill when tasks is empty.
                if(((int)this->tasks.size() < this->config.max_threads) && (ThreadFlag::kCore != thread_ptr->flags.load())) {
                    this->current_thread_num--;
                    return;
                }
                if(this->tasks.empty()) {
                    if(ThreadFlag::kCore != thread_ptr->flags.load()) {
                        return;
                    } else {
                        //To core thread, allow block long time.
                        this->condition.wait(lock,
                            [this,thread_ptr]{ return this->stop || !this->tasks.empty() || 
                                thread_ptr->state.load() == ThreadState::IDEL; });
                    }
                }

                //close all threads when ThreadPool was stoped and tasks is empty 
                if(this->stop)
                    return;
                //remove first task when got first task in the tasks queue
                task = std::move(this->tasks.front());
                this->tasks.pop();
                this->current_tasks_num--;
            }

            task();
        }
    };
    //Loading of the thread to threads vector
    thread_ptr->ptr = std::make_shared<std::thread>(std::move(func));
    if (thread_ptr->ptr->joinable()) {
        thread_ptr->ptr->detach();
    }
    threads.emplace_back(thread_ptr);
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(ExThreadPtr &thread : threads)
        thread->ptr->join();
}



    
}// namespace threadpool

#endif