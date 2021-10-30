#include <iostream>
#include <vector>
#include <chrono>
#include "thread_pool.h"

using namespace threadpool;

int main (int argc, char **argv)
{
    PoolConfig pool_conf;
    ThreadPool pool(pool_conf); 
    std::vector< std::shared_future<int> > results; 

    std::cout << "Thread Pool Init finished!" << std::endl;
    while(true) {
        for(int i = 0; i < 8; ++i) {    
            results.emplace_back(
                pool.AddTask([i] {
                    std::cout << "hello " << i << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    std::cout << "world " << i << std::endl;
                    return i*i;
                })
            );
        }

        for(auto && result: results)    //一次取出保存在results中的异步结果
            std::cout << result.get() << ' ';
        std::cout << std::endl;
    }
    pool.~ThreadPool();
    return 0;

}