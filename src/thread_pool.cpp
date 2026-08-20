#include <kern/thread_pool.hpp>

#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace kern {
    ThreadPool::ThreadPool(const size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
#if defined(__APPLE__)
                // Bias the scheduler toward P-cores: kernel workers carry no
                // QoS by default and macOS is free to run them on E-cores,
                // which shows up as large per-iteration jitter.
                pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ThreadPool::~ThreadPool() {
        stop = true;
        condition.notify_all();
    }

    void ThreadPool::enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::move(task));
        }
        condition.notify_one();
    }
}
