#include "core/platform/threadpool.h"
#include "core/session/onnxruntime_c_api.h"
#include <memory>
#include <string>

namespace onnxruntime {
namespace concurrency {

struct ThreadPoolOptions{
    //0: Use default setting. (All the physical cores or half of the logical cores)
    //1: Don't create thread pool
    //n: Create a thread pool with n threads
    int thread_pool_size = 0;
    //If it is true and thread_pool_size = 0, populate the thread affinity information in ThreadOptions. 
    //Otherwise if the thread_options has affinity information, we'll use it and set it.
    //In the other case, don't set affinity
    bool auto_set_affinity = false;
    bool allow_spinning = true;
    ThreadOptions thread_options;
    const ORTCHAR_T* name = nullptr;
};
std::unique_ptr<ThreadPool> CreateThreadPool(Env* env, ThreadPoolOptions options,
                                             Eigen::Allocator* allocator = nullptr);
}  // namespace concurrency
}  // namespace onnxruntime