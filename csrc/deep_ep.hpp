#pragma once

// Forcibly disable NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>
#include <tuple>
#include <vector>
#include <future>
#include <chrono>

#include "config.hpp"
#include "event.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_ep_cpp
#endif

namespace deep_ep {

template <typename T>
class FutureWrapper {
public:
    FutureWrapper(std::future<T>&& fut) : future_(std::move(fut)) {}

    T get() {
        return future_.get();
    }

    bool ready() const {
        return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    FutureWrapper(const FutureWrapper&) = delete;
    FutureWrapper& operator=(const FutureWrapper&) = delete;
    FutureWrapper(FutureWrapper&&) = default;
    FutureWrapper& operator=(FutureWrapper&&) = default;

private:
    std::future<T> future_;
};

template <typename Func, typename... Args>
auto async_execute(Func&& func, Args&&... args) 
    -> FutureWrapper<std::invoke_result_t<Func&&, Args&&...>> {
    
    using ReturnType = std::invoke_result_t<Func&&, Args&&...>;
    
    std::future<ReturnType> future_result =
        std::async(std::launch::async, std::forward<Func>(func), std::forward<Args>(args)...);
        
    return FutureWrapper<ReturnType>(std::move(future_result));
}

struct Buffer {
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

private:
    // Low-latency mode buffer
    int low_latency_buffer_idx = 0;
    bool low_latency_mode = false;

    // NVLink Buffer
    int64_t num_nvl_bytes;
    void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;

    // NVSHMEM Buffer
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    // Device info and communication
    int device_id;
    int rank, rdma_rank, nvl_rank;
    int num_ranks, num_rdma_ranks, num_nvl_ranks;
    cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];

    // Stream for communication
    at::cuda::CUDAStream comm_stream;

    // After IPC/NVSHMEM synchronization, this flag will be true
    bool available = false;

    // Task fifo
    int head = 0;
    int* task_fifo_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    int** task_fifo_ptrs_gpu = nullptr;

    // Workspace
    void* workspace = nullptr;

    // Host-side MoE info
    volatile int* moe_recv_counter = nullptr;
    int* moe_recv_counter_mapped = nullptr;

    // Host-side expert-level MoE info
    volatile int* moe_recv_expert_counter = nullptr;
    int* moe_recv_expert_counter_mapped = nullptr;

    // Host-side RDMA-level MoE info
    volatile int* moe_recv_rdma_counter = nullptr;
    int* moe_recv_rdma_counter_mapped = nullptr;

private:
    void move_fifo_slots(int num_slots = 1);

public:
    Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode);

    ~Buffer() noexcept(false);

    bool is_available() const;

    bool is_internode_available() const;

    int get_num_rdma_ranks() const;

    int get_rdma_rank() const;

    int get_root_rdma_rank(bool global) const;

    int get_local_device_id() const;

    pybind11::bytearray get_local_ipc_handle() const;

    pybind11::bytearray get_local_nvshmem_unique_id() const;

    torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const;

    void sync(const std::vector<int>& device_ids, const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles, const std::optional<pybind11::bytearray>& root_unique_id_opt);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
    get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts, std::optional<EventHandle>& previous_event,
                        bool async, bool allocate_on_comm_stream);

    inline auto
    get_dispatch_layout_async(const torch::Tensor& topk_idx, int num_experts, std::optional<EventHandle>& previous_event,
                               bool async, bool allocate_on_comm_stream)
    {
        return async_execute(
            [=]() {
                return get_dispatch_layout(topk_idx, num_experts, previous_event, async, allocate_on_comm_stream);
            }
        );
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
    intranode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank, const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens, const std::optional<torch::Tensor>& cached_rank_prefix_matrix, const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                       int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

    inline auto 
    intranode_dispatch_async(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                            const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                            const std::optional<torch::Tensor>& num_tokens_per_rank, const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                            int cached_num_recv_tokens, const std::optional<torch::Tensor>& cached_rank_prefix_matrix, const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                            int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) 
    {
        return async_execute(
            [=]() {
                return intranode_dispatch(x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, cached_num_recv_tokens, cached_rank_prefix_matrix, cached_channel_prefix_matrix, expert_alignment, config, previous_event, async, allocate_on_comm_stream);
            }
        );
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    intranode_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                      const torch::Tensor& src_idx, const torch::Tensor& rank_prefix_matrix, const torch::Tensor& channel_prefix_matrix,
                      const torch::Tensor& send_head, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

    inline auto
    intranode_combine_async(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                           const torch::Tensor& src_idx, const torch::Tensor& rank_prefix_matrix, const torch::Tensor& channel_prefix_matrix,
                           const torch::Tensor& send_head, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream)
    {
        return async_execute(
            [=]() {
                return intranode_combine(x, topk_weights, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head, config, previous_event, async, allocate_on_comm_stream);
            }
        );
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<EventHandle>>
    internode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank, const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                       const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
                       const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                       const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                       int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

    inline auto
    internode_dispatch_async(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                            const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                            const std::optional<torch::Tensor>& num_tokens_per_rank, const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                            const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                            int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
                            const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                            const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                            int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream)
    {
        return async_execute(
            [=]() {
                return internode_dispatch(x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert, cached_num_recv_tokens, cached_num_rdma_recv_tokens, cached_rdma_channel_prefix_matrix, cached_recv_rdma_rank_prefix_sum, cached_gbl_channel_prefix_matrix, cached_recv_gbl_rank_prefix_sum, expert_alignment, config, previous_event, async, allocate_on_comm_stream);
            }
        );
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    internode_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                      const torch::Tensor& src_meta, const torch::Tensor& is_combined_token_in_rank,
                      const torch::Tensor& rdma_channel_prefix_matrix, const torch::Tensor& rdma_rank_prefix_sum, const torch::Tensor& gbl_channel_prefix_matrix,
                      const torch::Tensor& combined_rdma_head, const torch::Tensor& combined_nvl_head,
                      const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

    inline auto
    internode_combine_async(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                           const torch::Tensor& src_meta, const torch::Tensor& is_combined_token_in_rank,
                           const torch::Tensor& rdma_channel_prefix_matrix, const torch::Tensor& rdma_rank_prefix_sum, const torch::Tensor& gbl_channel_prefix_matrix,
                           const torch::Tensor& combined_rdma_head, const torch::Tensor& combined_nvl_head,
                           const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream)
    {
        return async_execute(
            [=]() {
                return internode_combine(x, topk_weights, src_meta, is_combined_token_in_rank, rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix, combined_rdma_head, combined_nvl_head, config, previous_event, async, allocate_on_comm_stream);
            }
        );
    }

    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
    low_latency_dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx,
                         int num_max_dispatch_tokens_per_rank, int num_experts,
                         bool use_fp8, bool async, bool return_recv_hook);

    inline auto
    low_latency_dispatch_async(const torch::Tensor& x, const torch::Tensor& topk_idx,
                              int num_max_dispatch_tokens_per_rank, int num_experts,
                              bool use_fp8, bool async, bool return_recv_hook)
    {
        return async_execute(
            [=]() {
                return low_latency_dispatch(x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts, use_fp8, async, return_recv_hook);
            }
        );
    }

    std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
    low_latency_combine(const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                        const torch::Tensor& src_info, const torch::Tensor& layout_range,
                        int num_max_dispatch_tokens_per_rank, int num_experts,
                        bool zero_copy, bool async, bool return_recv_hook,
                        const std::optional<torch::Tensor>& out = std::nullopt);

    inline auto
    low_latency_combine_async(const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                             const torch::Tensor& src_info, const torch::Tensor& layout_range,
                             int num_max_dispatch_tokens_per_rank, int num_experts,
                             bool zero_copy, bool async, bool return_recv_hook,
                             const std::optional<torch::Tensor>& out = std::nullopt)
    {
        return async_execute(
            [=]() {
                return low_latency_combine(x, topk_idx, topk_weights, src_info, layout_range, num_max_dispatch_tokens_per_rank, num_experts, zero_copy, async, return_recv_hook, out);
            }
        );
    }

    torch::Tensor
    get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const;

};

} // namespace deep_ep
