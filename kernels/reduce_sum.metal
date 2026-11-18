#include <metal_stdlib>
using namespace metal;
kernel void reduce_sum(device const float* in [[buffer(0)]],
                       device float* out      [[buffer(1)]],
                       uint gid   [[thread_position_in_grid]],
                       uint tid   [[thread_position_in_threadgroup]],
                       uint group [[threadgroup_position_in_grid]]) {
    threadgroup float shared[256];
    shared[tid] = in[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride /= 2) {   // tree reduction
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[group] = shared[0];
    }
}
