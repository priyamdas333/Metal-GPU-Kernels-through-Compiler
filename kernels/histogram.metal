#include <metal_stdlib>
using namespace metal;
kernel void histogram(device const float* data [[buffer(0)]],
                      device atomic_uint* bins [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    uint bin = min(uint(data[i] * 16.0f), 15u);     // data is in [0, 1)
    atomic_fetch_add_explicit(&bins[bin], 1, memory_order_relaxed);
}
