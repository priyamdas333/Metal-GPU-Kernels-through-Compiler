#include <metal_stdlib>
using namespace metal;
kernel void matmul(device const float* A [[buffer(0)]],
                   device const float* B [[buffer(1)]],
                   device float* C       [[buffer(2)]],
                   constant uint& N      [[buffer(3)]],
                   uint2 pos [[thread_position_in_grid]]) {
    float sum = 0.0f;
    for (uint k = 0; k < N; k++) {
        sum += A[pos.y * N + k] * B[k * N + pos.x];
    }
    C[pos.y * N + pos.x] = sum;
}
