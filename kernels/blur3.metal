#include <metal_stdlib>
using namespace metal;
kernel void blur3(device const float* in [[buffer(0)]],
                  device float* out      [[buffer(1)]],
                  constant uint& n       [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i == 0 || i >= n - 1) {      // edges: just copy
        out[i] = in[i];
        return;
    }
    out[i] = (in[i - 1] + in[i] + in[i + 1]) / 3.0f;
}
