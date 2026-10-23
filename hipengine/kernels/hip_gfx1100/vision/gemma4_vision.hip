// Gemma 4 vision primitives. Equations follow Gemma4VisionModel; all operands
// are raw FP32 device pointers. Linear and attention contractions use rocBLAS.
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

__global__ void gemma4_vision_norm_kernel(const float* x, const float* w, float* y,
                                        int rows, int width, float eps) {
    __shared__ float partial[256];
    __shared__ float inv;
    const int row = blockIdx.x;
    float sum = 0;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        float v = x[row * width + c];
        sum += v * v;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (!threadIdx.x) inv = rsqrtf(partial[0] / width + eps);
    __syncthreads();
    for (int c = threadIdx.x; c < width; c += blockDim.x)
        y[row * width + c] = x[row * width + c] * inv * (w ? w[c] : 1.0f);
}

// Separate add, GELU, multiply, clamp and scalar-scale primitives.
__global__ void gemma4_vision_elementwise_kernel(const float* x, const float* b,
                                               float* y, int n, int op, float lo, float hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (op == 0) v += b[i];
    if (op == 1) v = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f*v*v*v)));
    if (op == 2) v *= b[i];
    if (op == 3) v = fminf(hi, fmaxf(lo, v));
    if (op == 4) v *= lo;
    y[i] = v;
}

// Token-major heads -> head-major heads, with optional independent X/Y NEOX RoPE.
__global__ void gemma4_vision_heads_kernel(const float* x, const int32_t* pos,
                                         float* y, int tokens, int heads, int dim,
                                         float theta, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * heads * dim) return;
    int c = i % dim, h = (i / dim) % heads, t = i / (dim * heads);
    if (mode == 2) {
        y[i] = x[(h * tokens + t) * dim + c];
        return;
    }
    float v = x[i];
    if (mode == 1) {
        int spatial = dim / 2, quarter = dim / 4;
        int axis = c / spatial, local = c % spatial, pair = local % quarter;
        int other = axis * spatial + (local < quarter ? local + quarter : local - quarter);
        float angle = pos[2*t + axis] * powf(theta, -2.0f * pair / spatial);
        v = v * cosf(angle) + (local < quarter ? -1.0f : 1.0f) *
            x[(t * heads + h) * dim + other] * sinf(angle);
    }
    y[(h * tokens + t) * dim + c] = v;
}

__global__ void gemma4_vision_softmax_kernel(float* x, int cols) {
    __shared__ float reduce[256];
    __shared__ float maximum, inverse;
    int row = blockIdx.x;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, x[row*cols+c]);
    reduce[threadIdx.x] = m;
    __syncthreads();
    for (int s=128; s; s>>=1) {
        if (threadIdx.x < s) reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x+s]);
        __syncthreads();
    }
    if (!threadIdx.x) maximum = reduce[0];
    __syncthreads();
    float sum = 0;
    for (int c=threadIdx.x; c<cols; c+=blockDim.x) sum += expf(x[row*cols+c]-maximum);
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (int s=128; s; s>>=1) {
        if (threadIdx.x < s) reduce[threadIdx.x] += reduce[threadIdx.x+s];
        __syncthreads();
    }
    if (!threadIdx.x) inverse = 1.0f / reduce[0];
    __syncthreads();
    for (int c=threadIdx.x; c<cols; c+=blockDim.x) x[row*cols+c] = expf(x[row*cols+c]-maximum)*inverse;
}

__global__ void gemma4_vision_pool_kernel(const float* x, const float* bias,
                                        const float* scale, float* y,
                                        int gh, int gw, int hidden, int merge) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int rows = (gh/merge)*(gw/merge);
    if (i >= rows*hidden) return;
    int c=i%hidden, row=i/hidden, py=row/(gw/merge), px=row%(gw/merge);
    float sum=0;
    for (int dy=0; dy<merge; ++dy)
        for (int dx=0; dx<merge; ++dx)
            sum += x[((py*merge+dy)*gw+px*merge+dx)*hidden+c];
    float v = sum / (merge*merge) * sqrtf((float)hidden);
    if (bias) v -= bias[c];
    if (scale) v *= scale[c];
    y[i] = v;
}

extern "C" int hipengine_gemma4_vision_norm(const float* x,const float* w,float* y,
                                          int rows,int width,float eps,hipStream_t stream) {
    if (!x || !y || rows<=0 || width<=0 || eps<=0) return hipErrorInvalidValue;
    gemma4_vision_norm_kernel<<<rows,256,0,stream>>>(x,w,y,rows,width,eps);
    return hipGetLastError();
}
extern "C" int hipengine_gemma4_vision_elementwise(const float* x,const float* b,float* y,
    int n,int op,float lo,float hi,hipStream_t stream) {
    if (!x || !y || n<=0 || op<0 || op>4 || ((op==0 || op==2) && !b)) return hipErrorInvalidValue;
    gemma4_vision_elementwise_kernel<<<(n+255)/256,256,0,stream>>>(x,b,y,n,op,lo,hi);
    return hipGetLastError();
}
extern "C" int hipengine_gemma4_vision_heads(const float* x,const int32_t* p,float* y,
    int tokens,int heads,int dim,float theta,int mode,hipStream_t stream) {
    if (!x || !y || tokens<=0 || heads<=0 || dim<=0 || mode<0 || mode>2 ||
        (mode==1 && (!p || dim%4 || theta<=0))) return hipErrorInvalidValue;
    int n=tokens*heads*dim;
    gemma4_vision_heads_kernel<<<(n+255)/256,256,0,stream>>>(x,p,y,tokens,heads,dim,theta,mode);
    return hipGetLastError();
}
extern "C" int hipengine_gemma4_vision_softmax(float* x,int rows,int cols,hipStream_t stream) {
    if (!x || rows<=0 || cols<=0) return hipErrorInvalidValue;
    gemma4_vision_softmax_kernel<<<rows,256,0,stream>>>(x,cols);
    return hipGetLastError();
}
extern "C" int hipengine_gemma4_vision_pool(const float* x,const float* b,const float* s,float* y,
    int gh,int gw,int hidden,int merge,hipStream_t stream) {
    if (!x || !y || gh<=0 || gw<=0 || hidden<=0 || merge<=0 || gh%merge || gw%merge)
        return hipErrorInvalidValue;
    int n=(gh/merge)*(gw/merge)*hidden;
    gemma4_vision_pool_kernel<<<(n+255)/256,256,0,stream>>>(x,b,s,y,gh,gw,hidden,merge);
    return hipGetLastError();
}
