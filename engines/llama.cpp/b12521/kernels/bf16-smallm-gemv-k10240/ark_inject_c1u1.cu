#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_bf16.h>
struct __align__(4) tg_bfloat162 { nv_bfloat16 x, y; }; __device__ tg_bfloat162 make_tg_bfloat162(nv_bfloat16 x, nv_bfloat16 y) { tg_bfloat162 r={x, y}; return r; }
extern "C" __global__ void __launch_bounds__(256) ark_inject_c1u1(float* data0_16, float* data1_40960, nv_bfloat16* data2_40960) {
  asm volatile("griddepcontrol.wait;":::"memory");
  int gidx0 = blockIdx.x; /* 4 */
  int gidx1 = blockIdx.y; /* 4 */
  int lidx0 = threadIdx.x; /* 256 */
  float buf0;
  buf0 = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 20; Ridx0++) {
    int alu1 = ((lidx0<<1)+(Ridx0<<9));
    tg_bfloat162 val0 = (*((tg_bfloat162*)((data2_40960+(alu1+(gidx0*10240))))));
    float2 val1 = (*((float2*)((data1_40960+(alu1+(gidx1*10240))))));
    buf0 = (buf0+(val1.x*((float)(val0.x)))+(val1.y*((float)(val0.y))));
  }
  __shared__ __align__(16) float buf1[256];
  float alu4 = (buf0+__shfl_xor_sync(0xffffffffu, buf0, 16));
  float alu5 = (alu4+__shfl_xor_sync(0xffffffffu, alu4, 8));
  float alu6 = (alu5+__shfl_xor_sync(0xffffffffu, alu5, 4));
  float alu7 = (alu6+__shfl_xor_sync(0xffffffffu, alu6, 2));
  *(buf1+lidx0) = (alu7+__shfl_xor_sync(0xffffffffu, alu7, 1));
  __syncthreads();
  float val2 = (*(buf1+0));
  float val3 = (*(buf1+32));
  float val4 = (*(buf1+64));
  float val5 = (*(buf1+96));
  float val6 = (*(buf1+128));
  float val7 = (*(buf1+160));
  float val8 = (*(buf1+192));
  float val9 = (*(buf1+224));
  if ((lidx0==0)) {
    *(data0_16+(gidx0+(gidx1<<2))) = (val2+val6+(val4+val8)+(val3+val7+(val5+val9)));
  }
  asm volatile("griddepcontrol.launch_dependents;");
}