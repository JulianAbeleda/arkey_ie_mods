#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_bf16.h>
extern "C" __global__ void __launch_bounds__(256) region_gdn_gate(float* data0_192, float* data1_10240, nv_bfloat16* data2_122880, nv_bfloat16* data3_122880, float* data4_48, float* data5_48, float* data6_192) {
  asm volatile("griddepcontrol.wait;":::"memory");
  int gidx0 = blockIdx.x; /* 96 */
  int gidx1 = blockIdx.y; /* 4 */
  int lidx0 = threadIdx.x; /* 256 */
  float buf0;
  bool alu0 = (gidx0<48);
  bool alu1 = (47<gidx0);
  buf0 = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 5; Ridx0++) {
    int alu3 = ((lidx0<<1)+(Ridx0<<9));
    int alu4 = (alu3+(gidx0*2560));
    nv_bfloat16 val0 = (alu1?*(data3_122880+(alu4+-122879)):((nv_bfloat16)(0.0f)));
    nv_bfloat16 val1 = (alu0?*(data2_122880+(alu4+1)):val0);
    nv_bfloat16 val2 = (alu1?*(data3_122880+(alu4+-122880)):((nv_bfloat16)(0.0f)));
    nv_bfloat16 val3 = (alu0?*(data2_122880+alu4):val2);
    float2 val4 = (*((float2*)((data1_10240+(alu3+(gidx1*2560))))));
    buf0 = fmaf(val4.y,((float)(val1)),fmaf(val4.x,((float)(val3)),buf0));
  }
  float val5 = (alu0?*(data4_48+gidx0):0.0f);
  float val6 = (alu0?*(data5_48+gidx0):0.0f);
  __shared__ __align__(16) float buf1[256];
  float alu7 = (buf0+__shfl_xor_sync(0xffffffffu, buf0, 16));
  float alu8 = (alu7+__shfl_xor_sync(0xffffffffu, alu7, 8));
  float alu9 = (alu8+__shfl_xor_sync(0xffffffffu, alu8, 4));
  float alu10 = (alu9+__shfl_xor_sync(0xffffffffu, alu9, 2));
  *(buf1+lidx0) = (alu10+__shfl_xor_sync(0xffffffffu, alu10, 1));
  __syncthreads();
  float val7 = (*(buf1+0));
  float val8 = (*(buf1+32));
  float val9 = (*(buf1+64));
  float val10 = (*(buf1+96));
  float val11 = (*(buf1+128));
  float val12 = (*(buf1+160));
  float val13 = (*(buf1+192));
  float val14 = (*(buf1+224));
  int alu13 = (gidx0+(gidx1*48));
  float alu14 = (val7+val11+(val9+val13)+(val8+val12+(val10+val14)));
  float alu15 = (alu14+val5);
  float alu16 = ((20.0f<alu15)?alu15:log1pf(exp2((alu15*1.4426950408889634f))));
  bool alu17 = (lidx0==0);
  float alu18 = (alu0?(alu16*val6):(1/(1.0f+exp2((alu14*-1.4426950408889634f)))));
  if ((alu17&alu0)) {
    *(data0_192+alu13) = alu18;
  }
  if ((alu17&alu1)) {
    *(data6_192+(alu13+-48)) = alu18;
  }
  asm volatile("griddepcontrol.launch_dependents;");
}