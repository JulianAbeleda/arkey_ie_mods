#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_bf16.h>
struct __align__(4) tg_bfloat162 { nv_bfloat16 x, y; }; __device__ tg_bfloat162 make_tg_bfloat162(nv_bfloat16 x, nv_bfloat16 y) { tg_bfloat162 r={x, y}; return r; }
extern "C" __global__ void __launch_bounds__(256) region_shexp_gate(float* data0_10240, float* data1_10240, float* data2_10240, float* data3_10240, nv_bfloat16* data4_2560) {
  asm volatile("griddepcontrol.wait;":::"memory");
  int gidx0 = blockIdx.x; /* 4 */
  int lidx0 = threadIdx.x; /* 256 */
  float buf0;
  int alu0 = (gidx0*2560);
  buf0 = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 5; Ridx0++) {
    int alu2 = ((lidx0<<1)+(Ridx0<<9));
    tg_bfloat162 val0 = (*((tg_bfloat162*)((data4_2560+alu2))));
    float2 val1 = (*((float2*)((data3_10240+(alu2+alu0)))));
    buf0 = fmaf(val1.y,((float)(val0.y)),fmaf(val1.x,((float)(val0.x)),buf0));
  }
  int alu5 = (lidx0+alu0);
  float val2 = (*(data1_10240+alu5));
  int alu6 = (alu5+256);
  float val3 = (*(data1_10240+alu6));
  int alu7 = (alu5+512);
  float val4 = (*(data1_10240+alu7));
  int alu8 = (alu5+768);
  float val5 = (*(data1_10240+alu8));
  int alu9 = (alu5+1024);
  float val6 = (*(data1_10240+alu9));
  int alu10 = (alu5+1280);
  float val7 = (*(data1_10240+alu10));
  int alu11 = (alu5+1536);
  float val8 = (*(data1_10240+alu11));
  int alu12 = (alu5+1792);
  float val9 = (*(data1_10240+alu12));
  int alu13 = (alu5+2048);
  float val10 = (*(data1_10240+alu13));
  int alu14 = (alu5+2304);
  float val11 = (*(data1_10240+alu14));
  float val12 = (*(data2_10240+alu5));
  float val13 = (*(data2_10240+alu6));
  float val14 = (*(data2_10240+alu7));
  float val15 = (*(data2_10240+alu8));
  float val16 = (*(data2_10240+alu9));
  float val17 = (*(data2_10240+alu10));
  float val18 = (*(data2_10240+alu11));
  float val19 = (*(data2_10240+alu12));
  float val20 = (*(data2_10240+alu13));
  float val21 = (*(data2_10240+alu14));
  __shared__ __align__(16) float buf1[256];
  float alu15 = (buf0+__shfl_xor_sync(0xffffffffu, buf0, 16));
  float alu16 = (alu15+__shfl_xor_sync(0xffffffffu, alu15, 8));
  float alu17 = (alu16+__shfl_xor_sync(0xffffffffu, alu16, 4));
  float alu18 = (alu17+__shfl_xor_sync(0xffffffffu, alu17, 2));
  *(buf1+lidx0) = (alu18+__shfl_xor_sync(0xffffffffu, alu18, 1));
  __syncthreads();
  float val22 = (*(buf1+0));
  float val23 = (*(buf1+32));
  float val24 = (*(buf1+64));
  float val25 = (*(buf1+96));
  float val26 = (*(buf1+128));
  float val27 = (*(buf1+160));
  float val28 = (*(buf1+192));
  float val29 = (*(buf1+224));
  float alu21 = (1/(1.0f+exp2(((val22+val26+(val24+val28)+(val23+val27+(val25+val29)))*-1.4426950408889634f))));
  *(data0_10240+alu5) = (val2+(val12*alu21));
  *(data0_10240+alu6) = (val3+(val13*alu21));
  *(data0_10240+alu7) = (val4+(val14*alu21));
  *(data0_10240+alu8) = (val5+(val15*alu21));
  *(data0_10240+alu9) = (val6+(val16*alu21));
  *(data0_10240+alu10) = (val7+(val17*alu21));
  *(data0_10240+alu11) = (val8+(val18*alu21));
  *(data0_10240+alu12) = (val9+(val19*alu21));
  *(data0_10240+alu13) = (val10+(val20*alu21));
  *(data0_10240+alu14) = (val11+(val21*alu21));
  asm volatile("griddepcontrol.launch_dependents;");
}