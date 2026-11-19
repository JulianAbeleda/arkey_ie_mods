#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_bf16.h>
struct __align__(4) tg_bfloat162 { nv_bfloat16 x, y; }; __device__ tg_bfloat162 make_tg_bfloat162(nv_bfloat16 x, nv_bfloat16 y) { tg_bfloat162 r={x, y}; return r; }
extern "C" __global__ void __launch_bounds__(256) ark_down_c4u1(float* data0_1280, float* data1_40960, nv_bfloat16* data2_3276800) {
  asm volatile("griddepcontrol.wait;":::"memory");
  int gidx0 = blockIdx.x; /* 320 */
  int lidx0 = threadIdx.x; /* 256 */
  float buf0[4];
  (*(buf0+0)) = 0.0f;
  (*(buf0+1)) = 0.0f;
  (*(buf0+2)) = 0.0f;
  (*(buf0+3)) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 20; Ridx0++) {
    int alu4 = ((lidx0<<1)+(Ridx0<<9));
    tg_bfloat162 val0 = (*((tg_bfloat162*)((data2_3276800+(alu4+(gidx0*10240))))));
    float2 val1 = (*((float2*)((data1_40960+(alu4+10240)))));
    float2 val2 = (*((float2*)((data1_40960+(alu4+20480)))));
    float2 val3 = (*((float2*)((data1_40960+(alu4+30720)))));
    float2 val4 = (*((float2*)((data1_40960+alu4))));
    float cast0 = ((float)(val0.x));
    float cast1 = ((float)(val0.y));
    (*(buf0+0)) = ((*(buf0+0))+(val4.x*cast0)+(val4.y*cast1));
    (*(buf0+1)) = ((*(buf0+1))+(val1.x*cast0)+(val1.y*cast1));
    (*(buf0+2)) = ((*(buf0+2))+(val2.x*cast0)+(val2.y*cast1));
    (*(buf0+3)) = ((*(buf0+3))+(val3.x*cast0)+(val3.y*cast1));
  }
  __shared__ __align__(16) float buf1[1024];
  float alu10 = ((*(buf0+0))+__shfl_xor_sync(0xffffffffu, (*(buf0+0)), 16));
  float alu11 = ((*(buf0+1))+__shfl_xor_sync(0xffffffffu, (*(buf0+1)), 16));
  float alu12 = ((*(buf0+2))+__shfl_xor_sync(0xffffffffu, (*(buf0+2)), 16));
  float alu13 = ((*(buf0+3))+__shfl_xor_sync(0xffffffffu, (*(buf0+3)), 16));
  float alu14 = (alu10+__shfl_xor_sync(0xffffffffu, alu10, 8));
  float alu15 = (alu11+__shfl_xor_sync(0xffffffffu, alu11, 8));
  float alu16 = (alu12+__shfl_xor_sync(0xffffffffu, alu12, 8));
  float alu17 = (alu13+__shfl_xor_sync(0xffffffffu, alu13, 8));
  float alu18 = (alu14+__shfl_xor_sync(0xffffffffu, alu14, 4));
  float alu19 = (alu15+__shfl_xor_sync(0xffffffffu, alu15, 4));
  float alu20 = (alu16+__shfl_xor_sync(0xffffffffu, alu16, 4));
  float alu21 = (alu17+__shfl_xor_sync(0xffffffffu, alu17, 4));
  float alu22 = (alu18+__shfl_xor_sync(0xffffffffu, alu18, 2));
  float alu23 = (alu19+__shfl_xor_sync(0xffffffffu, alu19, 2));
  float alu24 = (alu20+__shfl_xor_sync(0xffffffffu, alu20, 2));
  float alu25 = (alu21+__shfl_xor_sync(0xffffffffu, alu21, 2));
  *((float4*)((buf1+(lidx0<<2)))) = make_float4((alu22+__shfl_xor_sync(0xffffffffu, alu22, 1)),(alu23+__shfl_xor_sync(0xffffffffu, alu23, 1)),(alu24+__shfl_xor_sync(0xffffffffu, alu24, 1)),(alu25+__shfl_xor_sync(0xffffffffu, alu25, 1)));
  __syncthreads();
  float4 val5 = (*((float4*)((buf1+0))));
  float4 val6 = (*((float4*)((buf1+128))));
  float4 val7 = (*((float4*)((buf1+256))));
  float4 val8 = (*((float4*)((buf1+384))));
  float4 val9 = (*((float4*)((buf1+512))));
  float4 val10 = (*((float4*)((buf1+640))));
  float4 val11 = (*((float4*)((buf1+768))));
  float4 val12 = (*((float4*)((buf1+896))));
  bool alu28 = (lidx0==0);
  if (alu28) {
    *(data0_1280+gidx0) = (val5.x+val9.x+(val7.x+val11.x)+(val6.x+val10.x+(val8.x+val12.x)));
  }
  if (alu28) {
    *(data0_1280+(gidx0+320)) = (val5.y+val9.y+(val7.y+val11.y)+(val6.y+val10.y+(val8.y+val12.y)));
  }
  if (alu28) {
    *(data0_1280+(gidx0+640)) = (val5.z+val9.z+(val7.z+val11.z)+(val6.z+val10.z+(val8.z+val12.z)));
  }
  if (alu28) {
    *(data0_1280+(gidx0+960)) = (val5.w+val9.w+(val7.w+val11.w)+(val6.w+val10.w+(val8.w+val12.w)));
  }
  asm volatile("griddepcontrol.launch_dependents;");
}