#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) sparse_mask_count_t4(int* data0_256, half* data1_1048576, const int nkv) {
  int gidx0 = blockIdx.x; /* 4 */
  int gidx1 = blockIdx.y; /* 64 */
  int lidx0 = threadIdx.x; /* 32 */
  int lidx1 = threadIdx.y; /* 8 */
  int buf0;
  buf0 = 0;
  for (int Ridx2 = 0; Ridx2 < 16; Ridx2++) {
    int alu1 = ((lidx0<<4)+(lidx1<<9)+(gidx1<<12)+Ridx2);
    bool alu2 = (alu1<nkv);
    int alu3 = (alu2?(alu1+(nkv*gidx0)):0);
    half val0 = (*(data1_1048576+alu3));
    int alu4 = ((alu2&(-1e+30f<((float)(val0))))?1:0);
    buf0 = (buf0+alu4);
  }
  __shared__ __align__(16) int buf1[8];
  float buf2;
  float cast0 = ((float)(buf0));
  buf2 = __shfl_xor_sync(0xffffffffu, cast0, 16);
  float alu8 = (cast0+buf2);
  buf2 = __shfl_xor_sync(0xffffffffu, alu8, 8);
  float alu10 = (alu8+buf2);
  buf2 = __shfl_xor_sync(0xffffffffu, alu10, 4);
  float alu12 = (alu10+buf2);
  buf2 = __shfl_xor_sync(0xffffffffu, alu12, 2);
  float alu14 = (alu12+buf2);
  buf2 = __shfl_xor_sync(0xffffffffu, alu14, 1);
  bool alu16 = (lidx0==0);
  if (alu16) {
    *(buf1+lidx1) = ((int)((alu14+buf2)));
  }
  __syncthreads();
  int val1 = (*(buf1+0));
  int val2 = (*(buf1+1));
  int val3 = (*(buf1+2));
  int val4 = (*(buf1+3));
  int val5 = (*(buf1+4));
  int val6 = (*(buf1+5));
  int val7 = (*(buf1+6));
  int val8 = (*(buf1+7));
  if ((alu16&(lidx1==0))) {
    *(data0_256+(gidx1+(gidx0<<6))) = (val1+val2+val3+val4+val5+val6+val7+val8);
  }
}