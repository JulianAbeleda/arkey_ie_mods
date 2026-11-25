#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) sparse_mask_write_t4_l2051(int* data0_8204, int* data1_4, half* data2_1048576, int* data3_256, const int nkv) {
  int gidx0 = blockIdx.x; /* 4 */
  int gidx1 = blockIdx.y; /* 64 */
  int lidx0 = threadIdx.x; /* 32 */
  int lidx1 = threadIdx.y; /* 8 */
  int buf0;
  buf0 = 0;
  int alu1 = ((lidx0<<4)+(lidx1<<9)+(gidx1<<12));
  int alu2 = (nkv*gidx0);
  for (int Ridx2 = 0; Ridx2 < 16; Ridx2++) {
    int alu3 = (alu1+Ridx2);
    bool alu4 = (alu3<nkv);
    int alu5 = (alu4?(alu3+alu2):0);
    half val0 = (*(data2_1048576+alu5));
    int alu6 = ((alu4&(-1e+30f<((float)(val0))))?1:0);
    buf0 = (buf0+alu6);
  }
  int alu9 = (lidx0+(lidx1<<5));
  int val1 = ((lidx1<2)?*(data3_256+(alu9+(gidx0<<6))):0);
  __shared__ __align__(16) int buf1[24];
  int buf2;
  int buf3;
  int buf4;
  int buf5;
  int buf6;
  int buf7;
  int buf8;
  buf2 = __shfl_up_sync(0xffffffffu, buf0, 1);
  int alu11 = ((lidx0<1)?0:buf2);
  int alu12 = (buf0+alu11);
  buf3 = __shfl_up_sync(0xffffffffu, alu12, 2);
  int alu14 = ((lidx0<2)?0:buf3);
  int alu15 = (alu12+alu14);
  buf4 = __shfl_up_sync(0xffffffffu, alu15, 4);
  int alu17 = ((lidx0<4)?0:buf4);
  int alu18 = (alu15+alu17);
  buf5 = __shfl_up_sync(0xffffffffu, alu18, 8);
  int alu20 = ((lidx0<8)?0:buf5);
  int alu21 = (alu18+alu20);
  buf6 = __shfl_up_sync(0xffffffffu, alu21, 16);
  int alu23 = ((lidx0<16)?0:buf6);
  int alu24 = (alu21+alu23);
  if ((lidx0==31)) {
    *(buf1+lidx1) = alu24;
  }
  int alu28 = ((alu9<gidx1)?val1:0);
  buf7 = __shfl_xor_sync(0xffffffffu, alu28, 16);
  int alu30 = (alu28+buf7);
  buf7 = __shfl_xor_sync(0xffffffffu, alu30, 8);
  int alu32 = (alu30+buf7);
  buf7 = __shfl_xor_sync(0xffffffffu, alu32, 4);
  int alu34 = (alu32+buf7);
  buf7 = __shfl_xor_sync(0xffffffffu, alu34, 2);
  int alu36 = (alu34+buf7);
  buf7 = __shfl_xor_sync(0xffffffffu, alu36, 1);
  bool alu38 = (lidx0==0);
  if (alu38) {
    *(buf1+(lidx1+8)) = (alu36+buf7);
  }
  buf8 = __shfl_xor_sync(0xffffffffu, val1, 16);
  int alu43 = (val1+buf8);
  buf8 = __shfl_xor_sync(0xffffffffu, alu43, 8);
  int alu45 = (alu43+buf8);
  buf8 = __shfl_xor_sync(0xffffffffu, alu45, 4);
  int alu47 = (alu45+buf8);
  buf8 = __shfl_xor_sync(0xffffffffu, alu47, 2);
  int alu49 = (alu47+buf8);
  buf8 = __shfl_xor_sync(0xffffffffu, alu49, 1);
  if (alu38) {
    *(buf1+(lidx1+16)) = (alu49+buf8);
  }
  __syncthreads();
  int val2 = (*(buf1+0));
  int val3 = (*(buf1+1));
  int val4 = (*(buf1+2));
  int val5 = (*(buf1+3));
  int val6 = (*(buf1+4));
  int val7 = (*(buf1+5));
  int val8 = (*(buf1+6));
  int val9 = (*(buf1+8));
  int val10 = (*(buf1+9));
  int val11 = (*(buf1+10));
  int val12 = (*(buf1+11));
  int val13 = (*(buf1+12));
  int val14 = (*(buf1+13));
  int val15 = (*(buf1+14));
  int val16 = (*(buf1+15));
  int buf9;
  int alu55 = ((0<lidx1)?val2:0);
  int alu56 = ((1<lidx1)?val3:0);
  int alu57 = ((2<lidx1)?val4:0);
  int alu58 = ((3<lidx1)?val5:0);
  int alu59 = ((4<lidx1)?val6:0);
  int alu60 = ((5<lidx1)?val7:0);
  int alu61 = ((6<lidx1)?val8:0);
  buf9 = ((val9+val10+val11+val12+val13+val14+val15+val16+alu55+alu56+alu57+alu58+alu59+alu60+alu61+alu24)-buf0);
  for (int Ridx3 = 0; Ridx3 < 16; Ridx3++) {
    int alu63 = (alu1+Ridx3);
    bool alu64 = (alu63<nkv);
    int alu65 = (alu64?(alu63+alu2):0);
    half val17 = (*(data2_1048576+alu65));
    bool alu66 = (buf9<2051);
    int alu67 = (alu66?buf9:0);
    int alu68 = ((alu64&(-1e+30f<((float)(val17))))?1:0);
    if (((alu68==1)&alu66)) {
      *(data0_8204+(((long)((gidx0*2051)))+((long)(alu67)))) = alu63;
    }
    buf9 = (buf9+alu68);
  }
  int val18 = (*(buf1+16));
  int val19 = (*(buf1+17));
  int val20 = (*(buf1+18));
  int val21 = (*(buf1+19));
  int val22 = (*(buf1+20));
  int val23 = (*(buf1+21));
  int val24 = (*(buf1+22));
  int val25 = (*(buf1+23));
  int alu74 = ((val18+val19+val20+val21+val22+val23+val24+val25)^-1);
  int alu75 = ((alu74<-2052)?-2052:alu74);
  if (((gidx1==63)&alu38&(lidx1==0))) {
    *(data1_4+gidx0) = (alu75^-1);
  }
}