extern "C" __attribute__((device, const)) float __ocml_log1p_f32(float);
extern "C" __attribute__((device, const)) float __ocml_rsqrt_f32(float);
extern "C" __attribute__((device, const)) float __ocml_exp_f32(float);
#define log1pf __ocml_log1p_f32
#define expf __ocml_exp_f32
#define rsqrtf __ocml_rsqrt_f32
#define __int_as_float(i) __builtin_bit_cast(float, (unsigned int)(i))
typedef long unsigned int size_t;
typedef unsigned short hip_bfloat16;
extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);
typedef float float2 __attribute__((ext_vector_type(2)));
static inline __attribute__((device)) float2 make_float2(float x, float y) { return { x, y }; }
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 256))) region_c91579f0a96b(float* data0_10240, float* data1_10240, float* data2_10240, float* data3_10240, hip_bfloat16* data4_2560) {
  int gidx0 = __ockl_get_group_id(0); /* 4 */
  int lidx0 = __ockl_get_local_id(0); /* 256 */
  float buf0;
  int alu0 = (gidx0*2560);
  buf0 = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 5; Ridx0++) {
    int alu2 = ((lidx0<<1)+(Ridx0<<9));
    hip_bfloat16 val0 = (*(data4_2560+(alu2+1)));
    hip_bfloat16 val1 = (*(data4_2560+alu2));
    float2 val2 = (*((float2*)((data3_10240+(alu2+alu0)))));
    buf0 = __builtin_fmaf(val2.y,__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (hip_bfloat16)(val0))))<<16u))),__builtin_fmaf(val2.x,__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (hip_bfloat16)(val1))))<<16u))),buf0));
  }
  int alu5 = (lidx0+alu0);
  float val3 = (*(data1_10240+alu5));
  int alu6 = (alu5+256);
  float val4 = (*(data1_10240+alu6));
  int alu7 = (alu5+512);
  float val5 = (*(data1_10240+alu7));
  int alu8 = (alu5+768);
  float val6 = (*(data1_10240+alu8));
  int alu9 = (alu5+1024);
  float val7 = (*(data1_10240+alu9));
  int alu10 = (alu5+1280);
  float val8 = (*(data1_10240+alu10));
  int alu11 = (alu5+1536);
  float val9 = (*(data1_10240+alu11));
  int alu12 = (alu5+1792);
  float val10 = (*(data1_10240+alu12));
  int alu13 = (alu5+2048);
  float val11 = (*(data1_10240+alu13));
  int alu14 = (alu5+2304);
  float val12 = (*(data1_10240+alu14));
  float val13 = (*(data2_10240+alu5));
  float val14 = (*(data2_10240+alu6));
  float val15 = (*(data2_10240+alu7));
  float val16 = (*(data2_10240+alu8));
  float val17 = (*(data2_10240+alu9));
  float val18 = (*(data2_10240+alu10));
  float val19 = (*(data2_10240+alu11));
  float val20 = (*(data2_10240+alu12));
  float val21 = (*(data2_10240+alu13));
  float val22 = (*(data2_10240+alu14));
  __attribute__((shared, aligned(16)))float buf1[256];
  float alu15 = (buf0+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^16)<<2), __builtin_bit_cast(int, buf0))));
  float alu16 = (alu15+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^8)<<2), __builtin_bit_cast(int, alu15))));
  float alu17 = (alu16+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^4)<<2), __builtin_bit_cast(int, alu16))));
  float alu18 = (alu17+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^2)<<2), __builtin_bit_cast(int, alu17))));
  *(buf1+lidx0) = (alu18+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^1)<<2), __builtin_bit_cast(int, alu18))));
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  float val23 = (*(buf1+0));
  float val24 = (*(buf1+32));
  float val25 = (*(buf1+64));
  float val26 = (*(buf1+96));
  float val27 = (*(buf1+128));
  float val28 = (*(buf1+160));
  float val29 = (*(buf1+192));
  float val30 = (*(buf1+224));
  int cast0 = ((int)(lidx0));
  float alu21 = (1.0f/(1.0f+expf(((-val23-[](int) { volatile float z16 = 0.0f; return z16; }(cast0))+(-[](int) { volatile float z8 = 0.0f; return z8; }(cast0)-[](int) { volatile float z24 = 0.0f; return z24; }(cast0))+((-val27-[](int) { volatile float z20 = 0.0f; return z20; }(cast0))+(-[](int) { volatile float z12 = 0.0f; return z12; }(cast0)-[](int) { volatile float z28 = 0.0f; return z28; }(cast0)))+((-val25-[](int) { volatile float z18 = 0.0f; return z18; }(cast0))+(-[](int) { volatile float z10 = 0.0f; return z10; }(cast0)-[](int) { volatile float z26 = 0.0f; return z26; }(cast0))+((-val29-[](int) { volatile float z22 = 0.0f; return z22; }(cast0))+(-[](int) { volatile float z14 = 0.0f; return z14; }(cast0)-[](int) { volatile float z30 = 0.0f; return z30; }(cast0))))+((-val24-[](int) { volatile float z17 = 0.0f; return z17; }(cast0))+(-[](int) { volatile float z9 = 0.0f; return z9; }(cast0)-[](int) { volatile float z25 = 0.0f; return z25; }(cast0))+((-val28-[](int) { volatile float z21 = 0.0f; return z21; }(cast0))+(-[](int) { volatile float z13 = 0.0f; return z13; }(cast0)-[](int) { volatile float z29 = 0.0f; return z29; }(cast0)))+((-val26-[](int) { volatile float z19 = 0.0f; return z19; }(cast0))+(-[](int) { volatile float z11 = 0.0f; return z11; }(cast0)-[](int) { volatile float z27 = 0.0f; return z27; }(cast0))+((-val30-[](int) { volatile float z23 = 0.0f; return z23; }(cast0))+(-[](int) { volatile float z15 = 0.0f; return z15; }(cast0)-[](int) { volatile float z31 = 0.0f; return z31; }(cast0)))))))));
  *(data0_10240+alu5) = (val3+(val13*alu21));
  *(data0_10240+alu6) = (val4+(val14*alu21));
  *(data0_10240+alu7) = (val5+(val15*alu21));
  *(data0_10240+alu8) = (val6+(val16*alu21));
  *(data0_10240+alu9) = (val7+(val17*alu21));
  *(data0_10240+alu10) = (val8+(val18*alu21));
  *(data0_10240+alu11) = (val9+(val19*alu21));
  *(data0_10240+alu12) = (val10+(val20*alu21));
  *(data0_10240+alu13) = (val11+(val21*alu21));
  *(data0_10240+alu14) = (val12+(val22*alu21));
}