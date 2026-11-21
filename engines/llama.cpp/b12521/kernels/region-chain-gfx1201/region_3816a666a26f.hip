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
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 256))) region_3816a666a26f(float* data0_192, float* data1_10240, hip_bfloat16* data2_122880, hip_bfloat16* data3_122880, float* data4_48, float* data5_48, float* data6_192) {
  int gidx0 = __ockl_get_group_id(0); /* 96 */
  int gidx1 = __ockl_get_group_id(1); /* 4 */
  int lidx0 = __ockl_get_local_id(0); /* 256 */
  float buf0;
  hip_bfloat16 cast0 = __builtin_bit_cast(hip_bfloat16, (unsigned short)(((unsigned short)(0u))));
  bool alu0 = (gidx0<48);
  bool alu1 = (47<gidx0);
  buf0 = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 5; Ridx0++) {
    int alu3 = ((lidx0<<1)+(Ridx0<<9));
    int alu4 = (alu3+(gidx0*2560));
    hip_bfloat16 val0 = (alu1?*(data3_122880+(alu4+-122879)):cast0);
    hip_bfloat16 val1 = (alu0?*(data2_122880+(alu4+1)):val0);
    hip_bfloat16 val2 = (alu1?*(data3_122880+(alu4+-122880)):cast0);
    hip_bfloat16 val3 = (alu0?*(data2_122880+alu4):val2);
    float2 val4 = (*((float2*)((data1_10240+(alu3+(gidx1*2560))))));
    buf0 = __builtin_fmaf(val4.y,__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (hip_bfloat16)(val1))))<<16u))),__builtin_fmaf(val4.x,__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (hip_bfloat16)(val3))))<<16u))),buf0));
  }
  float val5 = (alu0?*(data4_48+gidx0):0.0f);
  float val6 = (alu0?*(data5_48+gidx0):0.0f);
  __attribute__((shared, aligned(16)))float buf1[256];
  float alu7 = (buf0+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^16)<<2), __builtin_bit_cast(int, buf0))));
  float alu8 = (alu7+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^8)<<2), __builtin_bit_cast(int, alu7))));
  float alu9 = (alu8+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^4)<<2), __builtin_bit_cast(int, alu8))));
  float alu10 = (alu9+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^2)<<2), __builtin_bit_cast(int, alu9))));
  *(buf1+lidx0) = (alu10+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^1)<<2), __builtin_bit_cast(int, alu10))));
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  float val7 = (*(buf1+0));
  float val8 = (*(buf1+32));
  float val9 = (*(buf1+64));
  float val10 = (*(buf1+96));
  float val11 = (*(buf1+128));
  float val12 = (*(buf1+160));
  float val13 = (*(buf1+192));
  float val14 = (*(buf1+224));
  int alu13 = (gidx0+(gidx1*48));
  int cast1 = ((int)(lidx0));
  float alu14 = [](int) { volatile float z10 = 0.0f; return z10; }(cast1);
  float alu15 = [](int) { volatile float z11 = 0.0f; return z11; }(cast1);
  float alu16 = [](int) { volatile float z12 = 0.0f; return z12; }(cast1);
  float alu17 = [](int) { volatile float z13 = 0.0f; return z13; }(cast1);
  float alu18 = [](int) { volatile float z14 = 0.0f; return z14; }(cast1);
  float alu19 = [](int) { volatile float z15 = 0.0f; return z15; }(cast1);
  float alu20 = [](int) { volatile float z8 = 0.0f; return z8; }(cast1);
  float alu21 = [](int) { volatile float z9 = 0.0f; return z9; }(cast1);
  float alu22 = [](int) { volatile float z16 = 0.0f; return z16; }(cast1);
  float alu23 = [](int) { volatile float z17 = 0.0f; return z17; }(cast1);
  float alu24 = [](int) { volatile float z18 = 0.0f; return z18; }(cast1);
  float alu25 = [](int) { volatile float z19 = 0.0f; return z19; }(cast1);
  float alu26 = [](int) { volatile float z20 = 0.0f; return z20; }(cast1);
  float alu27 = [](int) { volatile float z21 = 0.0f; return z21; }(cast1);
  float alu28 = [](int) { volatile float z22 = 0.0f; return z22; }(cast1);
  float alu29 = [](int) { volatile float z23 = 0.0f; return z23; }(cast1);
  float alu30 = [](int) { volatile float z24 = 0.0f; return z24; }(cast1);
  float alu31 = [](int) { volatile float z25 = 0.0f; return z25; }(cast1);
  float alu32 = [](int) { volatile float z26 = 0.0f; return z26; }(cast1);
  float alu33 = [](int) { volatile float z27 = 0.0f; return z27; }(cast1);
  float alu34 = [](int) { volatile float z28 = 0.0f; return z28; }(cast1);
  float alu35 = [](int) { volatile float z29 = 0.0f; return z29; }(cast1);
  float alu36 = [](int) { volatile float z30 = 0.0f; return z30; }(cast1);
  float alu37 = [](int) { volatile float z31 = 0.0f; return z31; }(cast1);
  float alu38 = (val7+alu22+(alu20+alu30)+(val11+alu26+(alu16+alu34))+(val9+alu24+(alu14+alu32)+(val13+alu28+(alu18+alu36)))+(val8+alu23+(alu21+alu31)+(val12+alu27+(alu17+alu35))+(val10+alu25+(alu15+alu33)+(val14+alu29+(alu19+alu37))))+val5);
  float alu39 = ((20.0f<alu38)?alu38:log1pf(expf(alu38)));
  bool alu40 = (lidx0==0);
  float alu41 = (alu0?(alu39*val6):(1.0f/(1.0f+expf(((-val7-alu22)+(-alu20-alu30)+((-val11-alu26)+(-alu16-alu34))+((-val9-alu24)+(-alu14-alu32)+((-val13-alu28)+(-alu18-alu36)))+((-val8-alu23)+(-alu21-alu31)+((-val12-alu27)+(-alu17-alu35))+((-val10-alu25)+(-alu15-alu33)+((-val14-alu29)+(-alu19-alu37)))))))));
  if ((alu40&alu0)) {
    *(data0_192+alu13) = alu41;
  }
  if ((alu40&alu1)) {
    *(data6_192+(alu13+-48)) = alu41;
  }
}