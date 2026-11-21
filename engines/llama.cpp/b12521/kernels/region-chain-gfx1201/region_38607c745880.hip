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
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 256))) region_38607c745880(float* data0_16, float* data1_40960, hip_bfloat16* data2_40960) {
  int gidx0 = __ockl_get_group_id(0); /* 4 */
  int gidx1 = __ockl_get_group_id(1); /* 4 */
  int lidx0 = __ockl_get_local_id(0); /* 256 */
  float buf0;
  buf0 = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 20; Ridx0++) {
    int alu1 = ((lidx0<<1)+(Ridx0<<9));
    int alu2 = (alu1+(gidx0*10240));
    hip_bfloat16 val0 = (*(data2_40960+(alu2+1)));
    hip_bfloat16 val1 = (*(data2_40960+alu2));
    float2 val2 = (*((float2*)((data1_40960+(alu1+(gidx1*10240))))));
    buf0 = __builtin_fmaf(val2.y,__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (hip_bfloat16)(val0))))<<16u))),__builtin_fmaf(val2.x,__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (hip_bfloat16)(val1))))<<16u))),buf0));
  }
  __attribute__((shared, aligned(16)))float buf1[256];
  float alu5 = (buf0+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^16)<<2), __builtin_bit_cast(int, buf0))));
  float alu6 = (alu5+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^8)<<2), __builtin_bit_cast(int, alu5))));
  float alu7 = (alu6+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^4)<<2), __builtin_bit_cast(int, alu6))));
  float alu8 = (alu7+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^2)<<2), __builtin_bit_cast(int, alu7))));
  *(buf1+lidx0) = (alu8+__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lidx0^1)<<2), __builtin_bit_cast(int, alu8))));
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  float val3 = (*(buf1+0));
  float val4 = (*(buf1+32));
  float val5 = (*(buf1+64));
  float val6 = (*(buf1+96));
  float val7 = (*(buf1+128));
  float val8 = (*(buf1+160));
  float val9 = (*(buf1+192));
  float val10 = (*(buf1+224));
  int cast0 = ((int)(lidx0));
  if ((lidx0==0)) {
    *(data0_16+(gidx0+(gidx1<<2))) = __builtin_fmaf((1.0f/(1.0f+expf(-__builtin_fmaf((val3+[](int) { volatile float z16 = 0.0f; return z16; }(cast0)+([](int) { volatile float z8 = 0.0f; return z8; }(cast0)+[](int) { volatile float z24 = 0.0f; return z24; }(cast0))+(val7+[](int) { volatile float z20 = 0.0f; return z20; }(cast0)+([](int) { volatile float z12 = 0.0f; return z12; }(cast0)+[](int) { volatile float z28 = 0.0f; return z28; }(cast0)))+(val5+[](int) { volatile float z18 = 0.0f; return z18; }(cast0)+([](int) { volatile float z10 = 0.0f; return z10; }(cast0)+[](int) { volatile float z26 = 0.0f; return z26; }(cast0))+(val9+[](int) { volatile float z22 = 0.0f; return z22; }(cast0)+([](int) { volatile float z14 = 0.0f; return z14; }(cast0)+[](int) { volatile float z30 = 0.0f; return z30; }(cast0))))+(val4+[](int) { volatile float z17 = 0.0f; return z17; }(cast0)+([](int) { volatile float z9 = 0.0f; return z9; }(cast0)+[](int) { volatile float z25 = 0.0f; return z25; }(cast0))+(val8+[](int) { volatile float z21 = 0.0f; return z21; }(cast0)+([](int) { volatile float z13 = 0.0f; return z13; }(cast0)+[](int) { volatile float z29 = 0.0f; return z29; }(cast0)))+(val6+[](int) { volatile float z19 = 0.0f; return z19; }(cast0)+([](int) { volatile float z11 = 0.0f; return z11; }(cast0)+[](int) { volatile float z27 = 0.0f; return z27; }(cast0))+(val10+[](int) { volatile float z23 = 0.0f; return z23; }(cast0)+([](int) { volatile float z15 = 0.0f; return z15; }(cast0)+[](int) { volatile float z31 = 0.0f; return z31; }(cast0)))))),0.25f,0.0f)))),2.0f,0.0f);
  }
}