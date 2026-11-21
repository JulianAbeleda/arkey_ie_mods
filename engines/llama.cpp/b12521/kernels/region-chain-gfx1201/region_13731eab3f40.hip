extern "C" __attribute__((device, const)) float __ocml_log1p_f32(float);
extern "C" __attribute__((device, const)) float __ocml_rsqrt_f32(float);
extern "C" __attribute__((device, const)) float __ocml_exp_f32(float);
#define log1pf __ocml_log1p_f32
#define expf __ocml_exp_f32
#define rsqrtf __ocml_rsqrt_f32
#define __int_as_float(i) __builtin_bit_cast(float, (unsigned int)(i))
typedef long unsigned int size_t;
extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);
typedef float float4 __attribute__((ext_vector_type(4)));
static inline __attribute__((device)) float4 make_float4(float x, float y, float z, float w) { return { x, y, z, w }; }
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 64))) region_13731eab3f40(float* data0_1280, float* data1_1280) {
  int gidx0 = __ockl_get_group_id(0); /* 5 */
  int lidx0 = __ockl_get_local_id(0); /* 4 */
  int lidx1 = __ockl_get_local_id(1); /* 16 */
  int alu0 = ((gidx0<<6)+(lidx1<<2)+(lidx0*320));
  float4 val0 = (*((float4*)((data1_1280+alu0))));
  float alu1 = __builtin_fmaf(val0.x,0.25f,0.0f);
  float alu2 = __builtin_fmaf(val0.y,0.25f,0.0f);
  float alu3 = __builtin_fmaf(val0.z,0.25f,0.0f);
  float alu4 = __builtin_fmaf(val0.w,0.25f,0.0f);
  *((float4*)((data0_1280+alu0))) = make_float4((alu1/(1.0f+expf(-alu1))),(alu2/(1.0f+expf(-alu2))),(alu3/(1.0f+expf(-alu3))),(alu4/(1.0f+expf(-alu4))));
}