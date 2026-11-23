#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
extern "C" __global__ void __launch_bounds__(16) region_ff96d2ea18a8(float* data0_320, float* data1_320) {
  asm volatile("griddepcontrol.wait;":::"memory");
  int gidx0 = blockIdx.x; /* 5 */
  int lidx0 = threadIdx.x; /* 16 */
  int alu0 = ((gidx0<<6)+(lidx0<<2));
  float4 val0 = (*((float4*)((data1_320+alu0))));
  float alu1 = fmaf(val0.x,0.25f,0.0f);
  float alu2 = fmaf(val0.y,0.25f,0.0f);
  float alu3 = fmaf(val0.z,0.25f,0.0f);
  float alu4 = fmaf(val0.w,0.25f,0.0f);
  *((float4*)((data0_320+alu0))) = make_float4((alu1/(1.0f+exp2((alu1*-1.4426950408889634f)))),(alu2/(1.0f+exp2((alu2*-1.4426950408889634f)))),(alu3/(1.0f+exp2((alu3*-1.4426950408889634f)))),(alu4/(1.0f+exp2((alu4*-1.4426950408889634f)))));
  asm volatile("griddepcontrol.launch_dependents;");
}