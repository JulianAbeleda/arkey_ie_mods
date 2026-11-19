#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
extern "C" __global__ void __launch_bounds__(64) region_down_tail(float* data0_1280, float* data1_1280) {
  asm volatile("griddepcontrol.wait;":::"memory");
  int gidx0 = blockIdx.x; /* 5 */
  int lidx0 = threadIdx.x; /* 4 */
  int lidx1 = threadIdx.y; /* 16 */
  int alu0 = ((gidx0<<6)+(lidx1<<2)+(lidx0*320));
  float4 val0 = (*((float4*)((data1_1280+alu0))));
  float alu1 = fmaf(val0.x,0.25f,0.0f);
  float alu2 = fmaf(val0.y,0.25f,0.0f);
  float alu3 = fmaf(val0.z,0.25f,0.0f);
  float alu4 = fmaf(val0.w,0.25f,0.0f);
  *((float4*)((data0_1280+alu0))) = make_float4((alu1/(1.0f+exp2((alu1*-1.4426950408889634f)))),(alu2/(1.0f+exp2((alu2*-1.4426950408889634f)))),(alu3/(1.0f+exp2((alu3*-1.4426950408889634f)))),(alu4/(1.0f+exp2((alu4*-1.4426950408889634f)))));
  asm volatile("griddepcontrol.launch_dependents;");
}