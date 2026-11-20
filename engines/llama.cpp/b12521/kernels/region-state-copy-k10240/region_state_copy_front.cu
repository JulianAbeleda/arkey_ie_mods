#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
extern "C" __global__ void __launch_bounds__(224) region_state_copy_front(float* data0_71680, float* data1_30720, float* data2_40960, float* data3_122880) {
  int gidx0 = blockIdx.x; /* 320 */
  int lidx0 = threadIdx.x; /* 7 */
  int lidx1 = threadIdx.y; /* 32 */
  bool alu0 = (2<lidx0);
  float val0 = (alu0?*(data2_40960+(lidx1+(gidx0<<5)+(lidx0*10240)+-30720)):0.0f);
  int alu1 = (lidx0+(gidx0*96)+(lidx1*3));
  float val1 = ((lidx0<3)?*(data1_30720+alu1):val0);
  *(data0_71680+(lidx0+(gidx0*224)+(lidx1*7))) = val1;
  if ((3<lidx0)) {
    *(data3_122880+(alu1+-4)) = val1;
  }
  if ((alu0&(lidx0<6))) {
    *(data3_122880+(alu1+30717)) = val1;
  }
  if (((1<lidx0)&(lidx0<5))) {
    *(data3_122880+(alu1+61438)) = val1;
  }
  if (((0<lidx0)&(lidx0<4))) {
    *(data3_122880+(alu1+92159)) = val1;
  }
}