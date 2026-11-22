#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
extern "C" __global__ void __launch_bounds__(208) region_7fc28e3641f8(float* data0_133120, float* data1_92160, float* data2_40960, float* data3_368640) {
  int gidx0 = blockIdx.x; /* 640 */
  int lidx0 = threadIdx.x; /* 13 */
  int lidx1 = threadIdx.y; /* 16 */
  float val0 = ((8<lidx0)?*(data2_40960+(lidx1+(gidx0<<4)+(lidx0*10240)+-92160)):0.0f);
  int alu0 = (lidx0+(gidx0*144)+(lidx1*9));
  float val1 = ((lidx0<9)?*(data1_92160+alu0):val0);
  *(data0_133120+(lidx0+(gidx0*208)+(lidx1*13))) = val1;
  if ((3<lidx0)) {
    *(data3_368640+(alu0+-4)) = val1;
  }
  if (((2<lidx0)&(lidx0<12))) {
    *(data3_368640+(alu0+92157)) = val1;
  }
  if (((1<lidx0)&(lidx0<11))) {
    *(data3_368640+(alu0+184318)) = val1;
  }
  if (((0<lidx0)&(lidx0<10))) {
    *(data3_368640+(alu0+276479)) = val1;
  }
}