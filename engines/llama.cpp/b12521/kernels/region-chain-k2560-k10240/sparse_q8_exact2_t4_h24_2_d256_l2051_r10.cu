#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
struct __align__(128) half64 { half v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63; }; __device__ half64 make_half64(half v0, half v1, half v2, half v3, half v4, half v5, half v6, half v7, half v8, half v9, half v10, half v11, half v12, half v13, half v14, half v15, half v16, half v17, half v18, half v19, half v20, half v21, half v22, half v23, half v24, half v25, half v26, half v27, half v28, half v29, half v30, half v31, half v32, half v33, half v34, half v35, half v36, half v37, half v38, half v39, half v40, half v41, half v42, half v43, half v44, half v45, half v46, half v47, half v48, half v49, half v50, half v51, half v52, half v53, half v54, half v55, half v56, half v57, half v58, half v59, half v60, half v61, half v62, half v63) { half64 r={v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63}; return r; }
struct __align__(8) half4 { half x, y, z, w; }; __device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x, y, z, w}; return r; }
struct __align__(16) half8 { half x, y, z, w, a, b, c, d; }; __device__ half8 make_half8(half x, half y, half z, half w, half a, half b, half c, half d) { half8 r={x, y, z, w, a, b, c, d}; return r; }
__device__ float4 __WMMA_8_16_16_half_float(half8 a, half4 b, float4 c){
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7},"
      "{%8, %9}, {%0, %1, %2, %3};"
    : "+r"(c_pk[0]), "+r"(c_pk[1]), "+r"(c_pk[2]), "+r"(c_pk[3])
    : "r"(a_pk[0]), "r"(a_pk[1]), "r"(a_pk[2]), "r"(a_pk[3]), "r"(b_pk[0]), "r"(b_pk[1]));
  return c;
}
__device__ half4 __WMMA_8_16_16_half_half(half8 a, half4 b, half4 c){
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);
  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5},"
      "{%6, %7}, {%0, %1};"
    : "+r"(c_pk[0]), "+r"(c_pk[1])
    : "r"(a_pk[0]), "r"(a_pk[1]), "r"(a_pk[2]), "r"(a_pk[3]), "r"(b_pk[0]), "r"(b_pk[1]));
  return c;
}
extern "C" __global__ void __launch_bounds__(64) sparse_q8_exact2_t4_h24_2_d256_l2051_r10(half* data0_983040, float* data1_7680, float* data2_24576, int* data3_35651584, int* data4_35651584, int* data5_8204, int* data6_4) {
  half buf0[64];
  for (int Lidx3 = 0; Lidx3 < 64; Lidx3++) {
    (*(buf0+Lidx3)) = ((half)(0.0f));
  }
  float buf1[2];
  for (int Lidx13 = 0; Lidx13 < 2; Lidx13++) {
    (*(buf1+Lidx13)) = -1.7014117331926443e+38f;
  }
  float buf2[2];
  for (int Lidx14 = 0; Lidx14 < 2; Lidx14++) {
    (*(buf2+Lidx14)) = 0.0f;
  }
  int gidx0 = blockIdx.x; /* 8 */
  int alu6 = (gidx0&3);
  int val0 = (*(data6_4+alu6));
  int gidx1 = blockIdx.y; /* 4 */
  int gidx2 = blockIdx.z; /* 10 */
  int lidx0 = threadIdx.x; /* 32 */
  int lidx1 = threadIdx.y; /* 2 */
  __shared__ __align__(16) int buf3[4352];
  __shared__ __align__(16) int buf4[4352];
  long alu7 = (((long)(val0))+63ll);
  bool alu8 = (alu7<0ll);
  long alu9 = (alu8?63ll:0ll);
  int alu10 = (gidx0>>2);
  int alu11 = (gidx2*3);
  int alu12 = (alu11/10);
  int alu13 = (alu11+alu12);
  int alu14 = (alu13+1);
  int alu15 = ((gidx2*192)+(alu12<<6));
  int alu16 = (gidx1<<4);
  int alu17 = (((int)((((alu7+alu9)>>6ll)-((long)((((alu7%64ll)!=0ll)&(alu8!=0)))))))^-1);
  int alu18 = ((alu11+((alu11+3)/10)+3)^-1);
  int alu19 = ((alu18<alu17)?alu17:alu18);
  int alu20 = (alu19^-1);
  int alu21 = ((alu14<alu20)?alu20:alu14);
  for (int Ridx60 = 0; Ridx60 < 17; Ridx60++) {
    int alu22 = (lidx0+(lidx1*15)+(Ridx60*13));
    int alu23 = (lidx1+(Ridx60*3)+(alu22/17));
    int alu24 = (alu23>>4);
    int alu25 = (alu15+(alu24<<6)+alu16+(alu23&15));
    bool alu26 = (((alu13+alu24)<alu21)&(alu25<val0));
    int alu27 = (alu26?alu25:0);
    int val1 = (alu26?*(data5_8204+((alu6*2051)+alu27)):0);
    long alu28 = ((((long)(val1))*136ll)+((long)((alu10*68)))+((long)(((alu22%17)<<2))));
    int alu29 = ((lidx0<<2)+(lidx1*60)+(Ridx60*52)+(lidx1*68)+(Ridx60*204));
    if (1) *((int4*)((buf3+alu29))) = (alu26) ? __ldg((const int4*)((data3_35651584+alu28))) : make_int4(0, 0, 0, 0);
    if (1) *((int4*)((buf4+alu29))) = (alu26) ? __ldg((const int4*)((data4_35651584+alu28))) : make_int4(0, 0, 0, 0);
  }
  bool alu33 = ((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))<0);
  bool alu34 = (alu33!=0);
  int alu35 = ((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))&3);
  int alu36 = (alu35<<1);
  int alu37 = (alu33?3:0);
  int alu38 = (lidx1<<3);
  int alu39 = ((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))+alu37)>>2)-((int)(((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))%4)!=0)&alu34))));
  int alu40 = (alu38+alu39);
  bool alu41 = (alu40<12);
  int alu42 = (alu41?alu40:0);
  long alu43 = (((long)(((alu10*3072)+(alu6*6144))))+(((long)(alu42))<<8ll)+((long)(alu36)));
  float val2 = (*(data2_24576+alu43));
  float val3 = (*(data2_24576+(alu43+1ll)));
  float val4 = (*(data2_24576+(alu43+8ll)));
  float val5 = (*(data2_24576+(alu43+9ll)));
  float val6 = (*(data2_24576+(alu43+16ll)));
  float val7 = (*(data2_24576+(alu43+17ll)));
  float val8 = (*(data2_24576+(alu43+24ll)));
  float val9 = (*(data2_24576+(alu43+25ll)));
  float val10 = (*(data2_24576+(alu43+32ll)));
  float val11 = (*(data2_24576+(alu43+33ll)));
  float val12 = (*(data2_24576+(alu43+40ll)));
  float val13 = (*(data2_24576+(alu43+41ll)));
  float val14 = (*(data2_24576+(alu43+48ll)));
  float val15 = (*(data2_24576+(alu43+49ll)));
  float val16 = (*(data2_24576+(alu43+56ll)));
  float val17 = (*(data2_24576+(alu43+57ll)));
  float val18 = (*(data2_24576+(alu43+64ll)));
  float val19 = (*(data2_24576+(alu43+65ll)));
  float val20 = (*(data2_24576+(alu43+72ll)));
  float val21 = (*(data2_24576+(alu43+73ll)));
  float val22 = (*(data2_24576+(alu43+80ll)));
  float val23 = (*(data2_24576+(alu43+81ll)));
  float val24 = (*(data2_24576+(alu43+88ll)));
  float val25 = (*(data2_24576+(alu43+89ll)));
  float val26 = (*(data2_24576+(alu43+96ll)));
  float val27 = (*(data2_24576+(alu43+97ll)));
  float val28 = (*(data2_24576+(alu43+104ll)));
  float val29 = (*(data2_24576+(alu43+105ll)));
  float val30 = (*(data2_24576+(alu43+112ll)));
  float val31 = (*(data2_24576+(alu43+113ll)));
  float val32 = (*(data2_24576+(alu43+120ll)));
  float val33 = (*(data2_24576+(alu43+121ll)));
  float val34 = (*(data2_24576+(alu43+128ll)));
  float val35 = (*(data2_24576+(alu43+129ll)));
  float val36 = (*(data2_24576+(alu43+136ll)));
  float val37 = (*(data2_24576+(alu43+137ll)));
  float val38 = (*(data2_24576+(alu43+144ll)));
  float val39 = (*(data2_24576+(alu43+145ll)));
  float val40 = (*(data2_24576+(alu43+152ll)));
  float val41 = (*(data2_24576+(alu43+153ll)));
  float val42 = (*(data2_24576+(alu43+160ll)));
  float val43 = (*(data2_24576+(alu43+161ll)));
  float val44 = (*(data2_24576+(alu43+168ll)));
  float val45 = (*(data2_24576+(alu43+169ll)));
  float val46 = (*(data2_24576+(alu43+176ll)));
  float val47 = (*(data2_24576+(alu43+177ll)));
  float val48 = (*(data2_24576+(alu43+184ll)));
  float val49 = (*(data2_24576+(alu43+185ll)));
  float val50 = (*(data2_24576+(alu43+192ll)));
  float val51 = (*(data2_24576+(alu43+193ll)));
  float val52 = (*(data2_24576+(alu43+200ll)));
  float val53 = (*(data2_24576+(alu43+201ll)));
  float val54 = (*(data2_24576+(alu43+208ll)));
  float val55 = (*(data2_24576+(alu43+209ll)));
  float val56 = (*(data2_24576+(alu43+216ll)));
  float val57 = (*(data2_24576+(alu43+217ll)));
  float val58 = (*(data2_24576+(alu43+224ll)));
  float val59 = (*(data2_24576+(alu43+225ll)));
  float val60 = (*(data2_24576+(alu43+232ll)));
  float val61 = (*(data2_24576+(alu43+233ll)));
  float val62 = (*(data2_24576+(alu43+240ll)));
  float val63 = (*(data2_24576+(alu43+241ll)));
  float val64 = (*(data2_24576+(alu43+248ll)));
  float val65 = (*(data2_24576+(alu43+249ll)));
  __shared__ __align__(16) half buf5[256];
  float buf6;
  float buf7;
  float buf8;
  float buf9;
  float buf10;
  float buf11;
  __syncthreads();
  half alu45 = (alu41?(((half)(val2))*((half)(0.0625f))):((half)(0.0f)));
  half alu46 = (alu41?(((half)(val3))*((half)(0.0625f))):((half)(0.0f)));
  half alu47 = (alu41?(((half)(val4))*((half)(0.0625f))):((half)(0.0f)));
  half alu48 = (alu41?(((half)(val5))*((half)(0.0625f))):((half)(0.0f)));
  half alu49 = (alu41?(((half)(val6))*((half)(0.0625f))):((half)(0.0f)));
  half alu50 = (alu41?(((half)(val7))*((half)(0.0625f))):((half)(0.0f)));
  half alu51 = (alu41?(((half)(val8))*((half)(0.0625f))):((half)(0.0f)));
  half alu52 = (alu41?(((half)(val9))*((half)(0.0625f))):((half)(0.0f)));
  half alu53 = (alu41?(((half)(val10))*((half)(0.0625f))):((half)(0.0f)));
  half alu54 = (alu41?(((half)(val11))*((half)(0.0625f))):((half)(0.0f)));
  half alu55 = (alu41?(((half)(val12))*((half)(0.0625f))):((half)(0.0f)));
  half alu56 = (alu41?(((half)(val13))*((half)(0.0625f))):((half)(0.0f)));
  half alu57 = (alu41?(((half)(val14))*((half)(0.0625f))):((half)(0.0f)));
  half alu58 = (alu41?(((half)(val15))*((half)(0.0625f))):((half)(0.0f)));
  half alu59 = (alu41?(((half)(val16))*((half)(0.0625f))):((half)(0.0f)));
  half alu60 = (alu41?(((half)(val17))*((half)(0.0625f))):((half)(0.0f)));
  half alu61 = (alu41?(((half)(val18))*((half)(0.0625f))):((half)(0.0f)));
  half alu62 = (alu41?(((half)(val19))*((half)(0.0625f))):((half)(0.0f)));
  half alu63 = (alu41?(((half)(val20))*((half)(0.0625f))):((half)(0.0f)));
  half alu64 = (alu41?(((half)(val21))*((half)(0.0625f))):((half)(0.0f)));
  half alu65 = (alu41?(((half)(val22))*((half)(0.0625f))):((half)(0.0f)));
  half alu66 = (alu41?(((half)(val23))*((half)(0.0625f))):((half)(0.0f)));
  half alu67 = (alu41?(((half)(val24))*((half)(0.0625f))):((half)(0.0f)));
  half alu68 = (alu41?(((half)(val25))*((half)(0.0625f))):((half)(0.0f)));
  half alu69 = (alu41?(((half)(val26))*((half)(0.0625f))):((half)(0.0f)));
  half alu70 = (alu41?(((half)(val27))*((half)(0.0625f))):((half)(0.0f)));
  half alu71 = (alu41?(((half)(val28))*((half)(0.0625f))):((half)(0.0f)));
  half alu72 = (alu41?(((half)(val29))*((half)(0.0625f))):((half)(0.0f)));
  half alu73 = (alu41?(((half)(val30))*((half)(0.0625f))):((half)(0.0f)));
  half alu74 = (alu41?(((half)(val31))*((half)(0.0625f))):((half)(0.0f)));
  half alu75 = (alu41?(((half)(val32))*((half)(0.0625f))):((half)(0.0f)));
  half alu76 = (alu41?(((half)(val33))*((half)(0.0625f))):((half)(0.0f)));
  half alu77 = (alu41?(((half)(val34))*((half)(0.0625f))):((half)(0.0f)));
  half alu78 = (alu41?(((half)(val35))*((half)(0.0625f))):((half)(0.0f)));
  half alu79 = (alu41?(((half)(val36))*((half)(0.0625f))):((half)(0.0f)));
  half alu80 = (alu41?(((half)(val37))*((half)(0.0625f))):((half)(0.0f)));
  half alu81 = (alu41?(((half)(val38))*((half)(0.0625f))):((half)(0.0f)));
  half alu82 = (alu41?(((half)(val39))*((half)(0.0625f))):((half)(0.0f)));
  half alu83 = (alu41?(((half)(val40))*((half)(0.0625f))):((half)(0.0f)));
  half alu84 = (alu41?(((half)(val41))*((half)(0.0625f))):((half)(0.0f)));
  half alu85 = (alu41?(((half)(val42))*((half)(0.0625f))):((half)(0.0f)));
  half alu86 = (alu41?(((half)(val43))*((half)(0.0625f))):((half)(0.0f)));
  half alu87 = (alu41?(((half)(val44))*((half)(0.0625f))):((half)(0.0f)));
  half alu88 = (alu41?(((half)(val45))*((half)(0.0625f))):((half)(0.0f)));
  half alu89 = (alu41?(((half)(val46))*((half)(0.0625f))):((half)(0.0f)));
  half alu90 = (alu41?(((half)(val47))*((half)(0.0625f))):((half)(0.0f)));
  half alu91 = (alu41?(((half)(val48))*((half)(0.0625f))):((half)(0.0f)));
  half alu92 = (alu41?(((half)(val49))*((half)(0.0625f))):((half)(0.0f)));
  half alu93 = (alu41?(((half)(val50))*((half)(0.0625f))):((half)(0.0f)));
  half alu94 = (alu41?(((half)(val51))*((half)(0.0625f))):((half)(0.0f)));
  half alu95 = (alu41?(((half)(val52))*((half)(0.0625f))):((half)(0.0f)));
  half alu96 = (alu41?(((half)(val53))*((half)(0.0625f))):((half)(0.0f)));
  half alu97 = (alu41?(((half)(val54))*((half)(0.0625f))):((half)(0.0f)));
  half alu98 = (alu41?(((half)(val55))*((half)(0.0625f))):((half)(0.0f)));
  half alu99 = (alu41?(((half)(val56))*((half)(0.0625f))):((half)(0.0f)));
  half alu100 = (alu41?(((half)(val57))*((half)(0.0625f))):((half)(0.0f)));
  half alu101 = (alu41?(((half)(val58))*((half)(0.0625f))):((half)(0.0f)));
  half alu102 = (alu41?(((half)(val59))*((half)(0.0625f))):((half)(0.0f)));
  half alu103 = (alu41?(((half)(val60))*((half)(0.0625f))):((half)(0.0f)));
  half alu104 = (alu41?(((half)(val61))*((half)(0.0625f))):((half)(0.0f)));
  half alu105 = (alu41?(((half)(val62))*((half)(0.0625f))):((half)(0.0f)));
  half alu106 = (alu41?(((half)(val63))*((half)(0.0625f))):((half)(0.0f)));
  half alu107 = (alu41?(((half)(val64))*((half)(0.0625f))):((half)(0.0f)));
  half alu108 = (alu41?(((half)(val65))*((half)(0.0625f))):((half)(0.0f)));
  int alu109 = (lidx1<<7);
  long cast0 = ((long)((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))));
  long alu110 = (((long)(alu109))+(cast0<<1ll));
  long alu111 = (cast0+32ll);
  bool alu112 = (alu111<0ll);
  long alu113 = (alu112?127ll:0ll);
  int cast1 = ((int)((((alu111+alu113)>>7ll)-((long)((((alu111%128ll)!=0ll)&(alu112!=0)))))));
  int alu114 = (cast1*34);
  int alu115 = (alu114+((alu39+8)&31)+2);
  bool alu116 = (alu115<0);
  int alu117 = (alu114+((alu39+72)&31)+2);
  bool alu118 = (alu117<0);
  int alu119 = (alu114+((alu39+136)&31)+2);
  bool alu120 = (alu119<0);
  int alu121 = (alu114+((alu39+200)&31)+2);
  bool alu122 = (alu121<0);
  long alu123 = (cast0+64ll);
  bool alu124 = (alu123<0ll);
  long alu125 = (alu124?127ll:0ll);
  int cast2 = ((int)((((alu123+alu125)>>7ll)-((long)((((alu123%128ll)!=0ll)&(alu124!=0)))))));
  int alu126 = (cast2*34);
  int alu127 = (alu126+((alu39+16)&31)+2);
  bool alu128 = (alu127<0);
  int alu129 = (alu126+((alu39+80)&31)+2);
  bool alu130 = (alu129<0);
  int alu131 = (alu126+((alu39+144)&31)+2);
  bool alu132 = (alu131<0);
  int alu133 = (alu126+((alu39+208)&31)+2);
  bool alu134 = (alu133<0);
  long alu135 = (cast0+96ll);
  bool alu136 = (alu135<0ll);
  long alu137 = (alu136?127ll:0ll);
  int cast3 = ((int)((((alu135+alu137)>>7ll)-((long)((((alu135%128ll)!=0ll)&(alu136!=0)))))));
  int alu138 = (cast3*34);
  int alu139 = (alu138+((alu39+24)&31)+2);
  bool alu140 = (alu139<0);
  int alu141 = (alu138+((alu39+88)&31)+2);
  bool alu142 = (alu141<0);
  int alu143 = (alu138+((alu39+152)&31)+2);
  bool alu144 = (alu143<0);
  int alu145 = (alu138+((alu39+216)&31)+2);
  bool alu146 = (alu145<0);
  int alu147 = (alu33?127:0);
  int alu148 = ((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))+alu147)>>7)-((int)(((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))%128)!=0)&alu34))));
  int alu149 = (alu148*34);
  int alu150 = (alu149+((alu39+64)&31)+2);
  bool alu151 = (alu150<0);
  int alu152 = (alu149+((alu39+128)&31)+2);
  bool alu153 = (alu152<0);
  int alu154 = (alu149+((alu39+192)&31)+2);
  bool alu155 = (alu154<0);
  int alu156 = (alu149+(alu39&31)+2);
  bool alu157 = (alu156<0);
  int alu158 = (cast1*17);
  int alu159 = (alu158+1);
  bool alu160 = (alu159<0);
  int alu161 = (alu114+((alu39+40)&31));
  bool alu162 = (alu161<0);
  int alu163 = (alu114+((alu39+104)&31));
  bool alu164 = (alu163<0);
  int alu165 = (alu114+((alu39+168)&31));
  bool alu166 = (alu165<0);
  int alu167 = (alu114+((alu39+232)&31));
  bool alu168 = (alu167<0);
  int alu169 = (cast2*17);
  int alu170 = (alu169+1);
  bool alu171 = (alu170<0);
  int alu172 = (alu126+((alu39+48)&31));
  bool alu173 = (alu172<0);
  int alu174 = (alu126+((alu39+112)&31));
  bool alu175 = (alu174<0);
  int alu176 = (alu126+((alu39+176)&31));
  bool alu177 = (alu176<0);
  int alu178 = (alu126+((alu39+240)&31));
  bool alu179 = (alu178<0);
  int alu180 = (cast3*17);
  int alu181 = (alu180+1);
  bool alu182 = (alu181<0);
  int alu183 = (alu138+((alu39+56)&31));
  bool alu184 = (alu183<0);
  int alu185 = (alu138+((alu39+120)&31));
  bool alu186 = (alu185<0);
  int alu187 = (alu138+((alu39+184)&31));
  bool alu188 = (alu187<0);
  int alu189 = (alu138+((alu39+248)&31));
  bool alu190 = (alu189<0);
  int alu191 = (alu148*17);
  int alu192 = (alu191+1);
  bool alu193 = (alu192<0);
  int alu194 = (alu149+((alu39+32)&31));
  bool alu195 = (alu194<0);
  int alu196 = (alu149+((alu39+96)&31));
  bool alu197 = (alu196<0);
  int alu198 = (alu149+((alu39+160)&31));
  bool alu199 = (alu198<0);
  int alu200 = (alu149+((alu39+224)&31));
  bool alu201 = (alu200<0);
  bool alu202 = (alu158<0);
  bool alu203 = (alu169<0);
  bool alu204 = (alu180<0);
  bool alu205 = (alu191<0);
  int alu206 = (alu33?1:0);
  int alu207 = (alu116?3:0);
  int alu208 = (alu118?3:0);
  int alu209 = (alu120?3:0);
  int alu210 = (alu122?3:0);
  int alu211 = (alu128?3:0);
  int alu212 = (alu130?3:0);
  int alu213 = (alu132?3:0);
  int alu214 = (alu134?3:0);
  int alu215 = (alu140?3:0);
  int alu216 = (alu142?3:0);
  int alu217 = (alu144?3:0);
  int alu218 = (alu146?3:0);
  int alu219 = (alu151?3:0);
  int alu220 = (alu153?3:0);
  int alu221 = (alu155?3:0);
  int alu222 = (alu157?3:0);
  int alu223 = (alu109+(alu35<<4)+alu39);
  int alu224 = (alu160?1:0);
  int alu225 = (alu162?3:0);
  int alu226 = (alu164?3:0);
  int alu227 = (alu166?3:0);
  int alu228 = (alu168?3:0);
  int alu229 = (alu114+alu39);
  int alu230 = (alu171?1:0);
  int alu231 = (alu173?3:0);
  int alu232 = (alu175?3:0);
  int alu233 = (alu177?3:0);
  int alu234 = (alu179?3:0);
  int alu235 = (alu126+alu39);
  int alu236 = (alu182?1:0);
  int alu237 = (alu184?3:0);
  int alu238 = (alu186?3:0);
  int alu239 = (alu188?3:0);
  int alu240 = (alu190?3:0);
  int alu241 = (alu138+alu39);
  int alu242 = (alu193?1:0);
  int alu243 = (alu195?3:0);
  int alu244 = (alu197?3:0);
  int alu245 = (alu199?3:0);
  int alu246 = (alu201?3:0);
  int alu247 = (alu149+alu39);
  int alu248 = (alu202?1:0);
  int alu249 = (alu203?1:0);
  int alu250 = (alu204?1:0);
  int alu251 = (alu205?1:0);
  int alu252 = (((int)(((cast0+1ll)&1ll)))<<4);
  int alu253 = (alu252+8);
  int alu254 = (((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))&1)<<4);
  int alu255 = (alu254+8);
  int alu256 = (((alu229+10)&3)<<3);
  int alu257 = (((alu229+76)&3)<<3);
  int alu258 = (((alu229+142)&3)<<3);
  int alu259 = (((alu229+208)&3)<<3);
  int alu260 = (((alu229+274)&3)<<3);
  int alu261 = (((alu229+340)&3)<<3);
  int alu262 = (((alu229+406)&3)<<3);
  int alu263 = (((alu229+472)&3)<<3);
  int alu264 = (((alu235+18)&3)<<3);
  int alu265 = (((alu235+84)&3)<<3);
  int alu266 = (((alu235+150)&3)<<3);
  int alu267 = (((alu235+216)&3)<<3);
  int alu268 = (((alu235+282)&3)<<3);
  int alu269 = (((alu235+348)&3)<<3);
  int alu270 = (((alu235+414)&3)<<3);
  int alu271 = (((alu235+480)&3)<<3);
  int alu272 = (((alu241+26)&3)<<3);
  int alu273 = (((alu241+92)&3)<<3);
  int alu274 = (((alu241+158)&3)<<3);
  int alu275 = (((alu241+224)&3)<<3);
  int alu276 = (((alu241+290)&3)<<3);
  int alu277 = (((alu241+356)&3)<<3);
  int alu278 = (((alu241+422)&3)<<3);
  int alu279 = (((alu241+488)&3)<<3);
  int alu280 = (((alu247+2)&3)<<3);
  int alu281 = (((alu247+68)&3)<<3);
  int alu282 = (((alu247+134)&3)<<3);
  int alu283 = (((alu247+200)&3)<<3);
  int alu284 = (((alu247+266)&3)<<3);
  int alu285 = (((alu247+332)&3)<<3);
  int alu286 = (((alu247+398)&3)<<3);
  int alu287 = (((alu247+464)&3)<<3);
  int alu288 = (((alu158+17)&1)<<4);
  int alu289 = (((alu158+34)&1)<<4);
  int alu290 = (((alu158+51)&1)<<4);
  int alu291 = (((alu158+68)&1)<<4);
  int alu292 = (((alu158+85)&1)<<4);
  int alu293 = (((alu158+102)&1)<<4);
  int alu294 = (((alu158+119)&1)<<4);
  int alu295 = (((alu169+17)&1)<<4);
  int alu296 = (((alu169+34)&1)<<4);
  int alu297 = (((alu169+51)&1)<<4);
  int alu298 = (((alu169+68)&1)<<4);
  int alu299 = (((alu169+85)&1)<<4);
  int alu300 = (((alu169+102)&1)<<4);
  int alu301 = (((alu169+119)&1)<<4);
  int alu302 = (((alu180+17)&1)<<4);
  int alu303 = (((alu180+34)&1)<<4);
  int alu304 = (((alu180+51)&1)<<4);
  int alu305 = (((alu180+68)&1)<<4);
  int alu306 = (((alu180+85)&1)<<4);
  int alu307 = (((alu180+102)&1)<<4);
  int alu308 = (((alu180+119)&1)<<4);
  int alu309 = (((alu191+17)&1)<<4);
  int alu310 = (((alu191+34)&1)<<4);
  int alu311 = (((alu191+51)&1)<<4);
  int alu312 = (((alu191+68)&1)<<4);
  int alu313 = (((alu191+85)&1)<<4);
  int alu314 = (((alu191+102)&1)<<4);
  int alu315 = (((alu191+119)&1)<<4);
  int alu316 = ((alu158&1)<<4);
  int alu317 = ((alu169&1)<<4);
  int alu318 = ((alu180&1)<<4);
  int alu319 = ((alu191&1)<<4);
  for (int Ridx4 = 0; Ridx4 < 4; Ridx4++) {
    int alu320 = (Ridx4*1088);
    long alu321 = ((((long)(alu39))*68ll)+((long)(alu320)));
    long alu322 = (alu321+((long)(((alu35+1)>>1))));
    int val66 = (*(buf3+(alu322+2ll)));
    int val67 = (*(buf3+(alu322+4ll)));
    int val68 = (*(buf3+(alu322+6ll)));
    int val69 = (*(buf3+(alu322+17ll)));
    int val70 = (*(buf3+(alu322+19ll)));
    int val71 = (*(buf3+(alu322+21ll)));
    int val72 = (*(buf3+(alu322+23ll)));
    int val73 = (*(buf3+(alu322+34ll)));
    int val74 = (*(buf3+(alu322+36ll)));
    int val75 = (*(buf3+(alu322+38ll)));
    int val76 = (*(buf3+(alu322+40ll)));
    int val77 = (*(buf3+(alu322+51ll)));
    int val78 = (*(buf3+(alu322+53ll)));
    int val79 = (*(buf3+(alu322+55ll)));
    int val80 = (*(buf3+(alu322+57ll)));
    int val81 = (*(buf3+(alu322+544ll)));
    int val82 = (*(buf3+(alu322+546ll)));
    int val83 = (*(buf3+(alu322+548ll)));
    int val84 = (*(buf3+(alu322+550ll)));
    int val85 = (*(buf3+(alu322+561ll)));
    int val86 = (*(buf3+(alu322+563ll)));
    int val87 = (*(buf3+(alu322+565ll)));
    int val88 = (*(buf3+(alu322+567ll)));
    int val89 = (*(buf3+(alu322+578ll)));
    int val90 = (*(buf3+(alu322+580ll)));
    int val91 = (*(buf3+(alu322+582ll)));
    int val92 = (*(buf3+(alu322+584ll)));
    int val93 = (*(buf3+(alu322+595ll)));
    int val94 = (*(buf3+(alu322+597ll)));
    int val95 = (*(buf3+(alu322+599ll)));
    int val96 = (*(buf3+(alu322+601ll)));
    long alu323 = (alu321+((long)((((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))+alu206)>>1)-((int)(((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))%2)!=0)&alu34))))&1))));
    int val97 = (*(buf3+(alu323+9ll)));
    int val98 = (*(buf3+(alu323+11ll)));
    int val99 = (*(buf3+(alu323+13ll)));
    int val100 = (*(buf3+(alu323+15ll)));
    int val101 = (*(buf3+(alu323+26ll)));
    int val102 = (*(buf3+(alu323+28ll)));
    int val103 = (*(buf3+(alu323+30ll)));
    int val104 = (*(buf3+(alu323+32ll)));
    int val105 = (*(buf3+(alu323+43ll)));
    int val106 = (*(buf3+(alu323+45ll)));
    int val107 = (*(buf3+(alu323+47ll)));
    int val108 = (*(buf3+(alu323+49ll)));
    int val109 = (*(buf3+(alu323+60ll)));
    int val110 = (*(buf3+(alu323+62ll)));
    int val111 = (*(buf3+(alu323+64ll)));
    int val112 = (*(buf3+(alu323+66ll)));
    int val113 = (*(buf3+(alu323+553ll)));
    int val114 = (*(buf3+(alu323+555ll)));
    int val115 = (*(buf3+(alu323+557ll)));
    int val116 = (*(buf3+(alu323+559ll)));
    int val117 = (*(buf3+(alu323+570ll)));
    int val118 = (*(buf3+(alu323+572ll)));
    int val119 = (*(buf3+(alu323+574ll)));
    int val120 = (*(buf3+(alu323+576ll)));
    int val121 = (*(buf3+(alu323+587ll)));
    int val122 = (*(buf3+(alu323+589ll)));
    int val123 = (*(buf3+(alu323+591ll)));
    int val124 = (*(buf3+(alu323+593ll)));
    int val125 = (*(buf3+(alu323+604ll)));
    int val126 = (*(buf3+(alu323+606ll)));
    int val127 = (*(buf3+(alu323+608ll)));
    int val128 = (*(buf3+(alu323+610ll)));
    int val129 = (*(buf3+alu322));
    int val130 = (*(buf3+(alu321+8ll)));
    int val131 = (*(buf3+(alu321+17ll)));
    int val132 = (*(buf3+(alu321+25ll)));
    int val133 = (*(buf3+(alu321+34ll)));
    int val134 = (*(buf3+(alu321+42ll)));
    int val135 = (*(buf3+(alu321+51ll)));
    int val136 = (*(buf3+(alu321+59ll)));
    int val137 = (*(buf3+(alu321+544ll)));
    int val138 = (*(buf3+(alu321+552ll)));
    int val139 = (*(buf3+(alu321+561ll)));
    int val140 = (*(buf3+(alu321+569ll)));
    int val141 = (*(buf3+(alu321+578ll)));
    int val142 = (*(buf3+(alu321+586ll)));
    int val143 = (*(buf3+(alu321+595ll)));
    int val144 = (*(buf3+(alu321+603ll)));
    int val145 = (*(buf3+alu321));
    half cast4 = tg_bitcast<half>((unsigned short)(((unsigned short)((val137>>0)))));
    half cast5 = tg_bitcast<half>((unsigned short)(((unsigned short)((val145>>0)))));
    half cast6 = tg_bitcast<half>((unsigned short)(((unsigned short)((val131>>0)))));
    half cast7 = tg_bitcast<half>((unsigned short)(((unsigned short)((val139>>0)))));
    half cast8 = tg_bitcast<half>((unsigned short)(((unsigned short)((val133>>0)))));
    half cast9 = tg_bitcast<half>((unsigned short)(((unsigned short)((val141>>0)))));
    half cast10 = tg_bitcast<half>((unsigned short)(((unsigned short)((val135>>0)))));
    half cast11 = tg_bitcast<half>((unsigned short)(((unsigned short)((val143>>0)))));
    half cast12 = tg_bitcast<half>((unsigned short)(((unsigned short)((val130>>16)))));
    half cast13 = tg_bitcast<half>((unsigned short)(((unsigned short)((val138>>16)))));
    half cast14 = tg_bitcast<half>((unsigned short)(((unsigned short)((val132>>16)))));
    half cast15 = tg_bitcast<half>((unsigned short)(((unsigned short)((val140>>16)))));
    half cast16 = tg_bitcast<half>((unsigned short)(((unsigned short)((val134>>16)))));
    half cast17 = tg_bitcast<half>((unsigned short)(((unsigned short)((val142>>16)))));
    half cast18 = tg_bitcast<half>((unsigned short)(((unsigned short)((val136>>16)))));
    half cast19 = tg_bitcast<half>((unsigned short)(((unsigned short)((val144>>16)))));
    float4 wmma0 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val129>>alu252)))))*cast5),(((half)(((signed char)((val129>>alu253)))))*cast5),(((half)(((signed char)((val81>>alu252)))))*cast4),(((half)(((signed char)((val81>>alu253)))))*cast4),(((half)(((signed char)((val66>>alu252)))))*cast5),(((half)(((signed char)((val66>>alu253)))))*cast5),(((half)(((signed char)((val82>>alu252)))))*cast4),(((half)(((signed char)((val82>>alu253)))))*cast4)), make_half4(alu45,alu46,alu47,alu48), make_float4(0.0f,0.0f,0.0f,0.0f));
    float4 wmma1 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val67>>alu252)))))*cast5),(((half)(((signed char)((val67>>alu253)))))*cast5),(((half)(((signed char)((val83>>alu252)))))*cast4),(((half)(((signed char)((val83>>alu253)))))*cast4),(((half)(((signed char)((val68>>alu252)))))*cast5),(((half)(((signed char)((val68>>alu253)))))*cast5),(((half)(((signed char)((val84>>alu252)))))*cast4),(((half)(((signed char)((val84>>alu253)))))*cast4)), make_half4(alu49,alu50,alu51,alu52), wmma0);
    float4 wmma2 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val97>>alu254)))))*cast12),(((half)(((signed char)((val97>>alu255)))))*cast12),(((half)(((signed char)((val113>>alu254)))))*cast13),(((half)(((signed char)((val113>>alu255)))))*cast13),(((half)(((signed char)((val98>>alu254)))))*cast12),(((half)(((signed char)((val98>>alu255)))))*cast12),(((half)(((signed char)((val114>>alu254)))))*cast13),(((half)(((signed char)((val114>>alu255)))))*cast13)), make_half4(alu53,alu54,alu55,alu56), wmma1);
    float4 wmma3 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val99>>alu254)))))*cast12),(((half)(((signed char)((val99>>alu255)))))*cast12),(((half)(((signed char)((val115>>alu254)))))*cast13),(((half)(((signed char)((val115>>alu255)))))*cast13),(((half)(((signed char)((val100>>alu254)))))*cast12),(((half)(((signed char)((val100>>alu255)))))*cast12),(((half)(((signed char)((val116>>alu254)))))*cast13),(((half)(((signed char)((val116>>alu255)))))*cast13)), make_half4(alu57,alu58,alu59,alu60), wmma2);
    float4 wmma4 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val69>>alu252)))))*cast6),(((half)(((signed char)((val69>>alu253)))))*cast6),(((half)(((signed char)((val85>>alu252)))))*cast7),(((half)(((signed char)((val85>>alu253)))))*cast7),(((half)(((signed char)((val70>>alu252)))))*cast6),(((half)(((signed char)((val70>>alu253)))))*cast6),(((half)(((signed char)((val86>>alu252)))))*cast7),(((half)(((signed char)((val86>>alu253)))))*cast7)), make_half4(alu61,alu62,alu63,alu64), wmma3);
    float4 wmma5 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val71>>alu252)))))*cast6),(((half)(((signed char)((val71>>alu253)))))*cast6),(((half)(((signed char)((val87>>alu252)))))*cast7),(((half)(((signed char)((val87>>alu253)))))*cast7),(((half)(((signed char)((val72>>alu252)))))*cast6),(((half)(((signed char)((val72>>alu253)))))*cast6),(((half)(((signed char)((val88>>alu252)))))*cast7),(((half)(((signed char)((val88>>alu253)))))*cast7)), make_half4(alu65,alu66,alu67,alu68), wmma4);
    float4 wmma6 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val101>>alu254)))))*cast14),(((half)(((signed char)((val101>>alu255)))))*cast14),(((half)(((signed char)((val117>>alu254)))))*cast15),(((half)(((signed char)((val117>>alu255)))))*cast15),(((half)(((signed char)((val102>>alu254)))))*cast14),(((half)(((signed char)((val102>>alu255)))))*cast14),(((half)(((signed char)((val118>>alu254)))))*cast15),(((half)(((signed char)((val118>>alu255)))))*cast15)), make_half4(alu69,alu70,alu71,alu72), wmma5);
    float4 wmma7 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val103>>alu254)))))*cast14),(((half)(((signed char)((val103>>alu255)))))*cast14),(((half)(((signed char)((val119>>alu254)))))*cast15),(((half)(((signed char)((val119>>alu255)))))*cast15),(((half)(((signed char)((val104>>alu254)))))*cast14),(((half)(((signed char)((val104>>alu255)))))*cast14),(((half)(((signed char)((val120>>alu254)))))*cast15),(((half)(((signed char)((val120>>alu255)))))*cast15)), make_half4(alu73,alu74,alu75,alu76), wmma6);
    float4 wmma8 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val73>>alu252)))))*cast8),(((half)(((signed char)((val73>>alu253)))))*cast8),(((half)(((signed char)((val89>>alu252)))))*cast9),(((half)(((signed char)((val89>>alu253)))))*cast9),(((half)(((signed char)((val74>>alu252)))))*cast8),(((half)(((signed char)((val74>>alu253)))))*cast8),(((half)(((signed char)((val90>>alu252)))))*cast9),(((half)(((signed char)((val90>>alu253)))))*cast9)), make_half4(alu77,alu78,alu79,alu80), wmma7);
    float4 wmma9 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val75>>alu252)))))*cast8),(((half)(((signed char)((val75>>alu253)))))*cast8),(((half)(((signed char)((val91>>alu252)))))*cast9),(((half)(((signed char)((val91>>alu253)))))*cast9),(((half)(((signed char)((val76>>alu252)))))*cast8),(((half)(((signed char)((val76>>alu253)))))*cast8),(((half)(((signed char)((val92>>alu252)))))*cast9),(((half)(((signed char)((val92>>alu253)))))*cast9)), make_half4(alu81,alu82,alu83,alu84), wmma8);
    float4 wmma10 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val105>>alu254)))))*cast16),(((half)(((signed char)((val105>>alu255)))))*cast16),(((half)(((signed char)((val121>>alu254)))))*cast17),(((half)(((signed char)((val121>>alu255)))))*cast17),(((half)(((signed char)((val106>>alu254)))))*cast16),(((half)(((signed char)((val106>>alu255)))))*cast16),(((half)(((signed char)((val122>>alu254)))))*cast17),(((half)(((signed char)((val122>>alu255)))))*cast17)), make_half4(alu85,alu86,alu87,alu88), wmma9);
    float4 wmma11 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val107>>alu254)))))*cast16),(((half)(((signed char)((val107>>alu255)))))*cast16),(((half)(((signed char)((val123>>alu254)))))*cast17),(((half)(((signed char)((val123>>alu255)))))*cast17),(((half)(((signed char)((val108>>alu254)))))*cast16),(((half)(((signed char)((val108>>alu255)))))*cast16),(((half)(((signed char)((val124>>alu254)))))*cast17),(((half)(((signed char)((val124>>alu255)))))*cast17)), make_half4(alu89,alu90,alu91,alu92), wmma10);
    float4 wmma12 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val77>>alu252)))))*cast10),(((half)(((signed char)((val77>>alu253)))))*cast10),(((half)(((signed char)((val93>>alu252)))))*cast11),(((half)(((signed char)((val93>>alu253)))))*cast11),(((half)(((signed char)((val78>>alu252)))))*cast10),(((half)(((signed char)((val78>>alu253)))))*cast10),(((half)(((signed char)((val94>>alu252)))))*cast11),(((half)(((signed char)((val94>>alu253)))))*cast11)), make_half4(alu93,alu94,alu95,alu96), wmma11);
    float4 wmma13 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val79>>alu252)))))*cast10),(((half)(((signed char)((val79>>alu253)))))*cast10),(((half)(((signed char)((val95>>alu252)))))*cast11),(((half)(((signed char)((val95>>alu253)))))*cast11),(((half)(((signed char)((val80>>alu252)))))*cast10),(((half)(((signed char)((val80>>alu253)))))*cast10),(((half)(((signed char)((val96>>alu252)))))*cast11),(((half)(((signed char)((val96>>alu253)))))*cast11)), make_half4(alu97,alu98,alu99,alu100), wmma12);
    float4 wmma14 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val109>>alu254)))))*cast18),(((half)(((signed char)((val109>>alu255)))))*cast18),(((half)(((signed char)((val125>>alu254)))))*cast19),(((half)(((signed char)((val125>>alu255)))))*cast19),(((half)(((signed char)((val110>>alu254)))))*cast18),(((half)(((signed char)((val110>>alu255)))))*cast18),(((half)(((signed char)((val126>>alu254)))))*cast19),(((half)(((signed char)((val126>>alu255)))))*cast19)), make_half4(alu101,alu102,alu103,alu104), wmma13);
    float4 wmma15 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val111>>alu254)))))*cast18),(((half)(((signed char)((val111>>alu255)))))*cast18),(((half)(((signed char)((val127>>alu254)))))*cast19),(((half)(((signed char)((val127>>alu255)))))*cast19),(((half)(((signed char)((val112>>alu254)))))*cast18),(((half)(((signed char)((val112>>alu255)))))*cast18),(((half)(((signed char)((val128>>alu254)))))*cast19),(((half)(((signed char)((val128>>alu255)))))*cast19)), make_half4(alu105,alu106,alu107,alu108), wmma14);
    int alu324 = (alu15+(Ridx4<<6)+alu16+alu39);
    bool alu325 = ((alu13+Ridx4)<alu21);
    bool alu326 = (alu325&((alu324+8)<val0));
    float alu327 = (alu326?wmma15.z:((float)(-INFINITY)));
    float alu328 = (alu327+2.0792999267578125f);
    bool alu329 = (alu325&(alu324<val0));
    float alu330 = (alu329?wmma15.x:((float)(-INFINITY)));
    float alu331 = (alu330+2.0792999267578125f);
    float alu332 = ((alu331<alu328)?alu328:alu331);
    float alu333 = (((*(buf1+0))<alu332)?alu332:(*(buf1+0)));
    buf6 = __shfl_xor_sync(0xffffffffu, alu333, 16);
    float alu335 = ((alu333<buf6)?buf6:alu333);
    buf7 = __shfl_xor_sync(0xffffffffu, alu335, 8);
    float alu337 = ((alu335<buf7)?buf7:alu335);
    buf8 = __shfl_xor_sync(0xffffffffu, alu337, 4);
    float alu339 = (alu326?wmma15.w:((float)(-INFINITY)));
    float alu340 = (alu339+2.0792999267578125f);
    float alu341 = (alu329?wmma15.y:((float)(-INFINITY)));
    float alu342 = (alu341+2.0792999267578125f);
    float alu343 = ((alu342<alu340)?alu340:alu342);
    float alu344 = (((*(buf1+1))<alu343)?alu343:(*(buf1+1)));
    buf9 = __shfl_xor_sync(0xffffffffu, alu344, 16);
    float alu346 = ((alu344<buf9)?buf9:alu344);
    buf10 = __shfl_xor_sync(0xffffffffu, alu346, 8);
    float alu348 = ((alu346<buf10)?buf10:alu346);
    buf11 = __shfl_xor_sync(0xffffffffu, alu348, 4);
    float alu350 = ((alu337<buf8)?buf8:alu337);
    float alu351 = exp2(((alu327-alu350)*1.4426950216293335f));
    float alu352 = ((alu348<buf11)?buf11:alu348);
    float alu353 = exp2(((alu339-alu352)*1.4426950216293335f));
    float alu354 = exp2(((alu330-alu350)*1.4426950216293335f));
    float alu355 = exp2(((alu341-alu352)*1.4426950216293335f));
    float alu356 = ((*(buf1+0))-alu350);
    float alu357 = exp2((alu356*1.4426950216293335f));
    float alu358 = ((*(buf1+1))-alu352);
    float alu359 = exp2((alu358*1.4426950216293335f));
    bool alu360 = (alu356<-20.0f);
    bool alu361 = (alu358<-20.0f);
    float alu362 = (alu360?0.0f:alu357);
    float alu363 = (alu361?0.0f:alu359);
    *(buf5+alu110) = ((half)(alu354));
    *(buf5+(alu110+1ll)) = ((half)(alu355));
    *(buf5+(alu110+64ll)) = ((half)(alu351));
    *(buf5+(alu110+65ll)) = ((half)(alu353));
    (*(buf1+0)) = alu350;
    (*(buf1+1)) = alu352;
    (*(buf2+0)) = fmaf(alu362, (*(buf2+0)), (alu354+alu351));
    (*(buf2+1)) = fmaf(alu363, (*(buf2+1)), (alu355+alu353));
    __syncthreads();
    int alu373 = ((alu35*136)+alu320);
    int alu374 = (alu373+(((alu115+alu207)>>2)-((int)((((alu115%4)!=0)&(alu116!=0))))));
    int val146 = (*(buf4+((long)((alu374+68)))));
    int val147 = (*(buf4+((long)((alu374+544)))));
    int val148 = (*(buf4+((long)((alu374+612)))));
    int alu375 = (alu373+(((alu117+alu208)>>2)-((int)((((alu117%4)!=0)&(alu118!=0))))));
    int val149 = (*(buf4+((long)((alu375+17)))));
    int val150 = (*(buf4+((long)((alu375+85)))));
    int val151 = (*(buf4+((long)((alu375+561)))));
    int val152 = (*(buf4+((long)((alu375+629)))));
    int alu376 = (alu373+(((alu119+alu209)>>2)-((int)((((alu119%4)!=0)&(alu120!=0))))));
    int val153 = (*(buf4+((long)((alu376+34)))));
    int val154 = (*(buf4+((long)((alu376+102)))));
    int val155 = (*(buf4+((long)((alu376+578)))));
    int val156 = (*(buf4+((long)((alu376+646)))));
    int alu377 = (alu373+(((alu121+alu210)>>2)-((int)((((alu121%4)!=0)&(alu122!=0))))));
    int val157 = (*(buf4+((long)((alu377+51)))));
    int val158 = (*(buf4+((long)((alu377+119)))));
    int val159 = (*(buf4+((long)((alu377+595)))));
    int val160 = (*(buf4+((long)((alu377+663)))));
    int alu378 = (alu373+(((alu127+alu211)>>2)-((int)((((alu127%4)!=0)&(alu128!=0))))));
    int val161 = (*(buf4+((long)((alu378+68)))));
    int val162 = (*(buf4+((long)((alu378+544)))));
    int val163 = (*(buf4+((long)((alu378+612)))));
    int alu379 = (alu373+(((alu129+alu212)>>2)-((int)((((alu129%4)!=0)&(alu130!=0))))));
    int val164 = (*(buf4+((long)((alu379+17)))));
    int val165 = (*(buf4+((long)((alu379+85)))));
    int val166 = (*(buf4+((long)((alu379+561)))));
    int val167 = (*(buf4+((long)((alu379+629)))));
    int alu380 = (alu373+(((alu131+alu213)>>2)-((int)((((alu131%4)!=0)&(alu132!=0))))));
    int val168 = (*(buf4+((long)((alu380+34)))));
    int val169 = (*(buf4+((long)((alu380+102)))));
    int val170 = (*(buf4+((long)((alu380+578)))));
    int val171 = (*(buf4+((long)((alu380+646)))));
    int alu381 = (alu373+(((alu133+alu214)>>2)-((int)((((alu133%4)!=0)&(alu134!=0))))));
    int val172 = (*(buf4+((long)((alu381+51)))));
    int val173 = (*(buf4+((long)((alu381+119)))));
    int val174 = (*(buf4+((long)((alu381+595)))));
    int val175 = (*(buf4+((long)((alu381+663)))));
    int alu382 = (alu373+(((alu139+alu215)>>2)-((int)((((alu139%4)!=0)&(alu140!=0))))));
    int val176 = (*(buf4+((long)((alu382+68)))));
    int val177 = (*(buf4+((long)((alu382+544)))));
    int val178 = (*(buf4+((long)((alu382+612)))));
    int alu383 = (alu373+(((alu141+alu216)>>2)-((int)((((alu141%4)!=0)&(alu142!=0))))));
    int val179 = (*(buf4+((long)((alu383+17)))));
    int val180 = (*(buf4+((long)((alu383+85)))));
    int val181 = (*(buf4+((long)((alu383+561)))));
    int val182 = (*(buf4+((long)((alu383+629)))));
    int alu384 = (alu373+(((alu143+alu217)>>2)-((int)((((alu143%4)!=0)&(alu144!=0))))));
    int val183 = (*(buf4+((long)((alu384+34)))));
    int val184 = (*(buf4+((long)((alu384+102)))));
    int val185 = (*(buf4+((long)((alu384+578)))));
    int val186 = (*(buf4+((long)((alu384+646)))));
    int alu385 = (alu373+(((alu145+alu218)>>2)-((int)((((alu145%4)!=0)&(alu146!=0))))));
    int val187 = (*(buf4+((long)((alu385+51)))));
    int val188 = (*(buf4+((long)((alu385+119)))));
    int val189 = (*(buf4+((long)((alu385+595)))));
    int val190 = (*(buf4+((long)((alu385+663)))));
    int alu386 = (alu373+(((alu159+alu224)>>1)-((int)((((alu159%2)!=0)&(alu160!=0))))));
    int val191 = (*(buf4+((long)((alu386+8)))));
    int val192 = (*(buf4+((long)((alu386+25)))));
    int val193 = (*(buf4+((long)((alu386+42)))));
    int val194 = (*(buf4+((long)((alu386+59)))));
    int val195 = (*(buf4+((long)((alu386+76)))));
    int val196 = (*(buf4+((long)((alu386+93)))));
    int val197 = (*(buf4+((long)((alu386+110)))));
    int val198 = (*(buf4+((long)((alu386+127)))));
    int val199 = (*(buf4+((long)((alu386+552)))));
    int val200 = (*(buf4+((long)((alu386+569)))));
    int val201 = (*(buf4+((long)((alu386+586)))));
    int val202 = (*(buf4+((long)((alu386+603)))));
    int val203 = (*(buf4+((long)((alu386+620)))));
    int val204 = (*(buf4+((long)((alu386+637)))));
    int val205 = (*(buf4+((long)((alu386+654)))));
    int val206 = (*(buf4+((long)((alu386+671)))));
    int alu387 = (alu373+(((alu161+alu225)>>2)-((int)((((alu161%4)!=0)&(alu162!=0))))));
    int val207 = (*(buf4+((long)((alu387+9)))));
    int val208 = (*(buf4+((long)((alu387+77)))));
    int val209 = (*(buf4+((long)((alu387+553)))));
    int val210 = (*(buf4+((long)((alu387+621)))));
    int alu388 = (alu373+(((alu163+alu226)>>2)-((int)((((alu163%4)!=0)&(alu164!=0))))));
    int val211 = (*(buf4+((long)((alu388+26)))));
    int val212 = (*(buf4+((long)((alu388+94)))));
    int val213 = (*(buf4+((long)((alu388+570)))));
    int val214 = (*(buf4+((long)((alu388+638)))));
    int alu389 = (alu373+(((alu165+alu227)>>2)-((int)((((alu165%4)!=0)&(alu166!=0))))));
    int val215 = (*(buf4+((long)((alu389+43)))));
    int val216 = (*(buf4+((long)((alu389+111)))));
    int val217 = (*(buf4+((long)((alu389+587)))));
    int val218 = (*(buf4+((long)((alu389+655)))));
    int alu390 = (alu373+(((alu167+alu228)>>2)-((int)((((alu167%4)!=0)&(alu168!=0))))));
    int val219 = (*(buf4+((long)((alu390+60)))));
    int val220 = (*(buf4+((long)((alu390+128)))));
    int val221 = (*(buf4+((long)((alu390+604)))));
    int val222 = (*(buf4+((long)((alu390+672)))));
    int alu391 = (alu373+(((alu170+alu230)>>1)-((int)((((alu170%2)!=0)&(alu171!=0))))));
    int val223 = (*(buf4+((long)((alu391+8)))));
    int val224 = (*(buf4+((long)((alu391+25)))));
    int val225 = (*(buf4+((long)((alu391+42)))));
    int val226 = (*(buf4+((long)((alu391+59)))));
    int val227 = (*(buf4+((long)((alu391+76)))));
    int val228 = (*(buf4+((long)((alu391+93)))));
    int val229 = (*(buf4+((long)((alu391+110)))));
    int val230 = (*(buf4+((long)((alu391+127)))));
    int val231 = (*(buf4+((long)((alu391+552)))));
    int val232 = (*(buf4+((long)((alu391+569)))));
    int val233 = (*(buf4+((long)((alu391+586)))));
    int val234 = (*(buf4+((long)((alu391+603)))));
    int val235 = (*(buf4+((long)((alu391+620)))));
    int val236 = (*(buf4+((long)((alu391+637)))));
    int val237 = (*(buf4+((long)((alu391+654)))));
    int val238 = (*(buf4+((long)((alu391+671)))));
    int alu392 = (alu373+(((alu172+alu231)>>2)-((int)((((alu172%4)!=0)&(alu173!=0))))));
    int val239 = (*(buf4+((long)((alu392+9)))));
    int val240 = (*(buf4+((long)((alu392+77)))));
    int val241 = (*(buf4+((long)((alu392+553)))));
    int val242 = (*(buf4+((long)((alu392+621)))));
    int alu393 = (alu373+(((alu174+alu232)>>2)-((int)((((alu174%4)!=0)&(alu175!=0))))));
    int val243 = (*(buf4+((long)((alu393+26)))));
    int val244 = (*(buf4+((long)((alu393+94)))));
    int val245 = (*(buf4+((long)((alu393+570)))));
    int val246 = (*(buf4+((long)((alu393+638)))));
    int alu394 = (alu373+(((alu176+alu233)>>2)-((int)((((alu176%4)!=0)&(alu177!=0))))));
    int val247 = (*(buf4+((long)((alu394+43)))));
    int val248 = (*(buf4+((long)((alu394+111)))));
    int val249 = (*(buf4+((long)((alu394+587)))));
    int val250 = (*(buf4+((long)((alu394+655)))));
    int alu395 = (alu373+(((alu178+alu234)>>2)-((int)((((alu178%4)!=0)&(alu179!=0))))));
    int val251 = (*(buf4+((long)((alu395+60)))));
    int val252 = (*(buf4+((long)((alu395+128)))));
    int val253 = (*(buf4+((long)((alu395+604)))));
    int val254 = (*(buf4+((long)((alu395+672)))));
    int alu396 = (alu373+(((alu181+alu236)>>1)-((int)((((alu181%2)!=0)&(alu182!=0))))));
    int val255 = (*(buf4+((long)((alu396+8)))));
    int val256 = (*(buf4+((long)((alu396+25)))));
    int val257 = (*(buf4+((long)((alu396+42)))));
    int val258 = (*(buf4+((long)((alu396+59)))));
    int val259 = (*(buf4+((long)((alu396+76)))));
    int val260 = (*(buf4+((long)((alu396+93)))));
    int val261 = (*(buf4+((long)((alu396+110)))));
    int val262 = (*(buf4+((long)((alu396+127)))));
    int val263 = (*(buf4+((long)((alu396+552)))));
    int val264 = (*(buf4+((long)((alu396+569)))));
    int val265 = (*(buf4+((long)((alu396+586)))));
    int val266 = (*(buf4+((long)((alu396+603)))));
    int val267 = (*(buf4+((long)((alu396+620)))));
    int val268 = (*(buf4+((long)((alu396+637)))));
    int val269 = (*(buf4+((long)((alu396+654)))));
    int val270 = (*(buf4+((long)((alu396+671)))));
    int alu397 = (alu373+(((alu183+alu237)>>2)-((int)((((alu183%4)!=0)&(alu184!=0))))));
    int val271 = (*(buf4+((long)((alu397+9)))));
    int val272 = (*(buf4+((long)((alu397+77)))));
    int val273 = (*(buf4+((long)((alu397+553)))));
    int val274 = (*(buf4+((long)((alu397+621)))));
    int alu398 = (alu373+(((alu185+alu238)>>2)-((int)((((alu185%4)!=0)&(alu186!=0))))));
    int val275 = (*(buf4+((long)((alu398+26)))));
    int val276 = (*(buf4+((long)((alu398+94)))));
    int val277 = (*(buf4+((long)((alu398+570)))));
    int val278 = (*(buf4+((long)((alu398+638)))));
    int alu399 = (alu373+(((alu187+alu239)>>2)-((int)((((alu187%4)!=0)&(alu188!=0))))));
    int val279 = (*(buf4+((long)((alu399+43)))));
    int val280 = (*(buf4+((long)((alu399+111)))));
    int val281 = (*(buf4+((long)((alu399+587)))));
    int val282 = (*(buf4+((long)((alu399+655)))));
    int alu400 = (alu373+(((alu189+alu240)>>2)-((int)((((alu189%4)!=0)&(alu190!=0))))));
    int val283 = (*(buf4+((long)((alu400+60)))));
    int val284 = (*(buf4+((long)((alu400+128)))));
    int val285 = (*(buf4+((long)((alu400+604)))));
    int val286 = (*(buf4+((long)((alu400+672)))));
    int alu401 = (alu373+(((alu158+alu248)>>1)-((int)((((alu158%2)!=0)&(alu202!=0))))));
    int val287 = (*(buf4+((long)((alu401+17)))));
    int val288 = (*(buf4+((long)((alu401+34)))));
    int val289 = (*(buf4+((long)((alu401+51)))));
    int val290 = (*(buf4+((long)((alu401+68)))));
    int val291 = (*(buf4+((long)((alu401+85)))));
    int val292 = (*(buf4+((long)((alu401+102)))));
    int val293 = (*(buf4+((long)((alu401+119)))));
    int val294 = (*(buf4+((long)((alu401+544)))));
    int val295 = (*(buf4+((long)((alu401+561)))));
    int val296 = (*(buf4+((long)((alu401+578)))));
    int val297 = (*(buf4+((long)((alu401+595)))));
    int val298 = (*(buf4+((long)((alu401+612)))));
    int val299 = (*(buf4+((long)((alu401+629)))));
    int val300 = (*(buf4+((long)((alu401+646)))));
    int val301 = (*(buf4+((long)((alu401+663)))));
    int alu402 = (alu373+(((alu169+alu249)>>1)-((int)((((alu169%2)!=0)&(alu203!=0))))));
    int val302 = (*(buf4+((long)((alu402+17)))));
    int val303 = (*(buf4+((long)((alu402+34)))));
    int val304 = (*(buf4+((long)((alu402+51)))));
    int val305 = (*(buf4+((long)((alu402+68)))));
    int val306 = (*(buf4+((long)((alu402+85)))));
    int val307 = (*(buf4+((long)((alu402+102)))));
    int val308 = (*(buf4+((long)((alu402+119)))));
    int val309 = (*(buf4+((long)((alu402+544)))));
    int val310 = (*(buf4+((long)((alu402+561)))));
    int val311 = (*(buf4+((long)((alu402+578)))));
    int val312 = (*(buf4+((long)((alu402+595)))));
    int val313 = (*(buf4+((long)((alu402+612)))));
    int val314 = (*(buf4+((long)((alu402+629)))));
    int val315 = (*(buf4+((long)((alu402+646)))));
    int val316 = (*(buf4+((long)((alu402+663)))));
    int alu403 = (alu373+(((alu180+alu250)>>1)-((int)((((alu180%2)!=0)&(alu204!=0))))));
    int val317 = (*(buf4+((long)((alu403+17)))));
    int val318 = (*(buf4+((long)((alu403+34)))));
    int val319 = (*(buf4+((long)((alu403+51)))));
    int val320 = (*(buf4+((long)((alu403+68)))));
    int val321 = (*(buf4+((long)((alu403+85)))));
    int val322 = (*(buf4+((long)((alu403+102)))));
    int val323 = (*(buf4+((long)((alu403+119)))));
    int val324 = (*(buf4+((long)((alu403+544)))));
    int val325 = (*(buf4+((long)((alu403+561)))));
    int val326 = (*(buf4+((long)((alu403+578)))));
    int val327 = (*(buf4+((long)((alu403+595)))));
    int val328 = (*(buf4+((long)((alu403+612)))));
    int val329 = (*(buf4+((long)((alu403+629)))));
    int val330 = (*(buf4+((long)((alu403+646)))));
    int val331 = (*(buf4+((long)((alu403+663)))));
    int val332 = (*(buf4+((long)(alu374))));
    int val333 = (*(buf4+((long)(alu378))));
    int val334 = (*(buf4+((long)(alu382))));
    int val335 = (*(buf4+((long)(alu401))));
    int val336 = (*(buf4+((long)(alu402))));
    int val337 = (*(buf4+((long)(alu403))));
    int alu404 = (alu373+(((alu150+alu219)>>2)-((int)((((alu150%4)!=0)&(alu151!=0))))));
    int val338 = (*(buf4+(alu404+17)));
    int val339 = (*(buf4+(alu404+85)));
    int val340 = (*(buf4+(alu404+561)));
    int val341 = (*(buf4+(alu404+629)));
    int alu405 = (alu373+(((alu152+alu220)>>2)-((int)((((alu152%4)!=0)&(alu153!=0))))));
    int val342 = (*(buf4+(alu405+34)));
    int val343 = (*(buf4+(alu405+102)));
    int val344 = (*(buf4+(alu405+578)));
    int val345 = (*(buf4+(alu405+646)));
    int alu406 = (alu373+(((alu154+alu221)>>2)-((int)((((alu154%4)!=0)&(alu155!=0))))));
    int val346 = (*(buf4+(alu406+51)));
    int val347 = (*(buf4+(alu406+119)));
    int val348 = (*(buf4+(alu406+595)));
    int val349 = (*(buf4+(alu406+663)));
    int alu407 = (alu373+(((alu156+alu222)>>2)-((int)((((alu156%4)!=0)&(alu157!=0))))));
    int val350 = (*(buf4+(alu407+68)));
    int val351 = (*(buf4+(alu407+544)));
    int val352 = (*(buf4+(alu407+612)));
    int alu408 = (alu373+(((alu192+alu242)>>1)-((int)((((alu192%2)!=0)&(alu193!=0))))));
    int val353 = (*(buf4+(alu408+8)));
    int val354 = (*(buf4+(alu408+25)));
    int val355 = (*(buf4+(alu408+42)));
    int val356 = (*(buf4+(alu408+59)));
    int val357 = (*(buf4+(alu408+76)));
    int val358 = (*(buf4+(alu408+93)));
    int val359 = (*(buf4+(alu408+110)));
    int val360 = (*(buf4+(alu408+127)));
    int val361 = (*(buf4+(alu408+552)));
    int val362 = (*(buf4+(alu408+569)));
    int val363 = (*(buf4+(alu408+586)));
    int val364 = (*(buf4+(alu408+603)));
    int val365 = (*(buf4+(alu408+620)));
    int val366 = (*(buf4+(alu408+637)));
    int val367 = (*(buf4+(alu408+654)));
    int val368 = (*(buf4+(alu408+671)));
    int alu409 = (alu373+(((alu194+alu243)>>2)-((int)((((alu194%4)!=0)&(alu195!=0))))));
    int val369 = (*(buf4+(alu409+9)));
    int val370 = (*(buf4+(alu409+77)));
    int val371 = (*(buf4+(alu409+553)));
    int val372 = (*(buf4+(alu409+621)));
    int alu410 = (alu373+(((alu196+alu244)>>2)-((int)((((alu196%4)!=0)&(alu197!=0))))));
    int val373 = (*(buf4+(alu410+26)));
    int val374 = (*(buf4+(alu410+94)));
    int val375 = (*(buf4+(alu410+570)));
    int val376 = (*(buf4+(alu410+638)));
    int alu411 = (alu373+(((alu198+alu245)>>2)-((int)((((alu198%4)!=0)&(alu199!=0))))));
    int val377 = (*(buf4+(alu411+43)));
    int val378 = (*(buf4+(alu411+111)));
    int val379 = (*(buf4+(alu411+587)));
    int val380 = (*(buf4+(alu411+655)));
    int alu412 = (alu373+(((alu200+alu246)>>2)-((int)((((alu200%4)!=0)&(alu201!=0))))));
    int val381 = (*(buf4+(alu412+60)));
    int val382 = (*(buf4+(alu412+128)));
    int val383 = (*(buf4+(alu412+604)));
    int val384 = (*(buf4+(alu412+672)));
    int alu413 = (alu373+(((alu191+alu251)>>1)-((int)((((alu191%2)!=0)&(alu205!=0))))));
    int val385 = (*(buf4+(alu413+17)));
    int val386 = (*(buf4+(alu413+34)));
    int val387 = (*(buf4+(alu413+51)));
    int val388 = (*(buf4+(alu413+68)));
    int val389 = (*(buf4+(alu413+85)));
    int val390 = (*(buf4+(alu413+102)));
    int val391 = (*(buf4+(alu413+119)));
    int val392 = (*(buf4+(alu413+544)));
    int val393 = (*(buf4+(alu413+561)));
    int val394 = (*(buf4+(alu413+578)));
    int val395 = (*(buf4+(alu413+595)));
    int val396 = (*(buf4+(alu413+612)));
    int val397 = (*(buf4+(alu413+629)));
    int val398 = (*(buf4+(alu413+646)));
    int val399 = (*(buf4+(alu413+663)));
    int val400 = (*(buf4+alu407));
    int val401 = (*(buf4+alu413));
    half val402 = (*(buf5+(alu223+8)));
    half val403 = (*(buf5+(alu223+64)));
    half val404 = (*(buf5+(alu223+72)));
    half val405 = (*(buf5+alu223));
    half4 cast20 = make_half4(val405,val402,val403,val404);
    half alu414 = (alu360?((half)(0.0f)):((half)(alu357)));
    half alu415 = (alu361?((half)(0.0f)):((half)(alu359)));
    half4 wmma16 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val164>>alu266)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val302>>alu296)))))),(((half)(((signed char)((val165>>alu266)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val306>>alu296)))))),(((half)(((signed char)((val179>>alu274)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val317>>alu303)))))),(((half)(((signed char)((val180>>alu274)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val321>>alu303)))))),(((half)(((signed char)((val166>>alu266)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val310>>alu296)))))),(((half)(((signed char)((val167>>alu266)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val314>>alu296)))))),(((half)(((signed char)((val181>>alu274)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val325>>alu303)))))),(((half)(((signed char)((val182>>alu274)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val329>>alu303))))))), cast20, make_half4(((*(buf0+20))*alu414),((*(buf0+21))*alu415),((*(buf0+22))*alu414),((*(buf0+23))*alu415)));
    half4 wmma17 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val168>>alu268)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val303>>alu298)))))),(((half)(((signed char)((val169>>alu268)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val307>>alu298)))))),(((half)(((signed char)((val183>>alu276)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val318>>alu305)))))),(((half)(((signed char)((val184>>alu276)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val322>>alu305)))))),(((half)(((signed char)((val170>>alu268)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val311>>alu298)))))),(((half)(((signed char)((val171>>alu268)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val315>>alu298)))))),(((half)(((signed char)((val185>>alu276)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val326>>alu305)))))),(((half)(((signed char)((val186>>alu276)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val330>>alu305))))))), cast20, make_half4(((*(buf0+36))*alu414),((*(buf0+37))*alu415),((*(buf0+38))*alu414),((*(buf0+39))*alu415)));
    half4 wmma18 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val172>>alu270)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val304>>alu300)))))),(((half)(((signed char)((val173>>alu270)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val308>>alu300)))))),(((half)(((signed char)((val187>>alu278)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val319>>alu307)))))),(((half)(((signed char)((val188>>alu278)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val323>>alu307)))))),(((half)(((signed char)((val174>>alu270)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val312>>alu300)))))),(((half)(((signed char)((val175>>alu270)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val316>>alu300)))))),(((half)(((signed char)((val189>>alu278)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val327>>alu307)))))),(((half)(((signed char)((val190>>alu278)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val331>>alu307))))))), cast20, make_half4(((*(buf0+52))*alu414),((*(buf0+53))*alu415),((*(buf0+54))*alu414),((*(buf0+55))*alu415)));
    half4 wmma19 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val239>>alu265)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val223>>alu295)))))),(((half)(((signed char)((val240>>alu265)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val227>>alu295)))))),(((half)(((signed char)((val271>>alu273)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val255>>alu302)))))),(((half)(((signed char)((val272>>alu273)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val259>>alu302)))))),(((half)(((signed char)((val241>>alu265)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val231>>alu295)))))),(((half)(((signed char)((val242>>alu265)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val235>>alu295)))))),(((half)(((signed char)((val273>>alu273)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val263>>alu302)))))),(((half)(((signed char)((val274>>alu273)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val267>>alu302))))))), cast20, make_half4(((*(buf0+12))*alu414),((*(buf0+13))*alu415),((*(buf0+14))*alu414),((*(buf0+15))*alu415)));
    half4 wmma20 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val243>>alu267)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val224>>alu297)))))),(((half)(((signed char)((val244>>alu267)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val228>>alu297)))))),(((half)(((signed char)((val275>>alu275)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val256>>alu304)))))),(((half)(((signed char)((val276>>alu275)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val260>>alu304)))))),(((half)(((signed char)((val245>>alu267)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val232>>alu297)))))),(((half)(((signed char)((val246>>alu267)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val236>>alu297)))))),(((half)(((signed char)((val277>>alu275)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val264>>alu304)))))),(((half)(((signed char)((val278>>alu275)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val268>>alu304))))))), cast20, make_half4(((*(buf0+28))*alu414),((*(buf0+29))*alu415),((*(buf0+30))*alu414),((*(buf0+31))*alu415)));
    half4 wmma21 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val247>>alu269)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val225>>alu299)))))),(((half)(((signed char)((val248>>alu269)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val229>>alu299)))))),(((half)(((signed char)((val279>>alu277)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val257>>alu306)))))),(((half)(((signed char)((val280>>alu277)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val261>>alu306)))))),(((half)(((signed char)((val249>>alu269)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val233>>alu299)))))),(((half)(((signed char)((val250>>alu269)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val237>>alu299)))))),(((half)(((signed char)((val281>>alu277)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val265>>alu306)))))),(((half)(((signed char)((val282>>alu277)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val269>>alu306))))))), cast20, make_half4(((*(buf0+44))*alu414),((*(buf0+45))*alu415),((*(buf0+46))*alu414),((*(buf0+47))*alu415)));
    half4 wmma22 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val251>>alu271)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val226>>alu301)))))),(((half)(((signed char)((val252>>alu271)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val230>>alu301)))))),(((half)(((signed char)((val283>>alu279)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val258>>alu308)))))),(((half)(((signed char)((val284>>alu279)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val262>>alu308)))))),(((half)(((signed char)((val253>>alu271)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val234>>alu301)))))),(((half)(((signed char)((val254>>alu271)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val238>>alu301)))))),(((half)(((signed char)((val285>>alu279)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val266>>alu308)))))),(((half)(((signed char)((val286>>alu279)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val270>>alu308))))))), cast20, make_half4(((*(buf0+60))*alu414),((*(buf0+61))*alu415),((*(buf0+62))*alu414),((*(buf0+63))*alu415)));
    half4 wmma23 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val333>>alu264)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val336>>alu317)))))),(((half)(((signed char)((val161>>alu264)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val305>>alu317)))))),(((half)(((signed char)((val334>>alu272)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val337>>alu318)))))),(((half)(((signed char)((val176>>alu272)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val320>>alu318)))))),(((half)(((signed char)((val162>>alu264)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val309>>alu317)))))),(((half)(((signed char)((val163>>alu264)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val313>>alu317)))))),(((half)(((signed char)((val177>>alu272)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val324>>alu318)))))),(((half)(((signed char)((val178>>alu272)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val328>>alu318))))))), cast20, make_half4(((*(buf0+4))*alu414),((*(buf0+5))*alu415),((*(buf0+6))*alu414),((*(buf0+7))*alu415)));
    half4 wmma24 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val338>>alu282)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val385>>alu310)))))),(((half)(((signed char)((val339>>alu282)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val389>>alu310)))))),(((half)(((signed char)((val149>>alu258)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val287>>alu289)))))),(((half)(((signed char)((val150>>alu258)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val291>>alu289)))))),(((half)(((signed char)((val340>>alu282)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val393>>alu310)))))),(((half)(((signed char)((val341>>alu282)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val397>>alu310)))))),(((half)(((signed char)((val151>>alu258)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val295>>alu289)))))),(((half)(((signed char)((val152>>alu258)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val299>>alu289))))))), cast20, make_half4(((*(buf0+16))*alu414),((*(buf0+17))*alu415),((*(buf0+18))*alu414),((*(buf0+19))*alu415)));
    half4 wmma25 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val342>>alu284)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val386>>alu312)))))),(((half)(((signed char)((val343>>alu284)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val390>>alu312)))))),(((half)(((signed char)((val153>>alu260)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val288>>alu291)))))),(((half)(((signed char)((val154>>alu260)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val292>>alu291)))))),(((half)(((signed char)((val344>>alu284)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val394>>alu312)))))),(((half)(((signed char)((val345>>alu284)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val398>>alu312)))))),(((half)(((signed char)((val155>>alu260)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val296>>alu291)))))),(((half)(((signed char)((val156>>alu260)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val300>>alu291))))))), cast20, make_half4(((*(buf0+32))*alu414),((*(buf0+33))*alu415),((*(buf0+34))*alu414),((*(buf0+35))*alu415)));
    half4 wmma26 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val346>>alu286)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val387>>alu314)))))),(((half)(((signed char)((val347>>alu286)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val391>>alu314)))))),(((half)(((signed char)((val157>>alu262)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val289>>alu293)))))),(((half)(((signed char)((val158>>alu262)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val293>>alu293)))))),(((half)(((signed char)((val348>>alu286)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val395>>alu314)))))),(((half)(((signed char)((val349>>alu286)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val399>>alu314)))))),(((half)(((signed char)((val159>>alu262)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val297>>alu293)))))),(((half)(((signed char)((val160>>alu262)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val301>>alu293))))))), cast20, make_half4(((*(buf0+48))*alu414),((*(buf0+49))*alu415),((*(buf0+50))*alu414),((*(buf0+51))*alu415)));
    half4 wmma27 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val369>>alu281)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val353>>alu309)))))),(((half)(((signed char)((val370>>alu281)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val357>>alu309)))))),(((half)(((signed char)((val207>>alu257)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val191>>alu288)))))),(((half)(((signed char)((val208>>alu257)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val195>>alu288)))))),(((half)(((signed char)((val371>>alu281)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val361>>alu309)))))),(((half)(((signed char)((val372>>alu281)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val365>>alu309)))))),(((half)(((signed char)((val209>>alu257)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val199>>alu288)))))),(((half)(((signed char)((val210>>alu257)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val203>>alu288))))))), cast20, make_half4(((*(buf0+8))*alu414),((*(buf0+9))*alu415),((*(buf0+10))*alu414),((*(buf0+11))*alu415)));
    half4 wmma28 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val373>>alu283)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val354>>alu311)))))),(((half)(((signed char)((val374>>alu283)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val358>>alu311)))))),(((half)(((signed char)((val211>>alu259)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val192>>alu290)))))),(((half)(((signed char)((val212>>alu259)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val196>>alu290)))))),(((half)(((signed char)((val375>>alu283)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val362>>alu311)))))),(((half)(((signed char)((val376>>alu283)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val366>>alu311)))))),(((half)(((signed char)((val213>>alu259)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val200>>alu290)))))),(((half)(((signed char)((val214>>alu259)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val204>>alu290))))))), cast20, make_half4(((*(buf0+24))*alu414),((*(buf0+25))*alu415),((*(buf0+26))*alu414),((*(buf0+27))*alu415)));
    half4 wmma29 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val377>>alu285)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val355>>alu313)))))),(((half)(((signed char)((val378>>alu285)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val359>>alu313)))))),(((half)(((signed char)((val215>>alu261)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val193>>alu292)))))),(((half)(((signed char)((val216>>alu261)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val197>>alu292)))))),(((half)(((signed char)((val379>>alu285)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val363>>alu313)))))),(((half)(((signed char)((val380>>alu285)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val367>>alu313)))))),(((half)(((signed char)((val217>>alu261)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val201>>alu292)))))),(((half)(((signed char)((val218>>alu261)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val205>>alu292))))))), cast20, make_half4(((*(buf0+40))*alu414),((*(buf0+41))*alu415),((*(buf0+42))*alu414),((*(buf0+43))*alu415)));
    half4 wmma30 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val381>>alu287)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val356>>alu315)))))),(((half)(((signed char)((val382>>alu287)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val360>>alu315)))))),(((half)(((signed char)((val219>>alu263)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val194>>alu294)))))),(((half)(((signed char)((val220>>alu263)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val198>>alu294)))))),(((half)(((signed char)((val383>>alu287)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val364>>alu315)))))),(((half)(((signed char)((val384>>alu287)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val368>>alu315)))))),(((half)(((signed char)((val221>>alu263)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val202>>alu294)))))),(((half)(((signed char)((val222>>alu263)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val206>>alu294))))))), cast20, make_half4(((*(buf0+56))*alu414),((*(buf0+57))*alu415),((*(buf0+58))*alu414),((*(buf0+59))*alu415)));
    half4 wmma31 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val400>>alu280)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val401>>alu319)))))),(((half)(((signed char)((val350>>alu280)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val388>>alu319)))))),(((half)(((signed char)((val332>>alu256)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val335>>alu316)))))),(((half)(((signed char)((val146>>alu256)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val290>>alu316)))))),(((half)(((signed char)((val351>>alu280)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val392>>alu319)))))),(((half)(((signed char)((val352>>alu280)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val396>>alu319)))))),(((half)(((signed char)((val147>>alu256)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val294>>alu316)))))),(((half)(((signed char)((val148>>alu256)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val298>>alu316))))))), cast20, make_half4(((*(buf0+0))*alu414),((*(buf0+1))*alu415),((*(buf0+2))*alu414),((*(buf0+3))*alu415)));
    (*(buf0+0)) = wmma31.x;
    (*(buf0+1)) = wmma31.y;
    (*(buf0+2)) = wmma31.z;
    (*(buf0+3)) = wmma31.w;
    (*(buf0+4)) = wmma23.x;
    (*(buf0+5)) = wmma23.y;
    (*(buf0+6)) = wmma23.z;
    (*(buf0+7)) = wmma23.w;
    (*(buf0+8)) = wmma27.x;
    (*(buf0+9)) = wmma27.y;
    (*(buf0+10)) = wmma27.z;
    (*(buf0+11)) = wmma27.w;
    (*(buf0+12)) = wmma19.x;
    (*(buf0+13)) = wmma19.y;
    (*(buf0+14)) = wmma19.z;
    (*(buf0+15)) = wmma19.w;
    (*(buf0+16)) = wmma24.x;
    (*(buf0+17)) = wmma24.y;
    (*(buf0+18)) = wmma24.z;
    (*(buf0+19)) = wmma24.w;
    (*(buf0+20)) = wmma16.x;
    (*(buf0+21)) = wmma16.y;
    (*(buf0+22)) = wmma16.z;
    (*(buf0+23)) = wmma16.w;
    (*(buf0+24)) = wmma28.x;
    (*(buf0+25)) = wmma28.y;
    (*(buf0+26)) = wmma28.z;
    (*(buf0+27)) = wmma28.w;
    (*(buf0+28)) = wmma20.x;
    (*(buf0+29)) = wmma20.y;
    (*(buf0+30)) = wmma20.z;
    (*(buf0+31)) = wmma20.w;
    (*(buf0+32)) = wmma25.x;
    (*(buf0+33)) = wmma25.y;
    (*(buf0+34)) = wmma25.z;
    (*(buf0+35)) = wmma25.w;
    (*(buf0+36)) = wmma17.x;
    (*(buf0+37)) = wmma17.y;
    (*(buf0+38)) = wmma17.z;
    (*(buf0+39)) = wmma17.w;
    (*(buf0+40)) = wmma29.x;
    (*(buf0+41)) = wmma29.y;
    (*(buf0+42)) = wmma29.z;
    (*(buf0+43)) = wmma29.w;
    (*(buf0+44)) = wmma21.x;
    (*(buf0+45)) = wmma21.y;
    (*(buf0+46)) = wmma21.z;
    (*(buf0+47)) = wmma21.w;
    (*(buf0+48)) = wmma26.x;
    (*(buf0+49)) = wmma26.y;
    (*(buf0+50)) = wmma26.z;
    (*(buf0+51)) = wmma26.w;
    (*(buf0+52)) = wmma18.x;
    (*(buf0+53)) = wmma18.y;
    (*(buf0+54)) = wmma18.z;
    (*(buf0+55)) = wmma18.w;
    (*(buf0+56)) = wmma30.x;
    (*(buf0+57)) = wmma30.y;
    (*(buf0+58)) = wmma30.z;
    (*(buf0+59)) = wmma30.w;
    (*(buf0+60)) = wmma22.x;
    (*(buf0+61)) = wmma22.y;
    (*(buf0+62)) = wmma22.z;
    (*(buf0+63)) = wmma22.w;
    __syncthreads();
  }
  float buf12;
  float buf13;
  float buf14;
  float buf15;
  float buf16;
  float buf17;
  buf12 = __shfl_xor_sync(0xffffffffu, (*(buf2+0)), 16);
  float alu483 = ((*(buf2+0))+buf12);
  buf13 = __shfl_xor_sync(0xffffffffu, alu483, 8);
  float alu485 = (alu483+buf13);
  buf14 = __shfl_xor_sync(0xffffffffu, alu485, 4);
  buf15 = __shfl_xor_sync(0xffffffffu, (*(buf2+1)), 16);
  float alu488 = ((*(buf2+1))+buf15);
  buf16 = __shfl_xor_sync(0xffffffffu, alu488, 8);
  float alu490 = (alu488+buf16);
  buf17 = __shfl_xor_sync(0xffffffffu, alu490, 4);
  int alu492 = ((alu10*122880)+(alu6*245760));
  int alu493 = (alu38+alu36);
  bool alu494 = (((lidx1<<2)+alu35)<6);
  int alu495 = (alu494?alu493:0);
  int alu496 = (gidx2<<10);
  int alu497 = (gidx1<<8);
  int alu498 = (alu492+(alu495*10240)+alu496+alu497+alu39);
  bool alu499 = (alu493<11);
  int alu500 = (alu499?(alu493+1):0);
  int alu501 = (alu492+(alu500*10240)+alu496+alu497+alu39);
  int alu502 = ((alu10*960)+(alu6*1920));
  int alu503 = (gidx2<<3);
  int alu504 = (gidx1<<1);
  int alu505 = (alu502+(alu495*80)+alu503+alu504);
  int alu506 = (alu502+(alu500*80)+alu503+alu504);
  if (alu494) {
    *(data0_983040+(alu498+8)) = (*(buf0+2));
  }
  if (alu494) {
    *(data0_983040+(alu498+16)) = (*(buf0+4));
  }
  if (alu494) {
    *(data0_983040+(alu498+24)) = (*(buf0+6));
  }
  if (alu494) {
    *(data0_983040+(alu498+32)) = (*(buf0+8));
  }
  if (alu494) {
    *(data0_983040+(alu498+40)) = (*(buf0+10));
  }
  if (alu494) {
    *(data0_983040+(alu498+48)) = (*(buf0+12));
  }
  if (alu494) {
    *(data0_983040+(alu498+56)) = (*(buf0+14));
  }
  if (alu494) {
    *(data0_983040+(alu498+64)) = (*(buf0+16));
  }
  if (alu494) {
    *(data0_983040+(alu498+72)) = (*(buf0+18));
  }
  if (alu494) {
    *(data0_983040+(alu498+80)) = (*(buf0+20));
  }
  if (alu494) {
    *(data0_983040+(alu498+88)) = (*(buf0+22));
  }
  if (alu494) {
    *(data0_983040+(alu498+96)) = (*(buf0+24));
  }
  if (alu494) {
    *(data0_983040+(alu498+104)) = (*(buf0+26));
  }
  if (alu494) {
    *(data0_983040+(alu498+112)) = (*(buf0+28));
  }
  if (alu494) {
    *(data0_983040+(alu498+120)) = (*(buf0+30));
  }
  if (alu494) {
    *(data0_983040+(alu498+128)) = (*(buf0+32));
  }
  if (alu494) {
    *(data0_983040+(alu498+136)) = (*(buf0+34));
  }
  if (alu494) {
    *(data0_983040+(alu498+144)) = (*(buf0+36));
  }
  if (alu494) {
    *(data0_983040+(alu498+152)) = (*(buf0+38));
  }
  if (alu494) {
    *(data0_983040+(alu498+160)) = (*(buf0+40));
  }
  if (alu494) {
    *(data0_983040+(alu498+168)) = (*(buf0+42));
  }
  if (alu494) {
    *(data0_983040+(alu498+176)) = (*(buf0+44));
  }
  if (alu494) {
    *(data0_983040+(alu498+184)) = (*(buf0+46));
  }
  if (alu494) {
    *(data0_983040+(alu498+192)) = (*(buf0+48));
  }
  if (alu494) {
    *(data0_983040+(alu498+200)) = (*(buf0+50));
  }
  if (alu494) {
    *(data0_983040+(alu498+208)) = (*(buf0+52));
  }
  if (alu494) {
    *(data0_983040+(alu498+216)) = (*(buf0+54));
  }
  if (alu494) {
    *(data0_983040+(alu498+224)) = (*(buf0+56));
  }
  if (alu494) {
    *(data0_983040+(alu498+232)) = (*(buf0+58));
  }
  if (alu494) {
    *(data0_983040+(alu498+240)) = (*(buf0+60));
  }
  if (alu494) {
    *(data0_983040+(alu498+248)) = (*(buf0+62));
  }
  if (alu499) {
    *(data0_983040+(alu501+8)) = (*(buf0+3));
  }
  if (alu499) {
    *(data0_983040+(alu501+16)) = (*(buf0+5));
  }
  if (alu499) {
    *(data0_983040+(alu501+24)) = (*(buf0+7));
  }
  if (alu499) {
    *(data0_983040+(alu501+32)) = (*(buf0+9));
  }
  if (alu499) {
    *(data0_983040+(alu501+40)) = (*(buf0+11));
  }
  if (alu499) {
    *(data0_983040+(alu501+48)) = (*(buf0+13));
  }
  if (alu499) {
    *(data0_983040+(alu501+56)) = (*(buf0+15));
  }
  if (alu499) {
    *(data0_983040+(alu501+64)) = (*(buf0+17));
  }
  if (alu499) {
    *(data0_983040+(alu501+72)) = (*(buf0+19));
  }
  if (alu499) {
    *(data0_983040+(alu501+80)) = (*(buf0+21));
  }
  if (alu499) {
    *(data0_983040+(alu501+88)) = (*(buf0+23));
  }
  if (alu499) {
    *(data0_983040+(alu501+96)) = (*(buf0+25));
  }
  if (alu499) {
    *(data0_983040+(alu501+104)) = (*(buf0+27));
  }
  if (alu499) {
    *(data0_983040+(alu501+112)) = (*(buf0+29));
  }
  if (alu499) {
    *(data0_983040+(alu501+120)) = (*(buf0+31));
  }
  if (alu499) {
    *(data0_983040+(alu501+128)) = (*(buf0+33));
  }
  if (alu499) {
    *(data0_983040+(alu501+136)) = (*(buf0+35));
  }
  if (alu499) {
    *(data0_983040+(alu501+144)) = (*(buf0+37));
  }
  if (alu499) {
    *(data0_983040+(alu501+152)) = (*(buf0+39));
  }
  if (alu499) {
    *(data0_983040+(alu501+160)) = (*(buf0+41));
  }
  if (alu499) {
    *(data0_983040+(alu501+168)) = (*(buf0+43));
  }
  if (alu499) {
    *(data0_983040+(alu501+176)) = (*(buf0+45));
  }
  if (alu499) {
    *(data0_983040+(alu501+184)) = (*(buf0+47));
  }
  if (alu499) {
    *(data0_983040+(alu501+192)) = (*(buf0+49));
  }
  if (alu499) {
    *(data0_983040+(alu501+200)) = (*(buf0+51));
  }
  if (alu499) {
    *(data0_983040+(alu501+208)) = (*(buf0+53));
  }
  if (alu499) {
    *(data0_983040+(alu501+216)) = (*(buf0+55));
  }
  if (alu499) {
    *(data0_983040+(alu501+224)) = (*(buf0+57));
  }
  if (alu499) {
    *(data0_983040+(alu501+232)) = (*(buf0+59));
  }
  if (alu499) {
    *(data0_983040+(alu501+240)) = (*(buf0+61));
  }
  if (alu499) {
    *(data0_983040+(alu501+248)) = (*(buf0+63));
  }
  if (alu494) {
    *(data0_983040+alu498) = (*(buf0+0));
  }
  if (alu499) {
    *(data0_983040+alu501) = (*(buf0+1));
  }
  if (alu494) {
    *(data1_7680+(alu505+1)) = (alu485+buf14);
  }
  if (alu499) {
    *(data1_7680+(alu506+1)) = (alu490+buf17);
  }
  if (alu494) {
    *(data1_7680+alu505) = (*(buf1+0));
  }
  if (alu499) {
    *(data1_7680+alu506) = (*(buf1+1));
  }
}