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
extern "C" __global__ void __launch_bounds__(64) sparse_q8_exact2_t3_h24_2_d256_l2051_r14(half* data0_1032192, float* data1_8064, float* data2_18432, int* data3_35651584, int* data4_35651584, int* data5_6153, int* data6_3) {
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
  int gidx0 = blockIdx.x; /* 6 */
  int alu6 = (gidx0%3);
  int val0 = (*(data6_3+alu6));
  int gidx1 = blockIdx.y; /* 4 */
  int gidx2 = blockIdx.z; /* 14 */
  int lidx0 = threadIdx.x; /* 32 */
  int lidx1 = threadIdx.y; /* 2 */
  __shared__ __align__(16) int buf3[3264];
  __shared__ __align__(16) int buf4[3264];
  long alu7 = (((long)(val0))+63ll);
  bool alu8 = (alu7<0ll);
  long alu9 = (alu8?63ll:0ll);
  int alu10 = (gidx0/3);
  int alu11 = (gidx2*5);
  int alu12 = (gidx2<<1);
  int alu13 = (alu11/14);
  int alu14 = (alu12+alu13);
  int alu15 = (alu14+1);
  int alu16 = ((gidx2<<7)+(alu13<<6));
  int alu17 = (gidx1<<4);
  int alu18 = (((int)((((alu7+alu9)>>6ll)-((long)((((alu7%64ll)!=0ll)&(alu8!=0)))))))^-1);
  int alu19 = ((alu12+((alu11+5)/14)+2)^-1);
  int alu20 = ((alu19<alu18)?alu18:alu19);
  int alu21 = (alu20^-1);
  int alu22 = ((alu15<alu21)?alu21:alu15);
  for (int Ridx60 = 0; Ridx60 < 13; Ridx60++) {
    int alu23 = (lidx0+(lidx1*15)+(Ridx60*13));
    int alu24 = (lidx1+(Ridx60*3)+(alu23/17));
    int alu25 = (alu24>>4);
    int alu26 = (alu16+(alu25<<6)+alu17+(alu24&15));
    bool alu27 = (((alu14+alu25)<alu22)&(alu26<val0));
    int alu28 = (alu27?alu26:0);
    bool alu29 = ((lidx0+(lidx1<<5)+(Ridx60<<6))<816);
    bool alu30 = (alu27&alu29);
    int val1 = (alu30?*(data5_6153+((alu6*2051)+alu28)):0);
    long alu31 = ((((long)(val1))*136ll)+((long)((alu10*68)))+((long)(((alu23%17)<<2))));
    int alu32 = ((lidx0<<2)+(lidx1*60)+(Ridx60*52)+(lidx1*68)+(Ridx60*204));
    if (alu29) *((int4*)((buf3+alu32))) = (alu30) ? __ldg((const int4*)((data3_35651584+alu31))) : make_int4(0, 0, 0, 0);
    if (alu29) *((int4*)((buf4+alu32))) = (alu30) ? __ldg((const int4*)((data4_35651584+alu31))) : make_int4(0, 0, 0, 0);
  }
  bool alu36 = ((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))<0);
  bool alu37 = (alu36!=0);
  int alu38 = ((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))&3);
  int alu39 = (alu38<<1);
  int alu40 = (alu36?3:0);
  int alu41 = (lidx1<<3);
  int alu42 = ((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))+alu40)>>2)-((int)(((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))%4)!=0)&alu37))));
  int alu43 = (alu41+alu42);
  bool alu44 = (alu43<12);
  int alu45 = (alu44?alu43:0);
  long alu46 = (((long)(((alu10*3072)+(alu6*6144))))+(((long)(alu45))<<8ll)+((long)(alu39)));
  float val2 = (*(data2_18432+alu46));
  float val3 = (*(data2_18432+(alu46+1ll)));
  float val4 = (*(data2_18432+(alu46+8ll)));
  float val5 = (*(data2_18432+(alu46+9ll)));
  float val6 = (*(data2_18432+(alu46+16ll)));
  float val7 = (*(data2_18432+(alu46+17ll)));
  float val8 = (*(data2_18432+(alu46+24ll)));
  float val9 = (*(data2_18432+(alu46+25ll)));
  float val10 = (*(data2_18432+(alu46+32ll)));
  float val11 = (*(data2_18432+(alu46+33ll)));
  float val12 = (*(data2_18432+(alu46+40ll)));
  float val13 = (*(data2_18432+(alu46+41ll)));
  float val14 = (*(data2_18432+(alu46+48ll)));
  float val15 = (*(data2_18432+(alu46+49ll)));
  float val16 = (*(data2_18432+(alu46+56ll)));
  float val17 = (*(data2_18432+(alu46+57ll)));
  float val18 = (*(data2_18432+(alu46+64ll)));
  float val19 = (*(data2_18432+(alu46+65ll)));
  float val20 = (*(data2_18432+(alu46+72ll)));
  float val21 = (*(data2_18432+(alu46+73ll)));
  float val22 = (*(data2_18432+(alu46+80ll)));
  float val23 = (*(data2_18432+(alu46+81ll)));
  float val24 = (*(data2_18432+(alu46+88ll)));
  float val25 = (*(data2_18432+(alu46+89ll)));
  float val26 = (*(data2_18432+(alu46+96ll)));
  float val27 = (*(data2_18432+(alu46+97ll)));
  float val28 = (*(data2_18432+(alu46+104ll)));
  float val29 = (*(data2_18432+(alu46+105ll)));
  float val30 = (*(data2_18432+(alu46+112ll)));
  float val31 = (*(data2_18432+(alu46+113ll)));
  float val32 = (*(data2_18432+(alu46+120ll)));
  float val33 = (*(data2_18432+(alu46+121ll)));
  float val34 = (*(data2_18432+(alu46+128ll)));
  float val35 = (*(data2_18432+(alu46+129ll)));
  float val36 = (*(data2_18432+(alu46+136ll)));
  float val37 = (*(data2_18432+(alu46+137ll)));
  float val38 = (*(data2_18432+(alu46+144ll)));
  float val39 = (*(data2_18432+(alu46+145ll)));
  float val40 = (*(data2_18432+(alu46+152ll)));
  float val41 = (*(data2_18432+(alu46+153ll)));
  float val42 = (*(data2_18432+(alu46+160ll)));
  float val43 = (*(data2_18432+(alu46+161ll)));
  float val44 = (*(data2_18432+(alu46+168ll)));
  float val45 = (*(data2_18432+(alu46+169ll)));
  float val46 = (*(data2_18432+(alu46+176ll)));
  float val47 = (*(data2_18432+(alu46+177ll)));
  float val48 = (*(data2_18432+(alu46+184ll)));
  float val49 = (*(data2_18432+(alu46+185ll)));
  float val50 = (*(data2_18432+(alu46+192ll)));
  float val51 = (*(data2_18432+(alu46+193ll)));
  float val52 = (*(data2_18432+(alu46+200ll)));
  float val53 = (*(data2_18432+(alu46+201ll)));
  float val54 = (*(data2_18432+(alu46+208ll)));
  float val55 = (*(data2_18432+(alu46+209ll)));
  float val56 = (*(data2_18432+(alu46+216ll)));
  float val57 = (*(data2_18432+(alu46+217ll)));
  float val58 = (*(data2_18432+(alu46+224ll)));
  float val59 = (*(data2_18432+(alu46+225ll)));
  float val60 = (*(data2_18432+(alu46+232ll)));
  float val61 = (*(data2_18432+(alu46+233ll)));
  float val62 = (*(data2_18432+(alu46+240ll)));
  float val63 = (*(data2_18432+(alu46+241ll)));
  float val64 = (*(data2_18432+(alu46+248ll)));
  float val65 = (*(data2_18432+(alu46+249ll)));
  __shared__ __align__(16) half buf5[256];
  float buf6;
  float buf7;
  float buf8;
  float buf9;
  float buf10;
  float buf11;
  __syncthreads();
  half alu48 = (alu44?(((half)(val2))*((half)(0.0625f))):((half)(0.0f)));
  half alu49 = (alu44?(((half)(val3))*((half)(0.0625f))):((half)(0.0f)));
  half alu50 = (alu44?(((half)(val4))*((half)(0.0625f))):((half)(0.0f)));
  half alu51 = (alu44?(((half)(val5))*((half)(0.0625f))):((half)(0.0f)));
  half alu52 = (alu44?(((half)(val6))*((half)(0.0625f))):((half)(0.0f)));
  half alu53 = (alu44?(((half)(val7))*((half)(0.0625f))):((half)(0.0f)));
  half alu54 = (alu44?(((half)(val8))*((half)(0.0625f))):((half)(0.0f)));
  half alu55 = (alu44?(((half)(val9))*((half)(0.0625f))):((half)(0.0f)));
  half alu56 = (alu44?(((half)(val10))*((half)(0.0625f))):((half)(0.0f)));
  half alu57 = (alu44?(((half)(val11))*((half)(0.0625f))):((half)(0.0f)));
  half alu58 = (alu44?(((half)(val12))*((half)(0.0625f))):((half)(0.0f)));
  half alu59 = (alu44?(((half)(val13))*((half)(0.0625f))):((half)(0.0f)));
  half alu60 = (alu44?(((half)(val14))*((half)(0.0625f))):((half)(0.0f)));
  half alu61 = (alu44?(((half)(val15))*((half)(0.0625f))):((half)(0.0f)));
  half alu62 = (alu44?(((half)(val16))*((half)(0.0625f))):((half)(0.0f)));
  half alu63 = (alu44?(((half)(val17))*((half)(0.0625f))):((half)(0.0f)));
  half alu64 = (alu44?(((half)(val18))*((half)(0.0625f))):((half)(0.0f)));
  half alu65 = (alu44?(((half)(val19))*((half)(0.0625f))):((half)(0.0f)));
  half alu66 = (alu44?(((half)(val20))*((half)(0.0625f))):((half)(0.0f)));
  half alu67 = (alu44?(((half)(val21))*((half)(0.0625f))):((half)(0.0f)));
  half alu68 = (alu44?(((half)(val22))*((half)(0.0625f))):((half)(0.0f)));
  half alu69 = (alu44?(((half)(val23))*((half)(0.0625f))):((half)(0.0f)));
  half alu70 = (alu44?(((half)(val24))*((half)(0.0625f))):((half)(0.0f)));
  half alu71 = (alu44?(((half)(val25))*((half)(0.0625f))):((half)(0.0f)));
  half alu72 = (alu44?(((half)(val26))*((half)(0.0625f))):((half)(0.0f)));
  half alu73 = (alu44?(((half)(val27))*((half)(0.0625f))):((half)(0.0f)));
  half alu74 = (alu44?(((half)(val28))*((half)(0.0625f))):((half)(0.0f)));
  half alu75 = (alu44?(((half)(val29))*((half)(0.0625f))):((half)(0.0f)));
  half alu76 = (alu44?(((half)(val30))*((half)(0.0625f))):((half)(0.0f)));
  half alu77 = (alu44?(((half)(val31))*((half)(0.0625f))):((half)(0.0f)));
  half alu78 = (alu44?(((half)(val32))*((half)(0.0625f))):((half)(0.0f)));
  half alu79 = (alu44?(((half)(val33))*((half)(0.0625f))):((half)(0.0f)));
  half alu80 = (alu44?(((half)(val34))*((half)(0.0625f))):((half)(0.0f)));
  half alu81 = (alu44?(((half)(val35))*((half)(0.0625f))):((half)(0.0f)));
  half alu82 = (alu44?(((half)(val36))*((half)(0.0625f))):((half)(0.0f)));
  half alu83 = (alu44?(((half)(val37))*((half)(0.0625f))):((half)(0.0f)));
  half alu84 = (alu44?(((half)(val38))*((half)(0.0625f))):((half)(0.0f)));
  half alu85 = (alu44?(((half)(val39))*((half)(0.0625f))):((half)(0.0f)));
  half alu86 = (alu44?(((half)(val40))*((half)(0.0625f))):((half)(0.0f)));
  half alu87 = (alu44?(((half)(val41))*((half)(0.0625f))):((half)(0.0f)));
  half alu88 = (alu44?(((half)(val42))*((half)(0.0625f))):((half)(0.0f)));
  half alu89 = (alu44?(((half)(val43))*((half)(0.0625f))):((half)(0.0f)));
  half alu90 = (alu44?(((half)(val44))*((half)(0.0625f))):((half)(0.0f)));
  half alu91 = (alu44?(((half)(val45))*((half)(0.0625f))):((half)(0.0f)));
  half alu92 = (alu44?(((half)(val46))*((half)(0.0625f))):((half)(0.0f)));
  half alu93 = (alu44?(((half)(val47))*((half)(0.0625f))):((half)(0.0f)));
  half alu94 = (alu44?(((half)(val48))*((half)(0.0625f))):((half)(0.0f)));
  half alu95 = (alu44?(((half)(val49))*((half)(0.0625f))):((half)(0.0f)));
  half alu96 = (alu44?(((half)(val50))*((half)(0.0625f))):((half)(0.0f)));
  half alu97 = (alu44?(((half)(val51))*((half)(0.0625f))):((half)(0.0f)));
  half alu98 = (alu44?(((half)(val52))*((half)(0.0625f))):((half)(0.0f)));
  half alu99 = (alu44?(((half)(val53))*((half)(0.0625f))):((half)(0.0f)));
  half alu100 = (alu44?(((half)(val54))*((half)(0.0625f))):((half)(0.0f)));
  half alu101 = (alu44?(((half)(val55))*((half)(0.0625f))):((half)(0.0f)));
  half alu102 = (alu44?(((half)(val56))*((half)(0.0625f))):((half)(0.0f)));
  half alu103 = (alu44?(((half)(val57))*((half)(0.0625f))):((half)(0.0f)));
  half alu104 = (alu44?(((half)(val58))*((half)(0.0625f))):((half)(0.0f)));
  half alu105 = (alu44?(((half)(val59))*((half)(0.0625f))):((half)(0.0f)));
  half alu106 = (alu44?(((half)(val60))*((half)(0.0625f))):((half)(0.0f)));
  half alu107 = (alu44?(((half)(val61))*((half)(0.0625f))):((half)(0.0f)));
  half alu108 = (alu44?(((half)(val62))*((half)(0.0625f))):((half)(0.0f)));
  half alu109 = (alu44?(((half)(val63))*((half)(0.0625f))):((half)(0.0f)));
  half alu110 = (alu44?(((half)(val64))*((half)(0.0625f))):((half)(0.0f)));
  half alu111 = (alu44?(((half)(val65))*((half)(0.0625f))):((half)(0.0f)));
  int alu112 = (lidx1<<7);
  long cast0 = ((long)((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))));
  long alu113 = (((long)(alu112))+(cast0<<1ll));
  long alu114 = (cast0+32ll);
  bool alu115 = (alu114<0ll);
  long alu116 = (alu115?127ll:0ll);
  int cast1 = ((int)((((alu114+alu116)>>7ll)-((long)((((alu114%128ll)!=0ll)&(alu115!=0)))))));
  int alu117 = (cast1*34);
  int alu118 = (alu117+((alu42+8)&31)+2);
  bool alu119 = (alu118<0);
  int alu120 = (alu117+((alu42+72)&31)+2);
  bool alu121 = (alu120<0);
  int alu122 = (alu117+((alu42+136)&31)+2);
  bool alu123 = (alu122<0);
  int alu124 = (alu117+((alu42+200)&31)+2);
  bool alu125 = (alu124<0);
  long alu126 = (cast0+64ll);
  bool alu127 = (alu126<0ll);
  long alu128 = (alu127?127ll:0ll);
  int cast2 = ((int)((((alu126+alu128)>>7ll)-((long)((((alu126%128ll)!=0ll)&(alu127!=0)))))));
  int alu129 = (cast2*34);
  int alu130 = (alu129+((alu42+16)&31)+2);
  bool alu131 = (alu130<0);
  int alu132 = (alu129+((alu42+80)&31)+2);
  bool alu133 = (alu132<0);
  int alu134 = (alu129+((alu42+144)&31)+2);
  bool alu135 = (alu134<0);
  int alu136 = (alu129+((alu42+208)&31)+2);
  bool alu137 = (alu136<0);
  long alu138 = (cast0+96ll);
  bool alu139 = (alu138<0ll);
  long alu140 = (alu139?127ll:0ll);
  int cast3 = ((int)((((alu138+alu140)>>7ll)-((long)((((alu138%128ll)!=0ll)&(alu139!=0)))))));
  int alu141 = (cast3*34);
  int alu142 = (alu141+((alu42+24)&31)+2);
  bool alu143 = (alu142<0);
  int alu144 = (alu141+((alu42+88)&31)+2);
  bool alu145 = (alu144<0);
  int alu146 = (alu141+((alu42+152)&31)+2);
  bool alu147 = (alu146<0);
  int alu148 = (alu141+((alu42+216)&31)+2);
  bool alu149 = (alu148<0);
  int alu150 = (alu36?127:0);
  int alu151 = ((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))+alu150)>>7)-((int)(((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))%128)!=0)&alu37))));
  int alu152 = (alu151*34);
  int alu153 = (alu152+((alu42+64)&31)+2);
  bool alu154 = (alu153<0);
  int alu155 = (alu152+((alu42+128)&31)+2);
  bool alu156 = (alu155<0);
  int alu157 = (alu152+((alu42+192)&31)+2);
  bool alu158 = (alu157<0);
  int alu159 = (alu152+(alu42&31)+2);
  bool alu160 = (alu159<0);
  int alu161 = (cast1*17);
  int alu162 = (alu161+1);
  bool alu163 = (alu162<0);
  int alu164 = (alu117+((alu42+40)&31));
  bool alu165 = (alu164<0);
  int alu166 = (alu117+((alu42+104)&31));
  bool alu167 = (alu166<0);
  int alu168 = (alu117+((alu42+168)&31));
  bool alu169 = (alu168<0);
  int alu170 = (alu117+((alu42+232)&31));
  bool alu171 = (alu170<0);
  int alu172 = (cast2*17);
  int alu173 = (alu172+1);
  bool alu174 = (alu173<0);
  int alu175 = (alu129+((alu42+48)&31));
  bool alu176 = (alu175<0);
  int alu177 = (alu129+((alu42+112)&31));
  bool alu178 = (alu177<0);
  int alu179 = (alu129+((alu42+176)&31));
  bool alu180 = (alu179<0);
  int alu181 = (alu129+((alu42+240)&31));
  bool alu182 = (alu181<0);
  int alu183 = (cast3*17);
  int alu184 = (alu183+1);
  bool alu185 = (alu184<0);
  int alu186 = (alu141+((alu42+56)&31));
  bool alu187 = (alu186<0);
  int alu188 = (alu141+((alu42+120)&31));
  bool alu189 = (alu188<0);
  int alu190 = (alu141+((alu42+184)&31));
  bool alu191 = (alu190<0);
  int alu192 = (alu141+((alu42+248)&31));
  bool alu193 = (alu192<0);
  int alu194 = (alu151*17);
  int alu195 = (alu194+1);
  bool alu196 = (alu195<0);
  int alu197 = (alu152+((alu42+32)&31));
  bool alu198 = (alu197<0);
  int alu199 = (alu152+((alu42+96)&31));
  bool alu200 = (alu199<0);
  int alu201 = (alu152+((alu42+160)&31));
  bool alu202 = (alu201<0);
  int alu203 = (alu152+((alu42+224)&31));
  bool alu204 = (alu203<0);
  bool alu205 = (alu161<0);
  bool alu206 = (alu172<0);
  bool alu207 = (alu183<0);
  bool alu208 = (alu194<0);
  int alu209 = (alu36?1:0);
  int alu210 = (alu119?3:0);
  int alu211 = (alu121?3:0);
  int alu212 = (alu123?3:0);
  int alu213 = (alu125?3:0);
  int alu214 = (alu131?3:0);
  int alu215 = (alu133?3:0);
  int alu216 = (alu135?3:0);
  int alu217 = (alu137?3:0);
  int alu218 = (alu143?3:0);
  int alu219 = (alu145?3:0);
  int alu220 = (alu147?3:0);
  int alu221 = (alu149?3:0);
  int alu222 = (alu154?3:0);
  int alu223 = (alu156?3:0);
  int alu224 = (alu158?3:0);
  int alu225 = (alu160?3:0);
  int alu226 = (alu112+(alu38<<4)+alu42);
  int alu227 = (alu163?1:0);
  int alu228 = (alu165?3:0);
  int alu229 = (alu167?3:0);
  int alu230 = (alu169?3:0);
  int alu231 = (alu171?3:0);
  int alu232 = (alu117+alu42);
  int alu233 = (alu174?1:0);
  int alu234 = (alu176?3:0);
  int alu235 = (alu178?3:0);
  int alu236 = (alu180?3:0);
  int alu237 = (alu182?3:0);
  int alu238 = (alu129+alu42);
  int alu239 = (alu185?1:0);
  int alu240 = (alu187?3:0);
  int alu241 = (alu189?3:0);
  int alu242 = (alu191?3:0);
  int alu243 = (alu193?3:0);
  int alu244 = (alu141+alu42);
  int alu245 = (alu196?1:0);
  int alu246 = (alu198?3:0);
  int alu247 = (alu200?3:0);
  int alu248 = (alu202?3:0);
  int alu249 = (alu204?3:0);
  int alu250 = (alu152+alu42);
  int alu251 = (alu205?1:0);
  int alu252 = (alu206?1:0);
  int alu253 = (alu207?1:0);
  int alu254 = (alu208?1:0);
  int alu255 = (((int)(((cast0+1ll)&1ll)))<<4);
  int alu256 = (alu255+8);
  int alu257 = (((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))&1)<<4);
  int alu258 = (alu257+8);
  int alu259 = (((alu232+10)&3)<<3);
  int alu260 = (((alu232+76)&3)<<3);
  int alu261 = (((alu232+142)&3)<<3);
  int alu262 = (((alu232+208)&3)<<3);
  int alu263 = (((alu232+274)&3)<<3);
  int alu264 = (((alu232+340)&3)<<3);
  int alu265 = (((alu232+406)&3)<<3);
  int alu266 = (((alu232+472)&3)<<3);
  int alu267 = (((alu238+18)&3)<<3);
  int alu268 = (((alu238+84)&3)<<3);
  int alu269 = (((alu238+150)&3)<<3);
  int alu270 = (((alu238+216)&3)<<3);
  int alu271 = (((alu238+282)&3)<<3);
  int alu272 = (((alu238+348)&3)<<3);
  int alu273 = (((alu238+414)&3)<<3);
  int alu274 = (((alu238+480)&3)<<3);
  int alu275 = (((alu244+26)&3)<<3);
  int alu276 = (((alu244+92)&3)<<3);
  int alu277 = (((alu244+158)&3)<<3);
  int alu278 = (((alu244+224)&3)<<3);
  int alu279 = (((alu244+290)&3)<<3);
  int alu280 = (((alu244+356)&3)<<3);
  int alu281 = (((alu244+422)&3)<<3);
  int alu282 = (((alu244+488)&3)<<3);
  int alu283 = (((alu250+2)&3)<<3);
  int alu284 = (((alu250+68)&3)<<3);
  int alu285 = (((alu250+134)&3)<<3);
  int alu286 = (((alu250+200)&3)<<3);
  int alu287 = (((alu250+266)&3)<<3);
  int alu288 = (((alu250+332)&3)<<3);
  int alu289 = (((alu250+398)&3)<<3);
  int alu290 = (((alu250+464)&3)<<3);
  int alu291 = (((alu161+17)&1)<<4);
  int alu292 = (((alu161+34)&1)<<4);
  int alu293 = (((alu161+51)&1)<<4);
  int alu294 = (((alu161+68)&1)<<4);
  int alu295 = (((alu161+85)&1)<<4);
  int alu296 = (((alu161+102)&1)<<4);
  int alu297 = (((alu161+119)&1)<<4);
  int alu298 = (((alu172+17)&1)<<4);
  int alu299 = (((alu172+34)&1)<<4);
  int alu300 = (((alu172+51)&1)<<4);
  int alu301 = (((alu172+68)&1)<<4);
  int alu302 = (((alu172+85)&1)<<4);
  int alu303 = (((alu172+102)&1)<<4);
  int alu304 = (((alu172+119)&1)<<4);
  int alu305 = (((alu183+17)&1)<<4);
  int alu306 = (((alu183+34)&1)<<4);
  int alu307 = (((alu183+51)&1)<<4);
  int alu308 = (((alu183+68)&1)<<4);
  int alu309 = (((alu183+85)&1)<<4);
  int alu310 = (((alu183+102)&1)<<4);
  int alu311 = (((alu183+119)&1)<<4);
  int alu312 = (((alu194+17)&1)<<4);
  int alu313 = (((alu194+34)&1)<<4);
  int alu314 = (((alu194+51)&1)<<4);
  int alu315 = (((alu194+68)&1)<<4);
  int alu316 = (((alu194+85)&1)<<4);
  int alu317 = (((alu194+102)&1)<<4);
  int alu318 = (((alu194+119)&1)<<4);
  int alu319 = ((alu161&1)<<4);
  int alu320 = ((alu172&1)<<4);
  int alu321 = ((alu183&1)<<4);
  int alu322 = ((alu194&1)<<4);
  for (int Ridx4 = 0; Ridx4 < 3; Ridx4++) {
    int alu323 = (Ridx4*1088);
    long alu324 = ((((long)(alu42))*68ll)+((long)(alu323)));
    long alu325 = (alu324+((long)(((alu38+1)>>1))));
    int val66 = (*(buf3+(alu325+2ll)));
    int val67 = (*(buf3+(alu325+4ll)));
    int val68 = (*(buf3+(alu325+6ll)));
    int val69 = (*(buf3+(alu325+17ll)));
    int val70 = (*(buf3+(alu325+19ll)));
    int val71 = (*(buf3+(alu325+21ll)));
    int val72 = (*(buf3+(alu325+23ll)));
    int val73 = (*(buf3+(alu325+34ll)));
    int val74 = (*(buf3+(alu325+36ll)));
    int val75 = (*(buf3+(alu325+38ll)));
    int val76 = (*(buf3+(alu325+40ll)));
    int val77 = (*(buf3+(alu325+51ll)));
    int val78 = (*(buf3+(alu325+53ll)));
    int val79 = (*(buf3+(alu325+55ll)));
    int val80 = (*(buf3+(alu325+57ll)));
    int val81 = (*(buf3+(alu325+544ll)));
    int val82 = (*(buf3+(alu325+546ll)));
    int val83 = (*(buf3+(alu325+548ll)));
    int val84 = (*(buf3+(alu325+550ll)));
    int val85 = (*(buf3+(alu325+561ll)));
    int val86 = (*(buf3+(alu325+563ll)));
    int val87 = (*(buf3+(alu325+565ll)));
    int val88 = (*(buf3+(alu325+567ll)));
    int val89 = (*(buf3+(alu325+578ll)));
    int val90 = (*(buf3+(alu325+580ll)));
    int val91 = (*(buf3+(alu325+582ll)));
    int val92 = (*(buf3+(alu325+584ll)));
    int val93 = (*(buf3+(alu325+595ll)));
    int val94 = (*(buf3+(alu325+597ll)));
    int val95 = (*(buf3+(alu325+599ll)));
    int val96 = (*(buf3+(alu325+601ll)));
    long alu326 = (alu324+((long)((((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))+alu209)>>1)-((int)(((((((int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31) + 0 * (((int)(lidx0))))%2)!=0)&alu37))))&1))));
    int val97 = (*(buf3+(alu326+9ll)));
    int val98 = (*(buf3+(alu326+11ll)));
    int val99 = (*(buf3+(alu326+13ll)));
    int val100 = (*(buf3+(alu326+15ll)));
    int val101 = (*(buf3+(alu326+26ll)));
    int val102 = (*(buf3+(alu326+28ll)));
    int val103 = (*(buf3+(alu326+30ll)));
    int val104 = (*(buf3+(alu326+32ll)));
    int val105 = (*(buf3+(alu326+43ll)));
    int val106 = (*(buf3+(alu326+45ll)));
    int val107 = (*(buf3+(alu326+47ll)));
    int val108 = (*(buf3+(alu326+49ll)));
    int val109 = (*(buf3+(alu326+60ll)));
    int val110 = (*(buf3+(alu326+62ll)));
    int val111 = (*(buf3+(alu326+64ll)));
    int val112 = (*(buf3+(alu326+66ll)));
    int val113 = (*(buf3+(alu326+553ll)));
    int val114 = (*(buf3+(alu326+555ll)));
    int val115 = (*(buf3+(alu326+557ll)));
    int val116 = (*(buf3+(alu326+559ll)));
    int val117 = (*(buf3+(alu326+570ll)));
    int val118 = (*(buf3+(alu326+572ll)));
    int val119 = (*(buf3+(alu326+574ll)));
    int val120 = (*(buf3+(alu326+576ll)));
    int val121 = (*(buf3+(alu326+587ll)));
    int val122 = (*(buf3+(alu326+589ll)));
    int val123 = (*(buf3+(alu326+591ll)));
    int val124 = (*(buf3+(alu326+593ll)));
    int val125 = (*(buf3+(alu326+604ll)));
    int val126 = (*(buf3+(alu326+606ll)));
    int val127 = (*(buf3+(alu326+608ll)));
    int val128 = (*(buf3+(alu326+610ll)));
    int val129 = (*(buf3+alu325));
    int val130 = (*(buf3+(alu324+8ll)));
    int val131 = (*(buf3+(alu324+17ll)));
    int val132 = (*(buf3+(alu324+25ll)));
    int val133 = (*(buf3+(alu324+34ll)));
    int val134 = (*(buf3+(alu324+42ll)));
    int val135 = (*(buf3+(alu324+51ll)));
    int val136 = (*(buf3+(alu324+59ll)));
    int val137 = (*(buf3+(alu324+544ll)));
    int val138 = (*(buf3+(alu324+552ll)));
    int val139 = (*(buf3+(alu324+561ll)));
    int val140 = (*(buf3+(alu324+569ll)));
    int val141 = (*(buf3+(alu324+578ll)));
    int val142 = (*(buf3+(alu324+586ll)));
    int val143 = (*(buf3+(alu324+595ll)));
    int val144 = (*(buf3+(alu324+603ll)));
    int val145 = (*(buf3+alu324));
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
    float4 wmma0 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val129>>alu255)))))*cast5),(((half)(((signed char)((val129>>alu256)))))*cast5),(((half)(((signed char)((val81>>alu255)))))*cast4),(((half)(((signed char)((val81>>alu256)))))*cast4),(((half)(((signed char)((val66>>alu255)))))*cast5),(((half)(((signed char)((val66>>alu256)))))*cast5),(((half)(((signed char)((val82>>alu255)))))*cast4),(((half)(((signed char)((val82>>alu256)))))*cast4)), make_half4(alu48,alu49,alu50,alu51), make_float4(0.0f,0.0f,0.0f,0.0f));
    float4 wmma1 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val67>>alu255)))))*cast5),(((half)(((signed char)((val67>>alu256)))))*cast5),(((half)(((signed char)((val83>>alu255)))))*cast4),(((half)(((signed char)((val83>>alu256)))))*cast4),(((half)(((signed char)((val68>>alu255)))))*cast5),(((half)(((signed char)((val68>>alu256)))))*cast5),(((half)(((signed char)((val84>>alu255)))))*cast4),(((half)(((signed char)((val84>>alu256)))))*cast4)), make_half4(alu52,alu53,alu54,alu55), wmma0);
    float4 wmma2 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val97>>alu257)))))*cast12),(((half)(((signed char)((val97>>alu258)))))*cast12),(((half)(((signed char)((val113>>alu257)))))*cast13),(((half)(((signed char)((val113>>alu258)))))*cast13),(((half)(((signed char)((val98>>alu257)))))*cast12),(((half)(((signed char)((val98>>alu258)))))*cast12),(((half)(((signed char)((val114>>alu257)))))*cast13),(((half)(((signed char)((val114>>alu258)))))*cast13)), make_half4(alu56,alu57,alu58,alu59), wmma1);
    float4 wmma3 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val99>>alu257)))))*cast12),(((half)(((signed char)((val99>>alu258)))))*cast12),(((half)(((signed char)((val115>>alu257)))))*cast13),(((half)(((signed char)((val115>>alu258)))))*cast13),(((half)(((signed char)((val100>>alu257)))))*cast12),(((half)(((signed char)((val100>>alu258)))))*cast12),(((half)(((signed char)((val116>>alu257)))))*cast13),(((half)(((signed char)((val116>>alu258)))))*cast13)), make_half4(alu60,alu61,alu62,alu63), wmma2);
    float4 wmma4 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val69>>alu255)))))*cast6),(((half)(((signed char)((val69>>alu256)))))*cast6),(((half)(((signed char)((val85>>alu255)))))*cast7),(((half)(((signed char)((val85>>alu256)))))*cast7),(((half)(((signed char)((val70>>alu255)))))*cast6),(((half)(((signed char)((val70>>alu256)))))*cast6),(((half)(((signed char)((val86>>alu255)))))*cast7),(((half)(((signed char)((val86>>alu256)))))*cast7)), make_half4(alu64,alu65,alu66,alu67), wmma3);
    float4 wmma5 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val71>>alu255)))))*cast6),(((half)(((signed char)((val71>>alu256)))))*cast6),(((half)(((signed char)((val87>>alu255)))))*cast7),(((half)(((signed char)((val87>>alu256)))))*cast7),(((half)(((signed char)((val72>>alu255)))))*cast6),(((half)(((signed char)((val72>>alu256)))))*cast6),(((half)(((signed char)((val88>>alu255)))))*cast7),(((half)(((signed char)((val88>>alu256)))))*cast7)), make_half4(alu68,alu69,alu70,alu71), wmma4);
    float4 wmma6 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val101>>alu257)))))*cast14),(((half)(((signed char)((val101>>alu258)))))*cast14),(((half)(((signed char)((val117>>alu257)))))*cast15),(((half)(((signed char)((val117>>alu258)))))*cast15),(((half)(((signed char)((val102>>alu257)))))*cast14),(((half)(((signed char)((val102>>alu258)))))*cast14),(((half)(((signed char)((val118>>alu257)))))*cast15),(((half)(((signed char)((val118>>alu258)))))*cast15)), make_half4(alu72,alu73,alu74,alu75), wmma5);
    float4 wmma7 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val103>>alu257)))))*cast14),(((half)(((signed char)((val103>>alu258)))))*cast14),(((half)(((signed char)((val119>>alu257)))))*cast15),(((half)(((signed char)((val119>>alu258)))))*cast15),(((half)(((signed char)((val104>>alu257)))))*cast14),(((half)(((signed char)((val104>>alu258)))))*cast14),(((half)(((signed char)((val120>>alu257)))))*cast15),(((half)(((signed char)((val120>>alu258)))))*cast15)), make_half4(alu76,alu77,alu78,alu79), wmma6);
    float4 wmma8 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val73>>alu255)))))*cast8),(((half)(((signed char)((val73>>alu256)))))*cast8),(((half)(((signed char)((val89>>alu255)))))*cast9),(((half)(((signed char)((val89>>alu256)))))*cast9),(((half)(((signed char)((val74>>alu255)))))*cast8),(((half)(((signed char)((val74>>alu256)))))*cast8),(((half)(((signed char)((val90>>alu255)))))*cast9),(((half)(((signed char)((val90>>alu256)))))*cast9)), make_half4(alu80,alu81,alu82,alu83), wmma7);
    float4 wmma9 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val75>>alu255)))))*cast8),(((half)(((signed char)((val75>>alu256)))))*cast8),(((half)(((signed char)((val91>>alu255)))))*cast9),(((half)(((signed char)((val91>>alu256)))))*cast9),(((half)(((signed char)((val76>>alu255)))))*cast8),(((half)(((signed char)((val76>>alu256)))))*cast8),(((half)(((signed char)((val92>>alu255)))))*cast9),(((half)(((signed char)((val92>>alu256)))))*cast9)), make_half4(alu84,alu85,alu86,alu87), wmma8);
    float4 wmma10 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val105>>alu257)))))*cast16),(((half)(((signed char)((val105>>alu258)))))*cast16),(((half)(((signed char)((val121>>alu257)))))*cast17),(((half)(((signed char)((val121>>alu258)))))*cast17),(((half)(((signed char)((val106>>alu257)))))*cast16),(((half)(((signed char)((val106>>alu258)))))*cast16),(((half)(((signed char)((val122>>alu257)))))*cast17),(((half)(((signed char)((val122>>alu258)))))*cast17)), make_half4(alu88,alu89,alu90,alu91), wmma9);
    float4 wmma11 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val107>>alu257)))))*cast16),(((half)(((signed char)((val107>>alu258)))))*cast16),(((half)(((signed char)((val123>>alu257)))))*cast17),(((half)(((signed char)((val123>>alu258)))))*cast17),(((half)(((signed char)((val108>>alu257)))))*cast16),(((half)(((signed char)((val108>>alu258)))))*cast16),(((half)(((signed char)((val124>>alu257)))))*cast17),(((half)(((signed char)((val124>>alu258)))))*cast17)), make_half4(alu92,alu93,alu94,alu95), wmma10);
    float4 wmma12 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val77>>alu255)))))*cast10),(((half)(((signed char)((val77>>alu256)))))*cast10),(((half)(((signed char)((val93>>alu255)))))*cast11),(((half)(((signed char)((val93>>alu256)))))*cast11),(((half)(((signed char)((val78>>alu255)))))*cast10),(((half)(((signed char)((val78>>alu256)))))*cast10),(((half)(((signed char)((val94>>alu255)))))*cast11),(((half)(((signed char)((val94>>alu256)))))*cast11)), make_half4(alu96,alu97,alu98,alu99), wmma11);
    float4 wmma13 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val79>>alu255)))))*cast10),(((half)(((signed char)((val79>>alu256)))))*cast10),(((half)(((signed char)((val95>>alu255)))))*cast11),(((half)(((signed char)((val95>>alu256)))))*cast11),(((half)(((signed char)((val80>>alu255)))))*cast10),(((half)(((signed char)((val80>>alu256)))))*cast10),(((half)(((signed char)((val96>>alu255)))))*cast11),(((half)(((signed char)((val96>>alu256)))))*cast11)), make_half4(alu100,alu101,alu102,alu103), wmma12);
    float4 wmma14 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val109>>alu257)))))*cast18),(((half)(((signed char)((val109>>alu258)))))*cast18),(((half)(((signed char)((val125>>alu257)))))*cast19),(((half)(((signed char)((val125>>alu258)))))*cast19),(((half)(((signed char)((val110>>alu257)))))*cast18),(((half)(((signed char)((val110>>alu258)))))*cast18),(((half)(((signed char)((val126>>alu257)))))*cast19),(((half)(((signed char)((val126>>alu258)))))*cast19)), make_half4(alu104,alu105,alu106,alu107), wmma13);
    float4 wmma15 = __WMMA_8_16_16_half_float(make_half8((((half)(((signed char)((val111>>alu257)))))*cast18),(((half)(((signed char)((val111>>alu258)))))*cast18),(((half)(((signed char)((val127>>alu257)))))*cast19),(((half)(((signed char)((val127>>alu258)))))*cast19),(((half)(((signed char)((val112>>alu257)))))*cast18),(((half)(((signed char)((val112>>alu258)))))*cast18),(((half)(((signed char)((val128>>alu257)))))*cast19),(((half)(((signed char)((val128>>alu258)))))*cast19)), make_half4(alu108,alu109,alu110,alu111), wmma14);
    int alu327 = (alu16+(Ridx4<<6)+alu17+alu42);
    bool alu328 = ((alu14+Ridx4)<alu22);
    bool alu329 = (alu328&((alu327+8)<val0));
    float alu330 = (alu329?wmma15.z:((float)(-INFINITY)));
    float alu331 = (alu330+2.0792999267578125f);
    bool alu332 = (alu328&(alu327<val0));
    float alu333 = (alu332?wmma15.x:((float)(-INFINITY)));
    float alu334 = (alu333+2.0792999267578125f);
    float alu335 = ((alu334<alu331)?alu331:alu334);
    float alu336 = (((*(buf1+0))<alu335)?alu335:(*(buf1+0)));
    buf6 = __shfl_xor_sync(0xffffffffu, alu336, 16);
    float alu338 = ((alu336<buf6)?buf6:alu336);
    buf7 = __shfl_xor_sync(0xffffffffu, alu338, 8);
    float alu340 = ((alu338<buf7)?buf7:alu338);
    buf8 = __shfl_xor_sync(0xffffffffu, alu340, 4);
    float alu342 = (alu329?wmma15.w:((float)(-INFINITY)));
    float alu343 = (alu342+2.0792999267578125f);
    float alu344 = (alu332?wmma15.y:((float)(-INFINITY)));
    float alu345 = (alu344+2.0792999267578125f);
    float alu346 = ((alu345<alu343)?alu343:alu345);
    float alu347 = (((*(buf1+1))<alu346)?alu346:(*(buf1+1)));
    buf9 = __shfl_xor_sync(0xffffffffu, alu347, 16);
    float alu349 = ((alu347<buf9)?buf9:alu347);
    buf10 = __shfl_xor_sync(0xffffffffu, alu349, 8);
    float alu351 = ((alu349<buf10)?buf10:alu349);
    buf11 = __shfl_xor_sync(0xffffffffu, alu351, 4);
    float alu353 = ((alu340<buf8)?buf8:alu340);
    float alu354 = exp2(((alu330-alu353)*1.4426950216293335f));
    float alu355 = ((alu351<buf11)?buf11:alu351);
    float alu356 = exp2(((alu342-alu355)*1.4426950216293335f));
    float alu357 = exp2(((alu333-alu353)*1.4426950216293335f));
    float alu358 = exp2(((alu344-alu355)*1.4426950216293335f));
    float alu359 = ((*(buf1+0))-alu353);
    float alu360 = exp2((alu359*1.4426950216293335f));
    float alu361 = ((*(buf1+1))-alu355);
    float alu362 = exp2((alu361*1.4426950216293335f));
    bool alu363 = (alu359<-20.0f);
    bool alu364 = (alu361<-20.0f);
    float alu365 = (alu363?0.0f:alu360);
    float alu366 = (alu364?0.0f:alu362);
    *(buf5+alu113) = ((half)(alu357));
    *(buf5+(alu113+1ll)) = ((half)(alu358));
    *(buf5+(alu113+64ll)) = ((half)(alu354));
    *(buf5+(alu113+65ll)) = ((half)(alu356));
    (*(buf1+0)) = alu353;
    (*(buf1+1)) = alu355;
    (*(buf2+0)) = fmaf(alu365, (*(buf2+0)), (alu357+alu354));
    (*(buf2+1)) = fmaf(alu366, (*(buf2+1)), (alu358+alu356));
    __syncthreads();
    int alu376 = ((alu38*136)+alu323);
    int alu377 = (alu376+(((alu118+alu210)>>2)-((int)((((alu118%4)!=0)&(alu119!=0))))));
    int val146 = (*(buf4+((long)((alu377+68)))));
    int val147 = (*(buf4+((long)((alu377+544)))));
    int val148 = (*(buf4+((long)((alu377+612)))));
    int alu378 = (alu376+(((alu120+alu211)>>2)-((int)((((alu120%4)!=0)&(alu121!=0))))));
    int val149 = (*(buf4+((long)((alu378+17)))));
    int val150 = (*(buf4+((long)((alu378+85)))));
    int val151 = (*(buf4+((long)((alu378+561)))));
    int val152 = (*(buf4+((long)((alu378+629)))));
    int alu379 = (alu376+(((alu122+alu212)>>2)-((int)((((alu122%4)!=0)&(alu123!=0))))));
    int val153 = (*(buf4+((long)((alu379+34)))));
    int val154 = (*(buf4+((long)((alu379+102)))));
    int val155 = (*(buf4+((long)((alu379+578)))));
    int val156 = (*(buf4+((long)((alu379+646)))));
    int alu380 = (alu376+(((alu124+alu213)>>2)-((int)((((alu124%4)!=0)&(alu125!=0))))));
    int val157 = (*(buf4+((long)((alu380+51)))));
    int val158 = (*(buf4+((long)((alu380+119)))));
    int val159 = (*(buf4+((long)((alu380+595)))));
    int val160 = (*(buf4+((long)((alu380+663)))));
    int alu381 = (alu376+(((alu130+alu214)>>2)-((int)((((alu130%4)!=0)&(alu131!=0))))));
    int val161 = (*(buf4+((long)((alu381+68)))));
    int val162 = (*(buf4+((long)((alu381+544)))));
    int val163 = (*(buf4+((long)((alu381+612)))));
    int alu382 = (alu376+(((alu132+alu215)>>2)-((int)((((alu132%4)!=0)&(alu133!=0))))));
    int val164 = (*(buf4+((long)((alu382+17)))));
    int val165 = (*(buf4+((long)((alu382+85)))));
    int val166 = (*(buf4+((long)((alu382+561)))));
    int val167 = (*(buf4+((long)((alu382+629)))));
    int alu383 = (alu376+(((alu134+alu216)>>2)-((int)((((alu134%4)!=0)&(alu135!=0))))));
    int val168 = (*(buf4+((long)((alu383+34)))));
    int val169 = (*(buf4+((long)((alu383+102)))));
    int val170 = (*(buf4+((long)((alu383+578)))));
    int val171 = (*(buf4+((long)((alu383+646)))));
    int alu384 = (alu376+(((alu136+alu217)>>2)-((int)((((alu136%4)!=0)&(alu137!=0))))));
    int val172 = (*(buf4+((long)((alu384+51)))));
    int val173 = (*(buf4+((long)((alu384+119)))));
    int val174 = (*(buf4+((long)((alu384+595)))));
    int val175 = (*(buf4+((long)((alu384+663)))));
    int alu385 = (alu376+(((alu142+alu218)>>2)-((int)((((alu142%4)!=0)&(alu143!=0))))));
    int val176 = (*(buf4+((long)((alu385+68)))));
    int val177 = (*(buf4+((long)((alu385+544)))));
    int val178 = (*(buf4+((long)((alu385+612)))));
    int alu386 = (alu376+(((alu144+alu219)>>2)-((int)((((alu144%4)!=0)&(alu145!=0))))));
    int val179 = (*(buf4+((long)((alu386+17)))));
    int val180 = (*(buf4+((long)((alu386+85)))));
    int val181 = (*(buf4+((long)((alu386+561)))));
    int val182 = (*(buf4+((long)((alu386+629)))));
    int alu387 = (alu376+(((alu146+alu220)>>2)-((int)((((alu146%4)!=0)&(alu147!=0))))));
    int val183 = (*(buf4+((long)((alu387+34)))));
    int val184 = (*(buf4+((long)((alu387+102)))));
    int val185 = (*(buf4+((long)((alu387+578)))));
    int val186 = (*(buf4+((long)((alu387+646)))));
    int alu388 = (alu376+(((alu148+alu221)>>2)-((int)((((alu148%4)!=0)&(alu149!=0))))));
    int val187 = (*(buf4+((long)((alu388+51)))));
    int val188 = (*(buf4+((long)((alu388+119)))));
    int val189 = (*(buf4+((long)((alu388+595)))));
    int val190 = (*(buf4+((long)((alu388+663)))));
    int alu389 = (alu376+(((alu162+alu227)>>1)-((int)((((alu162%2)!=0)&(alu163!=0))))));
    int val191 = (*(buf4+((long)((alu389+8)))));
    int val192 = (*(buf4+((long)((alu389+25)))));
    int val193 = (*(buf4+((long)((alu389+42)))));
    int val194 = (*(buf4+((long)((alu389+59)))));
    int val195 = (*(buf4+((long)((alu389+76)))));
    int val196 = (*(buf4+((long)((alu389+93)))));
    int val197 = (*(buf4+((long)((alu389+110)))));
    int val198 = (*(buf4+((long)((alu389+127)))));
    int val199 = (*(buf4+((long)((alu389+552)))));
    int val200 = (*(buf4+((long)((alu389+569)))));
    int val201 = (*(buf4+((long)((alu389+586)))));
    int val202 = (*(buf4+((long)((alu389+603)))));
    int val203 = (*(buf4+((long)((alu389+620)))));
    int val204 = (*(buf4+((long)((alu389+637)))));
    int val205 = (*(buf4+((long)((alu389+654)))));
    int val206 = (*(buf4+((long)((alu389+671)))));
    int alu390 = (alu376+(((alu164+alu228)>>2)-((int)((((alu164%4)!=0)&(alu165!=0))))));
    int val207 = (*(buf4+((long)((alu390+9)))));
    int val208 = (*(buf4+((long)((alu390+77)))));
    int val209 = (*(buf4+((long)((alu390+553)))));
    int val210 = (*(buf4+((long)((alu390+621)))));
    int alu391 = (alu376+(((alu166+alu229)>>2)-((int)((((alu166%4)!=0)&(alu167!=0))))));
    int val211 = (*(buf4+((long)((alu391+26)))));
    int val212 = (*(buf4+((long)((alu391+94)))));
    int val213 = (*(buf4+((long)((alu391+570)))));
    int val214 = (*(buf4+((long)((alu391+638)))));
    int alu392 = (alu376+(((alu168+alu230)>>2)-((int)((((alu168%4)!=0)&(alu169!=0))))));
    int val215 = (*(buf4+((long)((alu392+43)))));
    int val216 = (*(buf4+((long)((alu392+111)))));
    int val217 = (*(buf4+((long)((alu392+587)))));
    int val218 = (*(buf4+((long)((alu392+655)))));
    int alu393 = (alu376+(((alu170+alu231)>>2)-((int)((((alu170%4)!=0)&(alu171!=0))))));
    int val219 = (*(buf4+((long)((alu393+60)))));
    int val220 = (*(buf4+((long)((alu393+128)))));
    int val221 = (*(buf4+((long)((alu393+604)))));
    int val222 = (*(buf4+((long)((alu393+672)))));
    int alu394 = (alu376+(((alu173+alu233)>>1)-((int)((((alu173%2)!=0)&(alu174!=0))))));
    int val223 = (*(buf4+((long)((alu394+8)))));
    int val224 = (*(buf4+((long)((alu394+25)))));
    int val225 = (*(buf4+((long)((alu394+42)))));
    int val226 = (*(buf4+((long)((alu394+59)))));
    int val227 = (*(buf4+((long)((alu394+76)))));
    int val228 = (*(buf4+((long)((alu394+93)))));
    int val229 = (*(buf4+((long)((alu394+110)))));
    int val230 = (*(buf4+((long)((alu394+127)))));
    int val231 = (*(buf4+((long)((alu394+552)))));
    int val232 = (*(buf4+((long)((alu394+569)))));
    int val233 = (*(buf4+((long)((alu394+586)))));
    int val234 = (*(buf4+((long)((alu394+603)))));
    int val235 = (*(buf4+((long)((alu394+620)))));
    int val236 = (*(buf4+((long)((alu394+637)))));
    int val237 = (*(buf4+((long)((alu394+654)))));
    int val238 = (*(buf4+((long)((alu394+671)))));
    int alu395 = (alu376+(((alu175+alu234)>>2)-((int)((((alu175%4)!=0)&(alu176!=0))))));
    int val239 = (*(buf4+((long)((alu395+9)))));
    int val240 = (*(buf4+((long)((alu395+77)))));
    int val241 = (*(buf4+((long)((alu395+553)))));
    int val242 = (*(buf4+((long)((alu395+621)))));
    int alu396 = (alu376+(((alu177+alu235)>>2)-((int)((((alu177%4)!=0)&(alu178!=0))))));
    int val243 = (*(buf4+((long)((alu396+26)))));
    int val244 = (*(buf4+((long)((alu396+94)))));
    int val245 = (*(buf4+((long)((alu396+570)))));
    int val246 = (*(buf4+((long)((alu396+638)))));
    int alu397 = (alu376+(((alu179+alu236)>>2)-((int)((((alu179%4)!=0)&(alu180!=0))))));
    int val247 = (*(buf4+((long)((alu397+43)))));
    int val248 = (*(buf4+((long)((alu397+111)))));
    int val249 = (*(buf4+((long)((alu397+587)))));
    int val250 = (*(buf4+((long)((alu397+655)))));
    int alu398 = (alu376+(((alu181+alu237)>>2)-((int)((((alu181%4)!=0)&(alu182!=0))))));
    int val251 = (*(buf4+((long)((alu398+60)))));
    int val252 = (*(buf4+((long)((alu398+128)))));
    int val253 = (*(buf4+((long)((alu398+604)))));
    int val254 = (*(buf4+((long)((alu398+672)))));
    int alu399 = (alu376+(((alu184+alu239)>>1)-((int)((((alu184%2)!=0)&(alu185!=0))))));
    int val255 = (*(buf4+((long)((alu399+8)))));
    int val256 = (*(buf4+((long)((alu399+25)))));
    int val257 = (*(buf4+((long)((alu399+42)))));
    int val258 = (*(buf4+((long)((alu399+59)))));
    int val259 = (*(buf4+((long)((alu399+76)))));
    int val260 = (*(buf4+((long)((alu399+93)))));
    int val261 = (*(buf4+((long)((alu399+110)))));
    int val262 = (*(buf4+((long)((alu399+127)))));
    int val263 = (*(buf4+((long)((alu399+552)))));
    int val264 = (*(buf4+((long)((alu399+569)))));
    int val265 = (*(buf4+((long)((alu399+586)))));
    int val266 = (*(buf4+((long)((alu399+603)))));
    int val267 = (*(buf4+((long)((alu399+620)))));
    int val268 = (*(buf4+((long)((alu399+637)))));
    int val269 = (*(buf4+((long)((alu399+654)))));
    int val270 = (*(buf4+((long)((alu399+671)))));
    int alu400 = (alu376+(((alu186+alu240)>>2)-((int)((((alu186%4)!=0)&(alu187!=0))))));
    int val271 = (*(buf4+((long)((alu400+9)))));
    int val272 = (*(buf4+((long)((alu400+77)))));
    int val273 = (*(buf4+((long)((alu400+553)))));
    int val274 = (*(buf4+((long)((alu400+621)))));
    int alu401 = (alu376+(((alu188+alu241)>>2)-((int)((((alu188%4)!=0)&(alu189!=0))))));
    int val275 = (*(buf4+((long)((alu401+26)))));
    int val276 = (*(buf4+((long)((alu401+94)))));
    int val277 = (*(buf4+((long)((alu401+570)))));
    int val278 = (*(buf4+((long)((alu401+638)))));
    int alu402 = (alu376+(((alu190+alu242)>>2)-((int)((((alu190%4)!=0)&(alu191!=0))))));
    int val279 = (*(buf4+((long)((alu402+43)))));
    int val280 = (*(buf4+((long)((alu402+111)))));
    int val281 = (*(buf4+((long)((alu402+587)))));
    int val282 = (*(buf4+((long)((alu402+655)))));
    int alu403 = (alu376+(((alu192+alu243)>>2)-((int)((((alu192%4)!=0)&(alu193!=0))))));
    int val283 = (*(buf4+((long)((alu403+60)))));
    int val284 = (*(buf4+((long)((alu403+128)))));
    int val285 = (*(buf4+((long)((alu403+604)))));
    int val286 = (*(buf4+((long)((alu403+672)))));
    int alu404 = (alu376+(((alu161+alu251)>>1)-((int)((((alu161%2)!=0)&(alu205!=0))))));
    int val287 = (*(buf4+((long)((alu404+17)))));
    int val288 = (*(buf4+((long)((alu404+34)))));
    int val289 = (*(buf4+((long)((alu404+51)))));
    int val290 = (*(buf4+((long)((alu404+68)))));
    int val291 = (*(buf4+((long)((alu404+85)))));
    int val292 = (*(buf4+((long)((alu404+102)))));
    int val293 = (*(buf4+((long)((alu404+119)))));
    int val294 = (*(buf4+((long)((alu404+544)))));
    int val295 = (*(buf4+((long)((alu404+561)))));
    int val296 = (*(buf4+((long)((alu404+578)))));
    int val297 = (*(buf4+((long)((alu404+595)))));
    int val298 = (*(buf4+((long)((alu404+612)))));
    int val299 = (*(buf4+((long)((alu404+629)))));
    int val300 = (*(buf4+((long)((alu404+646)))));
    int val301 = (*(buf4+((long)((alu404+663)))));
    int alu405 = (alu376+(((alu172+alu252)>>1)-((int)((((alu172%2)!=0)&(alu206!=0))))));
    int val302 = (*(buf4+((long)((alu405+17)))));
    int val303 = (*(buf4+((long)((alu405+34)))));
    int val304 = (*(buf4+((long)((alu405+51)))));
    int val305 = (*(buf4+((long)((alu405+68)))));
    int val306 = (*(buf4+((long)((alu405+85)))));
    int val307 = (*(buf4+((long)((alu405+102)))));
    int val308 = (*(buf4+((long)((alu405+119)))));
    int val309 = (*(buf4+((long)((alu405+544)))));
    int val310 = (*(buf4+((long)((alu405+561)))));
    int val311 = (*(buf4+((long)((alu405+578)))));
    int val312 = (*(buf4+((long)((alu405+595)))));
    int val313 = (*(buf4+((long)((alu405+612)))));
    int val314 = (*(buf4+((long)((alu405+629)))));
    int val315 = (*(buf4+((long)((alu405+646)))));
    int val316 = (*(buf4+((long)((alu405+663)))));
    int alu406 = (alu376+(((alu183+alu253)>>1)-((int)((((alu183%2)!=0)&(alu207!=0))))));
    int val317 = (*(buf4+((long)((alu406+17)))));
    int val318 = (*(buf4+((long)((alu406+34)))));
    int val319 = (*(buf4+((long)((alu406+51)))));
    int val320 = (*(buf4+((long)((alu406+68)))));
    int val321 = (*(buf4+((long)((alu406+85)))));
    int val322 = (*(buf4+((long)((alu406+102)))));
    int val323 = (*(buf4+((long)((alu406+119)))));
    int val324 = (*(buf4+((long)((alu406+544)))));
    int val325 = (*(buf4+((long)((alu406+561)))));
    int val326 = (*(buf4+((long)((alu406+578)))));
    int val327 = (*(buf4+((long)((alu406+595)))));
    int val328 = (*(buf4+((long)((alu406+612)))));
    int val329 = (*(buf4+((long)((alu406+629)))));
    int val330 = (*(buf4+((long)((alu406+646)))));
    int val331 = (*(buf4+((long)((alu406+663)))));
    int val332 = (*(buf4+((long)(alu377))));
    int val333 = (*(buf4+((long)(alu381))));
    int val334 = (*(buf4+((long)(alu385))));
    int val335 = (*(buf4+((long)(alu404))));
    int val336 = (*(buf4+((long)(alu405))));
    int val337 = (*(buf4+((long)(alu406))));
    int alu407 = (alu376+(((alu153+alu222)>>2)-((int)((((alu153%4)!=0)&(alu154!=0))))));
    int val338 = (*(buf4+(alu407+17)));
    int val339 = (*(buf4+(alu407+85)));
    int val340 = (*(buf4+(alu407+561)));
    int val341 = (*(buf4+(alu407+629)));
    int alu408 = (alu376+(((alu155+alu223)>>2)-((int)((((alu155%4)!=0)&(alu156!=0))))));
    int val342 = (*(buf4+(alu408+34)));
    int val343 = (*(buf4+(alu408+102)));
    int val344 = (*(buf4+(alu408+578)));
    int val345 = (*(buf4+(alu408+646)));
    int alu409 = (alu376+(((alu157+alu224)>>2)-((int)((((alu157%4)!=0)&(alu158!=0))))));
    int val346 = (*(buf4+(alu409+51)));
    int val347 = (*(buf4+(alu409+119)));
    int val348 = (*(buf4+(alu409+595)));
    int val349 = (*(buf4+(alu409+663)));
    int alu410 = (alu376+(((alu159+alu225)>>2)-((int)((((alu159%4)!=0)&(alu160!=0))))));
    int val350 = (*(buf4+(alu410+68)));
    int val351 = (*(buf4+(alu410+544)));
    int val352 = (*(buf4+(alu410+612)));
    int alu411 = (alu376+(((alu195+alu245)>>1)-((int)((((alu195%2)!=0)&(alu196!=0))))));
    int val353 = (*(buf4+(alu411+8)));
    int val354 = (*(buf4+(alu411+25)));
    int val355 = (*(buf4+(alu411+42)));
    int val356 = (*(buf4+(alu411+59)));
    int val357 = (*(buf4+(alu411+76)));
    int val358 = (*(buf4+(alu411+93)));
    int val359 = (*(buf4+(alu411+110)));
    int val360 = (*(buf4+(alu411+127)));
    int val361 = (*(buf4+(alu411+552)));
    int val362 = (*(buf4+(alu411+569)));
    int val363 = (*(buf4+(alu411+586)));
    int val364 = (*(buf4+(alu411+603)));
    int val365 = (*(buf4+(alu411+620)));
    int val366 = (*(buf4+(alu411+637)));
    int val367 = (*(buf4+(alu411+654)));
    int val368 = (*(buf4+(alu411+671)));
    int alu412 = (alu376+(((alu197+alu246)>>2)-((int)((((alu197%4)!=0)&(alu198!=0))))));
    int val369 = (*(buf4+(alu412+9)));
    int val370 = (*(buf4+(alu412+77)));
    int val371 = (*(buf4+(alu412+553)));
    int val372 = (*(buf4+(alu412+621)));
    int alu413 = (alu376+(((alu199+alu247)>>2)-((int)((((alu199%4)!=0)&(alu200!=0))))));
    int val373 = (*(buf4+(alu413+26)));
    int val374 = (*(buf4+(alu413+94)));
    int val375 = (*(buf4+(alu413+570)));
    int val376 = (*(buf4+(alu413+638)));
    int alu414 = (alu376+(((alu201+alu248)>>2)-((int)((((alu201%4)!=0)&(alu202!=0))))));
    int val377 = (*(buf4+(alu414+43)));
    int val378 = (*(buf4+(alu414+111)));
    int val379 = (*(buf4+(alu414+587)));
    int val380 = (*(buf4+(alu414+655)));
    int alu415 = (alu376+(((alu203+alu249)>>2)-((int)((((alu203%4)!=0)&(alu204!=0))))));
    int val381 = (*(buf4+(alu415+60)));
    int val382 = (*(buf4+(alu415+128)));
    int val383 = (*(buf4+(alu415+604)));
    int val384 = (*(buf4+(alu415+672)));
    int alu416 = (alu376+(((alu194+alu254)>>1)-((int)((((alu194%2)!=0)&(alu208!=0))))));
    int val385 = (*(buf4+(alu416+17)));
    int val386 = (*(buf4+(alu416+34)));
    int val387 = (*(buf4+(alu416+51)));
    int val388 = (*(buf4+(alu416+68)));
    int val389 = (*(buf4+(alu416+85)));
    int val390 = (*(buf4+(alu416+102)));
    int val391 = (*(buf4+(alu416+119)));
    int val392 = (*(buf4+(alu416+544)));
    int val393 = (*(buf4+(alu416+561)));
    int val394 = (*(buf4+(alu416+578)));
    int val395 = (*(buf4+(alu416+595)));
    int val396 = (*(buf4+(alu416+612)));
    int val397 = (*(buf4+(alu416+629)));
    int val398 = (*(buf4+(alu416+646)));
    int val399 = (*(buf4+(alu416+663)));
    int val400 = (*(buf4+alu410));
    int val401 = (*(buf4+alu416));
    half val402 = (*(buf5+(alu226+8)));
    half val403 = (*(buf5+(alu226+64)));
    half val404 = (*(buf5+(alu226+72)));
    half val405 = (*(buf5+alu226));
    half4 cast20 = make_half4(val405,val402,val403,val404);
    half alu417 = (alu363?((half)(0.0f)):((half)(alu360)));
    half alu418 = (alu364?((half)(0.0f)):((half)(alu362)));
    half4 wmma16 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val164>>alu269)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val302>>alu299)))))),(((half)(((signed char)((val165>>alu269)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val306>>alu299)))))),(((half)(((signed char)((val179>>alu277)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val317>>alu306)))))),(((half)(((signed char)((val180>>alu277)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val321>>alu306)))))),(((half)(((signed char)((val166>>alu269)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val310>>alu299)))))),(((half)(((signed char)((val167>>alu269)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val314>>alu299)))))),(((half)(((signed char)((val181>>alu277)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val325>>alu306)))))),(((half)(((signed char)((val182>>alu277)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val329>>alu306))))))), cast20, make_half4(((*(buf0+20))*alu417),((*(buf0+21))*alu418),((*(buf0+22))*alu417),((*(buf0+23))*alu418)));
    half4 wmma17 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val168>>alu271)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val303>>alu301)))))),(((half)(((signed char)((val169>>alu271)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val307>>alu301)))))),(((half)(((signed char)((val183>>alu279)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val318>>alu308)))))),(((half)(((signed char)((val184>>alu279)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val322>>alu308)))))),(((half)(((signed char)((val170>>alu271)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val311>>alu301)))))),(((half)(((signed char)((val171>>alu271)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val315>>alu301)))))),(((half)(((signed char)((val185>>alu279)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val326>>alu308)))))),(((half)(((signed char)((val186>>alu279)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val330>>alu308))))))), cast20, make_half4(((*(buf0+36))*alu417),((*(buf0+37))*alu418),((*(buf0+38))*alu417),((*(buf0+39))*alu418)));
    half4 wmma18 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val172>>alu273)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val304>>alu303)))))),(((half)(((signed char)((val173>>alu273)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val308>>alu303)))))),(((half)(((signed char)((val187>>alu281)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val319>>alu310)))))),(((half)(((signed char)((val188>>alu281)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val323>>alu310)))))),(((half)(((signed char)((val174>>alu273)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val312>>alu303)))))),(((half)(((signed char)((val175>>alu273)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val316>>alu303)))))),(((half)(((signed char)((val189>>alu281)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val327>>alu310)))))),(((half)(((signed char)((val190>>alu281)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val331>>alu310))))))), cast20, make_half4(((*(buf0+52))*alu417),((*(buf0+53))*alu418),((*(buf0+54))*alu417),((*(buf0+55))*alu418)));
    half4 wmma19 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val239>>alu268)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val223>>alu298)))))),(((half)(((signed char)((val240>>alu268)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val227>>alu298)))))),(((half)(((signed char)((val271>>alu276)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val255>>alu305)))))),(((half)(((signed char)((val272>>alu276)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val259>>alu305)))))),(((half)(((signed char)((val241>>alu268)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val231>>alu298)))))),(((half)(((signed char)((val242>>alu268)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val235>>alu298)))))),(((half)(((signed char)((val273>>alu276)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val263>>alu305)))))),(((half)(((signed char)((val274>>alu276)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val267>>alu305))))))), cast20, make_half4(((*(buf0+12))*alu417),((*(buf0+13))*alu418),((*(buf0+14))*alu417),((*(buf0+15))*alu418)));
    half4 wmma20 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val243>>alu270)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val224>>alu300)))))),(((half)(((signed char)((val244>>alu270)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val228>>alu300)))))),(((half)(((signed char)((val275>>alu278)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val256>>alu307)))))),(((half)(((signed char)((val276>>alu278)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val260>>alu307)))))),(((half)(((signed char)((val245>>alu270)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val232>>alu300)))))),(((half)(((signed char)((val246>>alu270)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val236>>alu300)))))),(((half)(((signed char)((val277>>alu278)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val264>>alu307)))))),(((half)(((signed char)((val278>>alu278)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val268>>alu307))))))), cast20, make_half4(((*(buf0+28))*alu417),((*(buf0+29))*alu418),((*(buf0+30))*alu417),((*(buf0+31))*alu418)));
    half4 wmma21 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val247>>alu272)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val225>>alu302)))))),(((half)(((signed char)((val248>>alu272)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val229>>alu302)))))),(((half)(((signed char)((val279>>alu280)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val257>>alu309)))))),(((half)(((signed char)((val280>>alu280)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val261>>alu309)))))),(((half)(((signed char)((val249>>alu272)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val233>>alu302)))))),(((half)(((signed char)((val250>>alu272)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val237>>alu302)))))),(((half)(((signed char)((val281>>alu280)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val265>>alu309)))))),(((half)(((signed char)((val282>>alu280)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val269>>alu309))))))), cast20, make_half4(((*(buf0+44))*alu417),((*(buf0+45))*alu418),((*(buf0+46))*alu417),((*(buf0+47))*alu418)));
    half4 wmma22 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val251>>alu274)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val226>>alu304)))))),(((half)(((signed char)((val252>>alu274)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val230>>alu304)))))),(((half)(((signed char)((val283>>alu282)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val258>>alu311)))))),(((half)(((signed char)((val284>>alu282)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val262>>alu311)))))),(((half)(((signed char)((val253>>alu274)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val234>>alu304)))))),(((half)(((signed char)((val254>>alu274)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val238>>alu304)))))),(((half)(((signed char)((val285>>alu282)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val266>>alu311)))))),(((half)(((signed char)((val286>>alu282)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val270>>alu311))))))), cast20, make_half4(((*(buf0+60))*alu417),((*(buf0+61))*alu418),((*(buf0+62))*alu417),((*(buf0+63))*alu418)));
    half4 wmma23 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val333>>alu267)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val336>>alu320)))))),(((half)(((signed char)((val161>>alu267)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val305>>alu320)))))),(((half)(((signed char)((val334>>alu275)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val337>>alu321)))))),(((half)(((signed char)((val176>>alu275)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val320>>alu321)))))),(((half)(((signed char)((val162>>alu267)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val309>>alu320)))))),(((half)(((signed char)((val163>>alu267)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val313>>alu320)))))),(((half)(((signed char)((val177>>alu275)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val324>>alu321)))))),(((half)(((signed char)((val178>>alu275)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val328>>alu321))))))), cast20, make_half4(((*(buf0+4))*alu417),((*(buf0+5))*alu418),((*(buf0+6))*alu417),((*(buf0+7))*alu418)));
    half4 wmma24 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val338>>alu285)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val385>>alu313)))))),(((half)(((signed char)((val339>>alu285)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val389>>alu313)))))),(((half)(((signed char)((val149>>alu261)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val287>>alu292)))))),(((half)(((signed char)((val150>>alu261)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val291>>alu292)))))),(((half)(((signed char)((val340>>alu285)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val393>>alu313)))))),(((half)(((signed char)((val341>>alu285)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val397>>alu313)))))),(((half)(((signed char)((val151>>alu261)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val295>>alu292)))))),(((half)(((signed char)((val152>>alu261)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val299>>alu292))))))), cast20, make_half4(((*(buf0+16))*alu417),((*(buf0+17))*alu418),((*(buf0+18))*alu417),((*(buf0+19))*alu418)));
    half4 wmma25 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val342>>alu287)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val386>>alu315)))))),(((half)(((signed char)((val343>>alu287)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val390>>alu315)))))),(((half)(((signed char)((val153>>alu263)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val288>>alu294)))))),(((half)(((signed char)((val154>>alu263)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val292>>alu294)))))),(((half)(((signed char)((val344>>alu287)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val394>>alu315)))))),(((half)(((signed char)((val345>>alu287)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val398>>alu315)))))),(((half)(((signed char)((val155>>alu263)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val296>>alu294)))))),(((half)(((signed char)((val156>>alu263)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val300>>alu294))))))), cast20, make_half4(((*(buf0+32))*alu417),((*(buf0+33))*alu418),((*(buf0+34))*alu417),((*(buf0+35))*alu418)));
    half4 wmma26 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val346>>alu289)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val387>>alu317)))))),(((half)(((signed char)((val347>>alu289)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val391>>alu317)))))),(((half)(((signed char)((val157>>alu265)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val289>>alu296)))))),(((half)(((signed char)((val158>>alu265)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val293>>alu296)))))),(((half)(((signed char)((val348>>alu289)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val395>>alu317)))))),(((half)(((signed char)((val349>>alu289)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val399>>alu317)))))),(((half)(((signed char)((val159>>alu265)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val297>>alu296)))))),(((half)(((signed char)((val160>>alu265)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val301>>alu296))))))), cast20, make_half4(((*(buf0+48))*alu417),((*(buf0+49))*alu418),((*(buf0+50))*alu417),((*(buf0+51))*alu418)));
    half4 wmma27 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val369>>alu284)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val353>>alu312)))))),(((half)(((signed char)((val370>>alu284)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val357>>alu312)))))),(((half)(((signed char)((val207>>alu260)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val191>>alu291)))))),(((half)(((signed char)((val208>>alu260)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val195>>alu291)))))),(((half)(((signed char)((val371>>alu284)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val361>>alu312)))))),(((half)(((signed char)((val372>>alu284)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val365>>alu312)))))),(((half)(((signed char)((val209>>alu260)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val199>>alu291)))))),(((half)(((signed char)((val210>>alu260)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val203>>alu291))))))), cast20, make_half4(((*(buf0+8))*alu417),((*(buf0+9))*alu418),((*(buf0+10))*alu417),((*(buf0+11))*alu418)));
    half4 wmma28 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val373>>alu286)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val354>>alu314)))))),(((half)(((signed char)((val374>>alu286)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val358>>alu314)))))),(((half)(((signed char)((val211>>alu262)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val192>>alu293)))))),(((half)(((signed char)((val212>>alu262)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val196>>alu293)))))),(((half)(((signed char)((val375>>alu286)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val362>>alu314)))))),(((half)(((signed char)((val376>>alu286)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val366>>alu314)))))),(((half)(((signed char)((val213>>alu262)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val200>>alu293)))))),(((half)(((signed char)((val214>>alu262)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val204>>alu293))))))), cast20, make_half4(((*(buf0+24))*alu417),((*(buf0+25))*alu418),((*(buf0+26))*alu417),((*(buf0+27))*alu418)));
    half4 wmma29 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val377>>alu288)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val355>>alu316)))))),(((half)(((signed char)((val378>>alu288)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val359>>alu316)))))),(((half)(((signed char)((val215>>alu264)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val193>>alu295)))))),(((half)(((signed char)((val216>>alu264)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val197>>alu295)))))),(((half)(((signed char)((val379>>alu288)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val363>>alu316)))))),(((half)(((signed char)((val380>>alu288)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val367>>alu316)))))),(((half)(((signed char)((val217>>alu264)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val201>>alu295)))))),(((half)(((signed char)((val218>>alu264)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val205>>alu295))))))), cast20, make_half4(((*(buf0+40))*alu417),((*(buf0+41))*alu418),((*(buf0+42))*alu417),((*(buf0+43))*alu418)));
    half4 wmma30 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val381>>alu290)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val356>>alu318)))))),(((half)(((signed char)((val382>>alu290)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val360>>alu318)))))),(((half)(((signed char)((val219>>alu266)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val194>>alu297)))))),(((half)(((signed char)((val220>>alu266)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val198>>alu297)))))),(((half)(((signed char)((val383>>alu290)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val364>>alu318)))))),(((half)(((signed char)((val384>>alu290)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val368>>alu318)))))),(((half)(((signed char)((val221>>alu266)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val202>>alu297)))))),(((half)(((signed char)((val222>>alu266)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val206>>alu297))))))), cast20, make_half4(((*(buf0+56))*alu417),((*(buf0+57))*alu418),((*(buf0+58))*alu417),((*(buf0+59))*alu418)));
    half4 wmma31 = __WMMA_8_16_16_half_half(make_half8((((half)(((signed char)((val400>>alu283)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val401>>alu322)))))),(((half)(((signed char)((val350>>alu283)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val388>>alu322)))))),(((half)(((signed char)((val332>>alu259)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val335>>alu319)))))),(((half)(((signed char)((val146>>alu259)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val290>>alu319)))))),(((half)(((signed char)((val351>>alu283)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val392>>alu322)))))),(((half)(((signed char)((val352>>alu283)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val396>>alu322)))))),(((half)(((signed char)((val147>>alu259)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val294>>alu319)))))),(((half)(((signed char)((val148>>alu259)))))*tg_bitcast<half>((unsigned short)(((unsigned short)((val298>>alu319))))))), cast20, make_half4(((*(buf0+0))*alu417),((*(buf0+1))*alu418),((*(buf0+2))*alu417),((*(buf0+3))*alu418)));
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
  float alu486 = ((*(buf2+0))+buf12);
  buf13 = __shfl_xor_sync(0xffffffffu, alu486, 8);
  float alu488 = (alu486+buf13);
  buf14 = __shfl_xor_sync(0xffffffffu, alu488, 4);
  buf15 = __shfl_xor_sync(0xffffffffu, (*(buf2+1)), 16);
  float alu491 = ((*(buf2+1))+buf15);
  buf16 = __shfl_xor_sync(0xffffffffu, alu491, 8);
  float alu493 = (alu491+buf16);
  buf17 = __shfl_xor_sync(0xffffffffu, alu493, 4);
  int alu495 = ((alu10*172032)+(alu6*344064));
  int alu496 = (alu41+alu39);
  bool alu497 = (((lidx1<<2)+alu38)<6);
  int alu498 = (alu497?alu496:0);
  int alu499 = (gidx2<<10);
  int alu500 = (gidx1<<8);
  int alu501 = (alu495+(alu498*14336)+alu499+alu500+alu42);
  bool alu502 = (alu496<11);
  int alu503 = (alu502?(alu496+1):0);
  int alu504 = (alu495+(alu503*14336)+alu499+alu500+alu42);
  int alu505 = ((alu10*1344)+(alu6*2688));
  int alu506 = (gidx2<<3);
  int alu507 = (gidx1<<1);
  int alu508 = (alu505+(alu498*112)+alu506+alu507);
  int alu509 = (alu505+(alu503*112)+alu506+alu507);
  if (alu497) {
    *(data0_1032192+(alu501+8)) = (*(buf0+2));
  }
  if (alu497) {
    *(data0_1032192+(alu501+16)) = (*(buf0+4));
  }
  if (alu497) {
    *(data0_1032192+(alu501+24)) = (*(buf0+6));
  }
  if (alu497) {
    *(data0_1032192+(alu501+32)) = (*(buf0+8));
  }
  if (alu497) {
    *(data0_1032192+(alu501+40)) = (*(buf0+10));
  }
  if (alu497) {
    *(data0_1032192+(alu501+48)) = (*(buf0+12));
  }
  if (alu497) {
    *(data0_1032192+(alu501+56)) = (*(buf0+14));
  }
  if (alu497) {
    *(data0_1032192+(alu501+64)) = (*(buf0+16));
  }
  if (alu497) {
    *(data0_1032192+(alu501+72)) = (*(buf0+18));
  }
  if (alu497) {
    *(data0_1032192+(alu501+80)) = (*(buf0+20));
  }
  if (alu497) {
    *(data0_1032192+(alu501+88)) = (*(buf0+22));
  }
  if (alu497) {
    *(data0_1032192+(alu501+96)) = (*(buf0+24));
  }
  if (alu497) {
    *(data0_1032192+(alu501+104)) = (*(buf0+26));
  }
  if (alu497) {
    *(data0_1032192+(alu501+112)) = (*(buf0+28));
  }
  if (alu497) {
    *(data0_1032192+(alu501+120)) = (*(buf0+30));
  }
  if (alu497) {
    *(data0_1032192+(alu501+128)) = (*(buf0+32));
  }
  if (alu497) {
    *(data0_1032192+(alu501+136)) = (*(buf0+34));
  }
  if (alu497) {
    *(data0_1032192+(alu501+144)) = (*(buf0+36));
  }
  if (alu497) {
    *(data0_1032192+(alu501+152)) = (*(buf0+38));
  }
  if (alu497) {
    *(data0_1032192+(alu501+160)) = (*(buf0+40));
  }
  if (alu497) {
    *(data0_1032192+(alu501+168)) = (*(buf0+42));
  }
  if (alu497) {
    *(data0_1032192+(alu501+176)) = (*(buf0+44));
  }
  if (alu497) {
    *(data0_1032192+(alu501+184)) = (*(buf0+46));
  }
  if (alu497) {
    *(data0_1032192+(alu501+192)) = (*(buf0+48));
  }
  if (alu497) {
    *(data0_1032192+(alu501+200)) = (*(buf0+50));
  }
  if (alu497) {
    *(data0_1032192+(alu501+208)) = (*(buf0+52));
  }
  if (alu497) {
    *(data0_1032192+(alu501+216)) = (*(buf0+54));
  }
  if (alu497) {
    *(data0_1032192+(alu501+224)) = (*(buf0+56));
  }
  if (alu497) {
    *(data0_1032192+(alu501+232)) = (*(buf0+58));
  }
  if (alu497) {
    *(data0_1032192+(alu501+240)) = (*(buf0+60));
  }
  if (alu497) {
    *(data0_1032192+(alu501+248)) = (*(buf0+62));
  }
  if (alu502) {
    *(data0_1032192+(alu504+8)) = (*(buf0+3));
  }
  if (alu502) {
    *(data0_1032192+(alu504+16)) = (*(buf0+5));
  }
  if (alu502) {
    *(data0_1032192+(alu504+24)) = (*(buf0+7));
  }
  if (alu502) {
    *(data0_1032192+(alu504+32)) = (*(buf0+9));
  }
  if (alu502) {
    *(data0_1032192+(alu504+40)) = (*(buf0+11));
  }
  if (alu502) {
    *(data0_1032192+(alu504+48)) = (*(buf0+13));
  }
  if (alu502) {
    *(data0_1032192+(alu504+56)) = (*(buf0+15));
  }
  if (alu502) {
    *(data0_1032192+(alu504+64)) = (*(buf0+17));
  }
  if (alu502) {
    *(data0_1032192+(alu504+72)) = (*(buf0+19));
  }
  if (alu502) {
    *(data0_1032192+(alu504+80)) = (*(buf0+21));
  }
  if (alu502) {
    *(data0_1032192+(alu504+88)) = (*(buf0+23));
  }
  if (alu502) {
    *(data0_1032192+(alu504+96)) = (*(buf0+25));
  }
  if (alu502) {
    *(data0_1032192+(alu504+104)) = (*(buf0+27));
  }
  if (alu502) {
    *(data0_1032192+(alu504+112)) = (*(buf0+29));
  }
  if (alu502) {
    *(data0_1032192+(alu504+120)) = (*(buf0+31));
  }
  if (alu502) {
    *(data0_1032192+(alu504+128)) = (*(buf0+33));
  }
  if (alu502) {
    *(data0_1032192+(alu504+136)) = (*(buf0+35));
  }
  if (alu502) {
    *(data0_1032192+(alu504+144)) = (*(buf0+37));
  }
  if (alu502) {
    *(data0_1032192+(alu504+152)) = (*(buf0+39));
  }
  if (alu502) {
    *(data0_1032192+(alu504+160)) = (*(buf0+41));
  }
  if (alu502) {
    *(data0_1032192+(alu504+168)) = (*(buf0+43));
  }
  if (alu502) {
    *(data0_1032192+(alu504+176)) = (*(buf0+45));
  }
  if (alu502) {
    *(data0_1032192+(alu504+184)) = (*(buf0+47));
  }
  if (alu502) {
    *(data0_1032192+(alu504+192)) = (*(buf0+49));
  }
  if (alu502) {
    *(data0_1032192+(alu504+200)) = (*(buf0+51));
  }
  if (alu502) {
    *(data0_1032192+(alu504+208)) = (*(buf0+53));
  }
  if (alu502) {
    *(data0_1032192+(alu504+216)) = (*(buf0+55));
  }
  if (alu502) {
    *(data0_1032192+(alu504+224)) = (*(buf0+57));
  }
  if (alu502) {
    *(data0_1032192+(alu504+232)) = (*(buf0+59));
  }
  if (alu502) {
    *(data0_1032192+(alu504+240)) = (*(buf0+61));
  }
  if (alu502) {
    *(data0_1032192+(alu504+248)) = (*(buf0+63));
  }
  if (alu497) {
    *(data0_1032192+alu501) = (*(buf0+0));
  }
  if (alu502) {
    *(data0_1032192+alu504) = (*(buf0+1));
  }
  if (alu497) {
    *(data1_8064+(alu508+1)) = (alu488+buf14);
  }
  if (alu502) {
    *(data1_8064+(alu509+1)) = (alu493+buf17);
  }
  if (alu497) {
    *(data1_8064+alu508) = (*(buf1+0));
  }
  if (alu502) {
    *(data1_8064+alu509) = (*(buf1+1));
  }
}