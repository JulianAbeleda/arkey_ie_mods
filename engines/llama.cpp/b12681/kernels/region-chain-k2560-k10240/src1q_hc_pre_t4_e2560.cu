#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) src1q_hc_pre_t4_e2560(float* data0_10240, unsigned char* data1_23040, unsigned char* data2_23040, unsigned char* data3_11520, float* data4_40960, float* data5_40960) {
  int gidx0 = blockIdx.x; /* 4 */
  int gidx1 = blockIdx.y; /* 10 */
  int lidx0 = threadIdx.x; /* 16 */
  int lidx1 = threadIdx.y; /* 4 */
  int lidx2 = threadIdx.z; /* 2 */
  int alu0 = (lidx0<<2);
  int alu1 = (lidx1<<6);
  int alu2 = (gidx1<<8);
  int alu3 = (alu0+alu1+alu2+(gidx0*20480)+(lidx2*10240));
  int alu4 = (alu3+1);
  bool alu5 = (gidx0<2);
  float val0 = (alu5?*(data4_40960+alu4):0.0f);
  int alu6 = (alu3+2);
  float val1 = (alu5?*(data4_40960+alu6):0.0f);
  int alu7 = (alu3+3);
  float val2 = (alu5?*(data4_40960+alu7):0.0f);
  int alu8 = (alu3+2560);
  float val3 = (alu5?*(data4_40960+alu8):0.0f);
  int alu9 = (alu3+2561);
  float val4 = (alu5?*(data4_40960+alu9):0.0f);
  int alu10 = (alu3+2562);
  float val5 = (alu5?*(data4_40960+alu10):0.0f);
  int alu11 = (alu3+2563);
  float val6 = (alu5?*(data4_40960+alu11):0.0f);
  int alu12 = (alu3+5120);
  float val7 = (alu5?*(data4_40960+alu12):0.0f);
  int alu13 = (alu3+5121);
  float val8 = (alu5?*(data4_40960+alu13):0.0f);
  int alu14 = (alu3+5122);
  float val9 = (alu5?*(data4_40960+alu14):0.0f);
  int alu15 = (alu3+5123);
  float val10 = (alu5?*(data4_40960+alu15):0.0f);
  int alu16 = (alu3+7680);
  float val11 = (alu5?*(data4_40960+alu16):0.0f);
  int alu17 = (alu3+7681);
  float val12 = (alu5?*(data4_40960+alu17):0.0f);
  int alu18 = (alu3+7682);
  float val13 = (alu5?*(data4_40960+alu18):0.0f);
  int alu19 = (alu3+7683);
  float val14 = (alu5?*(data4_40960+alu19):0.0f);
  float val15 = (alu5?*(data4_40960+alu3):0.0f);
  float val16 = (alu5?*(data5_40960+alu4):0.0f);
  float val17 = (alu5?*(data5_40960+alu6):0.0f);
  float val18 = (alu5?*(data5_40960+alu7):0.0f);
  float val19 = (alu5?*(data5_40960+alu8):0.0f);
  float val20 = (alu5?*(data5_40960+alu9):0.0f);
  float val21 = (alu5?*(data5_40960+alu10):0.0f);
  float val22 = (alu5?*(data5_40960+alu11):0.0f);
  float val23 = (alu5?*(data5_40960+alu12):0.0f);
  float val24 = (alu5?*(data5_40960+alu13):0.0f);
  float val25 = (alu5?*(data5_40960+alu14):0.0f);
  float val26 = (alu5?*(data5_40960+alu15):0.0f);
  float val27 = (alu5?*(data5_40960+alu16):0.0f);
  float val28 = (alu5?*(data5_40960+alu17):0.0f);
  float val29 = (alu5?*(data5_40960+alu18):0.0f);
  float val30 = (alu5?*(data5_40960+alu19):0.0f);
  float val31 = (alu5?*(data5_40960+alu3):0.0f);
  float buf0;
  float buf1;
  float buf2;
  float buf3;
  float buf4;
  float buf5;
  int buf6;
  int buf7;
  int buf8;
  int buf9;
  int buf10;
  int buf11;
  float alu20 = (fmaf(val11,(1/(1.0f+exp2((val27*-1.4426950408889634f)))),fmaf(val7,(1/(1.0f+exp2((val23*-1.4426950408889634f)))),fmaf(val3,(1/(1.0f+exp2((val19*-1.4426950408889634f)))),fmaf(val15,(1/(1.0f+exp2((val31*-1.4426950408889634f)))),0.0f))))*__int_as_float(0x3e800000));
  float alu21 = (alu5?alu20:0.0f);
  float alu22 = (fmaf(val12,(1/(1.0f+exp2((val28*-1.4426950408889634f)))),fmaf(val8,(1/(1.0f+exp2((val24*-1.4426950408889634f)))),fmaf(val4,(1/(1.0f+exp2((val20*-1.4426950408889634f)))),fmaf(val0,(1/(1.0f+exp2((val16*-1.4426950408889634f)))),0.0f))))*__int_as_float(0x3e800000));
  float alu23 = (alu5?alu22:0.0f);
  float alu24 = (fmaf(val13,(1/(1.0f+exp2((val29*-1.4426950408889634f)))),fmaf(val9,(1/(1.0f+exp2((val25*-1.4426950408889634f)))),fmaf(val5,(1/(1.0f+exp2((val21*-1.4426950408889634f)))),fmaf(val1,(1/(1.0f+exp2((val17*-1.4426950408889634f)))),0.0f))))*__int_as_float(0x3e800000));
  float alu25 = (alu5?alu24:0.0f);
  float alu26 = (fmaf(val14,(1/(1.0f+exp2((val30*-1.4426950408889634f)))),fmaf(val10,(1/(1.0f+exp2((val26*-1.4426950408889634f)))),fmaf(val6,(1/(1.0f+exp2((val22*-1.4426950408889634f)))),fmaf(val2,(1/(1.0f+exp2((val18*-1.4426950408889634f)))),0.0f))))*__int_as_float(0x3e800000));
  float alu27 = (alu5?alu26:0.0f);
  float cast0 = tg_bitcast<float>((unsigned int)((tg_bitcast<unsigned int>((float)(alu21))&2147483647u)));
  float cast1 = tg_bitcast<float>((unsigned int)((tg_bitcast<unsigned int>((float)(alu23))&2147483647u)));
  float cast2 = tg_bitcast<float>((unsigned int)((tg_bitcast<unsigned int>((float)(alu25))&2147483647u)));
  float cast3 = tg_bitcast<float>((unsigned int)((tg_bitcast<unsigned int>((float)(alu27))&2147483647u)));
  float alu28 = ((cast0!=cast0)?((float)(-INFINITY)):cast0);
  float alu29 = ((cast1!=cast1)?((float)(-INFINITY)):cast1);
  float alu30 = ((alu28<alu29)?alu29:alu28);
  float alu31 = ((cast2!=cast2)?((float)(-INFINITY)):cast2);
  float alu32 = ((alu30<alu31)?alu31:alu30);
  float alu33 = ((cast3!=cast3)?((float)(-INFINITY)):cast3);
  float alu34 = ((alu32<alu33)?alu33:alu32);
  buf0 = __shfl_xor_sync(0xffffffffu, alu34, 4);
  float alu36 = ((alu34<buf0)?buf0:alu34);
  buf1 = __shfl_xor_sync(0xffffffffu, alu36, 2);
  float alu38 = ((alu36<buf1)?buf1:alu36);
  buf2 = __shfl_xor_sync(0xffffffffu, alu38, 1);
  float alu40 = ((alu38<buf2)?buf2:alu38);
  float alu41 = ((alu40!=((float)(-INFINITY)))?alu40:((float)(NAN)));
  float alu42 = fmaf((1.0f/alu41),127.0f,0.0f);
  signed char cast4 = ((signed char)(((int)(roundf((alu21*alu42))))));
  signed char cast5 = ((signed char)(((int)(roundf((alu23*alu42))))));
  signed char cast6 = ((signed char)(((int)(roundf((alu25*alu42))))));
  signed char cast7 = ((signed char)(((int)(roundf((alu27*alu42))))));
  bool alu43 = (alu41!=0.0f);
  int alu44 = (alu43?(((int)(cast4))+((int)(cast5))+((int)(cast6))+((int)(cast7))):0);
  buf6 = __shfl_xor_sync(0xffffffffu, alu44, 4);
  int alu46 = (alu44+buf6);
  buf7 = __shfl_xor_sync(0xffffffffu, alu46, 2);
  int alu48 = (alu46+buf7);
  buf8 = __shfl_xor_sync(0xffffffffu, alu48, 1);
  float cast8 = tg_bitcast<float>((unsigned int)((tg_bitcast<unsigned int>((float)(alu20))&2147483647u)));
  float cast9 = tg_bitcast<float>((unsigned int)((tg_bitcast<unsigned int>((float)(alu22))&2147483647u)));
  float cast10 = tg_bitcast<float>((unsigned int)((tg_bitcast<unsigned int>((float)(alu24))&2147483647u)));
  float cast11 = tg_bitcast<float>((unsigned int)((tg_bitcast<unsigned int>((float)(alu26))&2147483647u)));
  float alu50 = ((cast8!=cast8)?((float)(-INFINITY)):cast8);
  float alu51 = ((cast9!=cast9)?((float)(-INFINITY)):cast9);
  float alu52 = ((alu50<alu51)?alu51:alu50);
  float alu53 = ((cast10!=cast10)?((float)(-INFINITY)):cast10);
  float alu54 = ((alu52<alu53)?alu53:alu52);
  float alu55 = ((cast11!=cast11)?((float)(-INFINITY)):cast11);
  float alu56 = ((alu54<alu55)?alu55:alu54);
  buf3 = __shfl_xor_sync(0xffffffffu, alu56, 4);
  float alu58 = ((alu56<buf3)?buf3:alu56);
  buf4 = __shfl_xor_sync(0xffffffffu, alu58, 2);
  float alu60 = ((alu58<buf4)?buf4:alu58);
  buf5 = __shfl_xor_sync(0xffffffffu, alu60, 1);
  float alu62 = ((alu60<buf5)?buf5:alu60);
  float alu63 = ((alu62!=((float)(-INFINITY)))?alu62:((float)(NAN)));
  float alu64 = fmaf(alu63,0.007874015718698502f,0.0f);
  float alu65 = (1.0f/alu64);
  signed char cast12 = ((signed char)(((int)(roundf((alu20*alu65))))));
  signed char cast13 = ((signed char)(((int)(roundf((alu22*alu65))))));
  signed char cast14 = ((signed char)(((int)(roundf((alu24*alu65))))));
  signed char cast15 = ((signed char)(((int)(roundf((alu26*alu65))))));
  bool alu66 = (alu63!=0.0f);
  int alu67 = (alu66?(((int)(cast12))+((int)(cast13))+((int)(cast14))+((int)(cast15))):0);
  buf9 = __shfl_xor_sync(0xffffffffu, alu67, 4);
  int alu69 = (alu67+buf9);
  buf10 = __shfl_xor_sync(0xffffffffu, alu69, 2);
  int alu71 = (alu69+buf10);
  buf11 = __shfl_xor_sync(0xffffffffu, alu71, 1);
  int alu73 = (gidx1*2304);
  int alu74 = ((gidx0<<7)+((lidx0>>2)<<2)+(lidx2<<6)+alu73+(lidx1<<9)+((lidx0&3)<<4));
  int alu75 = (alu74+1);
  int alu76 = (alu74+2);
  int alu77 = (alu74+3);
  int alu78 = (lidx0>>3);
  int alu79 = ((gidx0<<6)+alu73+(lidx1<<4)+(alu78<<3)+(lidx2<<2));
  int alu80 = (alu79+2048);
  int alu81 = (alu79+2049);
  int alu82 = (alu79+2050);
  int alu83 = (alu79+2051);
  int alu84 = ((gidx1*288)+(lidx1*72)+(alu78*36)+(gidx0*5760)+(lidx2*2880));
  int alu85 = (lidx0&7);
  int alu86 = (alu84+(alu85<<2));
  int alu87 = (alu2+alu0+alu1+(gidx0*5120)+(lidx2*2560));
  float alu88 = (1.0f/alu42);
  unsigned int cast16 = tg_bitcast<unsigned int>((float)(alu88));
  unsigned int alu89 = (cast16>>24u);
  bool alu90 = (alu85<1);
  unsigned int alu91 = (alu90?(cast16>>0u):0u);
  unsigned int alu92 = (alu90?0u:alu89);
  unsigned int alu93 = (alu90?(cast16>>8u):0u);
  unsigned int alu94 = (alu90?(cast16>>16u):0u);
  unsigned int alu95 = (((unsigned int)(tg_bitcast<unsigned short>((half)(((half)(alu88))))))|(((unsigned int)(tg_bitcast<unsigned short>((half)(((half)((((float)((alu48+buf8)))*alu88)))))))<<16u));
  unsigned int alu96 = (alu95>>24u);
  unsigned int alu97 = (alu90?(alu95>>0u):0u);
  unsigned int alu98 = (alu90?0u:alu96);
  unsigned int alu99 = (alu90?(alu95>>8u):0u);
  unsigned int alu100 = (alu90?(alu95>>16u):0u);
  unsigned int alu101 = (((unsigned int)(tg_bitcast<unsigned short>((half)(((half)(alu64))))))|(((unsigned int)(tg_bitcast<unsigned short>((half)(((half)((alu64*((float)((alu71+buf11))))))))))<<16u));
  unsigned int alu102 = (alu101>>24u);
  unsigned int alu103 = (alu90?(alu101>>0u):0u);
  unsigned int alu104 = (alu90?0u:alu102);
  unsigned int alu105 = (alu90?(alu101>>8u):0u);
  unsigned int alu106 = (alu90?(alu101>>16u):0u);
  signed char alu107 = (alu43?cast4:((signed char)(0)));
  unsigned char cast17 = tg_bitcast<unsigned char>((signed char)(alu107));
  signed char alu108 = (alu43?cast5:((signed char)(0)));
  unsigned char cast18 = tg_bitcast<unsigned char>((signed char)(alu108));
  signed char alu109 = (alu43?cast6:((signed char)(0)));
  unsigned char cast19 = tg_bitcast<unsigned char>((signed char)(alu109));
  signed char alu110 = (alu43?cast7:((signed char)(0)));
  unsigned char cast20 = tg_bitcast<unsigned char>((signed char)(alu110));
  signed char alu111 = (alu66?cast12:((signed char)(0)));
  signed char alu112 = (alu66?cast13:((signed char)(0)));
  signed char alu113 = (alu66?cast14:((signed char)(0)));
  signed char alu114 = (alu66?cast15:((signed char)(0)));
  bool alu115 = (alu90&alu5);
  *(data1_23040+alu75) = cast18;
  *(data1_23040+alu76) = cast19;
  *(data1_23040+alu77) = cast20;
  if (alu90) {
    *(data1_23040+alu80) = ((unsigned char)((alu91+alu92)));
  }
  if (alu90) {
    *(data1_23040+alu81) = ((unsigned char)((alu93+alu92)));
  }
  if (alu90) {
    *(data1_23040+alu82) = ((unsigned char)((alu94+alu92)));
  }
  if (alu90) {
    *(data1_23040+alu83) = ((unsigned char)(alu89));
  }
  *(data1_23040+alu74) = cast17;
  *(data2_23040+alu75) = cast18;
  *(data2_23040+alu76) = cast19;
  *(data2_23040+alu77) = cast20;
  if (alu90) {
    *(data2_23040+alu80) = ((unsigned char)((alu97+alu98)));
  }
  if (alu90) {
    *(data2_23040+alu81) = ((unsigned char)((alu99+alu98)));
  }
  if (alu90) {
    *(data2_23040+alu82) = ((unsigned char)((alu100+alu98)));
  }
  if (alu90) {
    *(data2_23040+alu83) = ((unsigned char)(alu96));
  }
  *(data2_23040+alu74) = cast17;
  if (alu5) {
    *(data3_11520+(alu86+4)) = tg_bitcast<unsigned char>((signed char)(alu111));
  }
  if (alu5) {
    *(data3_11520+(alu86+5)) = tg_bitcast<unsigned char>((signed char)(alu112));
  }
  if (alu5) {
    *(data3_11520+(alu86+6)) = tg_bitcast<unsigned char>((signed char)(alu113));
  }
  if (alu5) {
    *(data3_11520+(alu86+7)) = tg_bitcast<unsigned char>((signed char)(alu114));
  }
  if (alu115) {
    *(data3_11520+(alu84+1)) = ((unsigned char)((alu105+alu104)));
  }
  if (alu115) {
    *(data3_11520+(alu84+2)) = ((unsigned char)((alu106+alu104)));
  }
  if (alu115) {
    *(data3_11520+(alu84+3)) = ((unsigned char)(alu102));
  }
  if (alu115) {
    *(data3_11520+alu84) = ((unsigned char)((alu103+alu104)));
  }
  if (alu5) {
    *(data0_10240+(alu87+1)) = alu22;
  }
  if (alu5) {
    *(data0_10240+(alu87+2)) = alu24;
  }
  if (alu5) {
    *(data0_10240+(alu87+3)) = alu26;
  }
  if (alu5) {
    *(data0_10240+alu87) = alu20;
  }
}