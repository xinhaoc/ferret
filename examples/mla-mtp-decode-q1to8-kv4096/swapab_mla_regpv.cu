// QK swap-AB (verified) + softmax (verified) + register-based PV (no MMA)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
constexpr int NH=64, DK=576, DV=512, TS=128, BK=64, MK=16, TB=128;
constexpr int TILE_B=128*BK*2;
inline void ccu(CUresult e){if(e!=CUDA_SUCCESS){const char*m;cuGetErrorString(e,&m);fprintf(stderr,"CU:%s\n",m);exit(1);}}
inline void ccd(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(e));exit(1);}}
__device__ inline uint32_t el(){uint32_t p=0;asm volatile("{\n\t.reg .pred %%px;\n\telect.sync _|%%px,0xFFFFFFFF;\n\t@%%px mov.s32 %0,1;\n\t}":"+r"(p));return p;}
__device__ inline constexpr uint64_t de(uint64_t x){return(x&0x3FFFFULL)>>4;}
__device__ inline uint64_t md(int a){return de(a)|(de(8ULL*128)<<32)|(1ULL<<46)|(2ULL<<61);}
__device__ float bmx(float v,float*s,int t){for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_xor_sync(0xFFFFFFFF,v,o));if(t%32==0)s[t/32]=v;__syncthreads();if(t<4){v=s[t];for(int o=2;o>0;o>>=1)v=fmaxf(v,__shfl_xor_sync(0xF,v,o));}if(t==0)s[0]=v;__syncthreads();return s[0];}
__device__ float bsm(float v,float*s,int t){for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xFFFFFFFF,v,o);if(t%32==0)s[t/32]=v;__syncthreads();if(t<4){v=s[t];for(int o=2;o>0;o>>=1)v+=__shfl_xor_sync(0xF,v,o);}if(t==0)s[0]=v;__syncthreads();return s[0];}

__global__ __launch_bounds__(TB)
void regpv_test(const __grid_constant__ CUtensorMap Kt,const __grid_constant__ CUtensorMap Qt,
  const nv_bfloat16*V,nv_bfloat16*O_out,float sc,int kv){
  const int tid=threadIdx.x,wid=tid/32;
  extern __shared__ __align__(1024) char sb[];
  const int sm=__cvta_generic_to_shared(sb);
  __shared__ uint64_t mb[2];__shared__ int ta[1];__shared__ float rs[4];
  const int tb=__cvta_generic_to_shared(&mb[0]),mb2=__cvta_generic_to_shared(&mb[1]);
  if(wid==0&&el()){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(tb),"r"(1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb2),"r"(1));
    asm volatile("fence.mbarrier_init.release.cluster;");}
  else if(wid==1){int as=__cvta_generic_to_shared(ta);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::"r"(as),"r"(NH));}
  __syncthreads();
  const int taddr=ta[0];
  constexpr uint32_t id=(1U<<4)|(1U<<7)|(1U<<10)|((uint32_t)(NH>>3)<<17)|((uint32_t)(TS>>4)<<24);
  int ph=0;
  for(int ki=0;ki<DK/BK;ki++){
    int ks=sm,qs=sm+TILE_B;
    if(wid==0&&el()){asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(tb),"r"(TILE_B+NH*BK*2):"memory");
      asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4}],[%5];"::"r"(ks),"l"(&Kt),"r"(0),"r"(0),"r"(ki),"r"(tb):"memory");
      asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4}],[%5];"::"r"(qs),"l"(&Qt),"r"(0),"r"(0),"r"(ki),"r"(tb):"memory");}
    asm volatile("{\n\t.reg .pred P;\n\tW: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P,[%0],%1,0x989680;\n\t@P bra D;\n\tbra W;\n\tD:\n\t}"::"r"(tb),"r"(ph));
    ph^=1; asm volatile("tcgen05.fence::after_thread_sync;");
    if(wid==1&&el()){for(int k2=0;k2<BK/MK;k2++){
      asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p,%4,0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n\t}"
        ::"r"(taddr),"l"(md(ks+k2*32)),"l"(md(qs+k2*32)),"r"(id),"r"((ki==0&&k2==0)?0:1));}}
    __syncthreads();
  }
  if(wid==1&&el())asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(mb2):"memory");
  __syncthreads();
  asm volatile("{\n\t.reg .pred P;\n\tW2: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P,[%0],%1,0x989680;\n\t@P bra D2;\n\tbra W2;\n\tD2:\n\t}"::"r"(mb2),"r"(0));
  asm volatile("tcgen05.fence::after_thread_sync;");
  // Softmax
  float sv[NH];
  for(int h=0;h<NH;h+=16){float t[16];int a=taddr+(tid<<16)+h;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},[%16];"
      :"=f"(t[0]),"=f"(t[1]),"=f"(t[2]),"=f"(t[3]),"=f"(t[4]),"=f"(t[5]),"=f"(t[6]),"=f"(t[7]),
       "=f"(t[8]),"=f"(t[9]),"=f"(t[10]),"=f"(t[11]),"=f"(t[12]),"=f"(t[13]),"=f"(t[14]),"=f"(t[15])
      :"r"(a));asm volatile("tcgen05.wait::ld.sync.aligned;");
    for(int i=0;i<16;i++)sv[h+i]=t[i]*sc;}
  bool vld=(tid<kv);
  for(int h=0;h<NH;h++){float v=vld?sv[h]:-1e30f;float mx=bmx(v,rs,tid);
    float e=vld?__expf(v-mx):0.0f;float s=bsm(e,rs,tid);sv[h]=(s>0.0f)?e/s:0.0f;}
  // Dealloc TMEM
  __syncthreads();
  if(wid==0)asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(taddr),"r"(NH));
  __syncthreads();
  // PV via registers: each thread tid is kv_pos tid, has P^T[tid, 0..NH-1]
  // Store P^T to SMEM as float: P_smem[tid * NH + h]
  __shared__ float P_sm[TS * NH];
  for(int h=0;h<NH;h++) P_sm[tid*NH+h]=sv[h];
  __syncthreads();
  // Each thread computes O[h, d] for one head h (threads 0..63)
  if(tid<NH){
    int h=tid;
    for(int d=0;d<DV;d++){
      float acc=0;
      for(int s=0;s<kv;s++) acc+=P_sm[s*NH+h]*__bfloat162float(V[s*DV+d]);
      O_out[h*DV+d]=__float2bfloat16(acc);
    }
  }
}

int main(){
  const int kv=TS;const float sc=1.0f/sqrtf((float)DK);
  printf("RegPV test: H=%d DK=%d DV=%d kv=%d\n",NH,DK,DV,kv);
  size_t sq=NH*DK,skv=kv*DK,sv=kv*DV,so=NH*DV;
  nv_bfloat16*hQ=new nv_bfloat16[sq],*hK=new nv_bfloat16[skv],*hV=new nv_bfloat16[sv];float*hR=new float[so];
  srand(42);
  for(size_t i=0;i<sq;i++)hQ[i]=__float2bfloat16((rand()%201-100)/200.0f);
  for(size_t i=0;i<skv;i++)hK[i]=__float2bfloat16((rand()%201-100)/200.0f);
  for(size_t i=0;i<sv;i++)hV[i]=__float2bfloat16((rand()%201-100)/200.0f);
  for(int h=0;h<NH;h++){float scores[TS],mx=-1e30f;
    for(int s=0;s<kv;s++){float d=0;for(int k=0;k<DK;k++)d+=__bfloat162float(hQ[h*DK+k])*__bfloat162float(hK[s*DK+k]);scores[s]=d*sc;mx=fmaxf(mx,scores[s]);}
    float se=0;for(int s=0;s<kv;s++){scores[s]=expf(scores[s]-mx);se+=scores[s];}
    for(int s=0;s<kv;s++)scores[s]/=se;
    for(int d=0;d<DV;d++){float a=0;for(int s=0;s<kv;s++)a+=scores[s]*__bfloat162float(hV[s*DV+d]);hR[h*DV+d]=a;}}
  nv_bfloat16*dQ,*dK,*dV2,*dO;
  ccd(cudaMalloc(&dQ,sq*2));ccd(cudaMalloc(&dK,skv*2));ccd(cudaMalloc(&dV2,sv*2));ccd(cudaMalloc(&dO,so*2));
  ccd(cudaMemcpy(dQ,hQ,sq*2,cudaMemcpyHostToDevice));ccd(cudaMemcpy(dK,hK,skv*2,cudaMemcpyHostToDevice));
  ccd(cudaMemcpy(dV2,hV,sv*2,cudaMemcpyHostToDevice));ccd(cudaMemset(dO,0,so*2));
  CUtensorMap Kt,Qt;
  auto itm=[](CUtensorMap*t,const nv_bfloat16*p,uint64_t r,uint32_t br,uint64_t c){
    uint64_t gd[3]={64,r,c/64};uint64_t gs[2]={c*2,128};uint32_t bd[3]={64,br,1};uint32_t es[3]={1,1,1};
    ccu(cuTensorMapEncodeTiled(t,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)p,gd,gs,bd,es,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));};
  itm(&Kt,dK,kv,TS,DK);itm(&Qt,dQ,NH,NH,DK);
  int smem=2*TILE_B+TS*NH*4;
  printf("SMEM: %d\n",smem);
  ccd(cudaFuncSetAttribute(regpv_test,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  regpv_test<<<1,TB,smem>>>(Kt,Qt,dV2,dO,sc,kv);
  ccd(cudaDeviceSynchronize());
  nv_bfloat16*hO=new nv_bfloat16[so];ccd(cudaMemcpy(hO,dO,so*2,cudaMemcpyDeviceToHost));
  float me=0;int ec=0;
  for(int h=0;h<NH;h++)for(int d=0;d<DV;d++){float g=__bfloat162float(hO[h*DV+d]),r=hR[h*DV+d];
    float e=fabsf(g-r)/fmaxf(fmaxf(fabsf(r),fabsf(g)),1e-6f);if(e>me)me=e;if(e>5e-2f)ec++;}
  printf("max_err=%.6f errs=%d/%d\n%s\n",me,ec,(int)so,ec==0?"PASS":"FAIL");
  delete[]hQ;delete[]hK;delete[]hV;delete[]hO;delete[]hR;cudaFree(dQ);cudaFree(dK);cudaFree(dV2);cudaFree(dO);
}
