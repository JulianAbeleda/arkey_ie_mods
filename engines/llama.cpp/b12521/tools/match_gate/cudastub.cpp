// cudastub: a CUDA runtime / driver / cuBLAS that does nothing, for a DRY RUN of an unmodified engine build on the CPU.
// The engine binary (llama-server + libggml-cuda.so as built for the GPU) loads these four libraries by SONAME from
// LD_LIBRARY_PATH instead of the real ones, so the GPU is never opened. "Device" memory is untouched anonymous host
// memory (mmap NORESERVE); kernels, cuBLAS calls, memsets and device-to-device copies are no-ops; device-to-host copies
// are done up to STUB_COPY_MAX bytes and host-to-device copies never, so every value read back from the "device" is 0.
// What stays real: the engine's graph build, the scheduler's splits, the graph allocator (addresses relative to each
// buffer), the backend's node loop and the kernels module's matcher. Nothing computed is meaningful.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

static std::mutex g_mu;
static std::map<void *, size_t> g_maps;
static const size_t STUB_COPY_MAX = getenv("STUB_COPY_MAX") ? strtoull(getenv("STUB_COPY_MAX"), nullptr, 10) : (size_t) 8 << 20;
static void * stub_map(size_t n) {
    if (n == 0) n = 4096;
    void * p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    std::lock_guard<std::mutex> l(g_mu); g_maps[p] = n; return p;
}
static void stub_unmap(void * p) {
    std::lock_guard<std::mutex> l(g_mu); auto it = g_maps.find(p);
    if (it != g_maps.end()) { munmap(p, it->second); g_maps.erase(it); }
}
// only device-to-host is done (a read-back of untouched "device" memory: zeros); nothing ever needs the device's bytes
static void stub_copy(void * dst, const void * src, size_t n, bool to_host) {
    if (to_host && n <= STUB_COPY_MAX && dst && src) memmove(dst, src, n);
}
static std::vector<std::string> g_fn_names;       // CUfunction = index + 1
static std::vector<unsigned long long> g_fn_launches;
static void stub_report() {
    if (const char * path = getenv("STUB_LAUNCH_REPORT")) {
        if (FILE * f = fopen(path, "w")) { for (size_t i = 0; i < g_fn_names.size(); ++i) fprintf(f, "%s %llu\n", g_fn_names[i].c_str(), g_fn_launches[i]); fclose(f); }
    }
}
extern "C" {
// ---- runtime: device
cudaError_t cudaGetDeviceCount(int * n) { *n = 1; return cudaSuccess; }
cudaError_t cudaGetDevice(int * d) { *d = 0; return cudaSuccess; }
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(cudaDeviceProp * p, int) {   // the dev box's card as the engine logs it
    memset(p, 0, sizeof(*p)); snprintf(p->name, sizeof(p->name), "NVIDIA GeForce RTX 5090 (cudastub dry run)");
    p->totalGlobalMem = (size_t) 32607 << 20; p->major = 12; p->minor = 0; p->warpSize = 32; p->multiProcessorCount = 170;
    p->sharedMemPerBlock = 49152; p->sharedMemPerBlockOptin = 101376; p->maxThreadsPerBlock = 1024; p->integrated = 0;
    return cudaSuccess;
}
cudaError_t cudaDeviceGetAttribute(int * v, cudaDeviceAttr a, int) {
    switch (a) {   // what the backend and the compiled-in CUB dispatch read
        case cudaDevAttrCooperativeLaunch:       *v = 1; break;
        case cudaDevAttrComputeCapabilityMajor:  *v = 12; break;
        case cudaDevAttrComputeCapabilityMinor:  *v = 0; break;
        case cudaDevAttrMultiProcessorCount:     *v = 170; break;
        case cudaDevAttrMaxThreadsPerBlock:      *v = 1024; break;
        case cudaDevAttrMaxSharedMemoryPerBlock: *v = 49152; break;
        case cudaDevAttrWarpSize:                *v = 32; break;
        case cudaDevAttrMaxBlockDimX: case cudaDevAttrMaxBlockDimY: *v = 1024; break;
        case cudaDevAttrMaxBlockDimZ:            *v = 64; break;
        case cudaDevAttrMaxGridDimX:             *v = 2147483647; break;
        case cudaDevAttrMaxGridDimY: case cudaDevAttrMaxGridDimZ: *v = 65535; break;
        default:                                 *v = 0;
    }
    return cudaSuccess;
}
cudaError_t cudaMemGetInfo(size_t * fr, size_t * tot) {
    *tot = (size_t) 32607 << 20; *fr = getenv("STUB_FREE_MIB") ? (size_t) atoll(getenv("STUB_FREE_MIB")) << 20 : (size_t) 31800 << 20; return cudaSuccess;
}
cudaError_t cudaDeviceCanAccessPeer(int * c, int, int) { *c = 0; return cudaSuccess; }
cudaError_t cudaDeviceGetPCIBusId(char * b, int len, int) { snprintf(b, len, "0000:00:00.0"); return cudaSuccess; }
cudaError_t cudaRuntimeGetVersion(int * v) { *v = 13020; return cudaSuccess; }
const char * cudaGetErrorString(cudaError_t) { return "cudastub"; }
// ---- runtime: memory
cudaError_t cudaMalloc(void ** p, size_t n) { *p = stub_map(n); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
cudaError_t cudaMallocManaged(void ** p, size_t n, unsigned) { *p = stub_map(n); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
cudaError_t cudaMallocHost(void ** p, size_t n) { *p = stub_map(n); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
cudaError_t cudaHostAlloc(void ** p, size_t n, unsigned) { *p = stub_map(n); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
cudaError_t cudaFree(void * p) { stub_unmap(p); return cudaSuccess; }
cudaError_t cudaFreeHost(void * p) { stub_unmap(p); return cudaSuccess; }
cudaError_t cudaHostGetDevicePointer(void ** d, void * h, unsigned) { *d = h; return cudaSuccess; }
cudaError_t cudaMemcpy(void * d, const void * s, size_t n, cudaMemcpyKind k) { stub_copy(d, s, n, k == cudaMemcpyDeviceToHost); return cudaSuccess; }
cudaError_t cudaMemcpyAsync(void * d, const void * s, size_t n, cudaMemcpyKind k, cudaStream_t) { stub_copy(d, s, n, k == cudaMemcpyDeviceToHost); return cudaSuccess; }
// ---- runtime: streams, events, capture
cudaError_t cudaStreamCreate(cudaStream_t * s) { *s = (cudaStream_t) malloc(8); return cudaSuccess; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t * s, unsigned) { *s = (cudaStream_t) malloc(8); return cudaSuccess; }
cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus * st) { *st = cudaStreamCaptureStatusNone; return cudaSuccess; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t * e, unsigned) { *e = (cudaEvent_t) malloc(8); return cudaSuccess; }
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * a, const void *) { memset(a, 0, sizeof(*a)); a->maxThreadsPerBlock = 1024; a->ptxVersion = 120; a->binaryVersion = 120; return cudaSuccess; }
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * n, const void *, int, size_t, unsigned) { *n = 1; return cudaSuccess; }
cudaError_t cudaGraphInstantiate(cudaGraphExec_t * e, cudaGraph_t, unsigned long long) { *e = (cudaGraphExec_t) malloc(8); return cudaSuccess; }
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t * g) { *g = (cudaGraph_t) malloc(8); return cudaSuccess; }
cudaError_t cudaGraphGetNodes(cudaGraph_t, cudaGraphNode_t *, size_t * n) { *n = 0; return cudaSuccess; }
// ---- the fat-binary registration nvcc emits (runs when the backend library loads)
void ** __cudaRegisterFatBinary(void *) { static void * h[4]; return h; }
void __cudaRegisterFatBinaryEnd(void **) {}
void __cudaUnregisterFatBinary(void **) {}
void __cudaRegisterFunction(void **, const char *, char *, const char *, int, void *, void *, void *, void *, int *) {}
void __cudaRegisterVar(void **, char *, char *, const char *, int, size_t, int, int) {}
// ---- driver
CUresult cuDeviceGet(CUdevice * d, int) { *d = 0; return CUDA_SUCCESS; }
CUresult cuCtxGetDevice(CUdevice * d) { *d = 0; return CUDA_SUCCESS; }
CUresult cuDeviceGetAttribute(int * v, CUdevice_attribute a, CUdevice) {
    *v = a == CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED ? 1 : a == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? 12 : 0; return CUDA_SUCCESS;
}
CUresult cuGetErrorString(CUresult, const char ** s) { *s = "cudastub"; return CUDA_SUCCESS; }
CUresult cuMemGetAllocationGranularity(size_t * g, const CUmemAllocationProp *, CUmemAllocationGranularity_flags) { *g = (size_t) 2 << 20; return CUDA_SUCCESS; }
CUresult cuMemAddressReserve(CUdeviceptr * p, size_t n, size_t, CUdeviceptr, unsigned long long) { *p = (CUdeviceptr) (uintptr_t) stub_map(n); return *p ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY; }
CUresult cuMemAddressFree(CUdeviceptr p, size_t) { stub_unmap((void *) (uintptr_t) p); return CUDA_SUCCESS; }
CUresult cuMemCreate(CUmemGenericAllocationHandle * h, size_t, const CUmemAllocationProp *, unsigned long long) { *h = 1; return CUDA_SUCCESS; }
CUresult cuMemcpyDtoH_v2(void * d, CUdeviceptr s, size_t n) { stub_copy(d, (const void *) (uintptr_t) s, n, true); return CUDA_SUCCESS; }
CUresult cuMemcpyHtoD_v2(CUdeviceptr d, const void * s, size_t n) { stub_copy((void *) (uintptr_t) d, s, n, false); return CUDA_SUCCESS; }
CUresult cuStreamIsCapturing(CUstream, CUstreamCaptureStatus * st) { *st = CU_STREAM_CAPTURE_STATUS_NONE; return CUDA_SUCCESS; }
CUresult cuModuleLoadData(CUmodule * m, const void *) { *m = (CUmodule) malloc(8); return CUDA_SUCCESS; }
CUresult cuModuleGetFunction(CUfunction * f, CUmodule, const char * name) {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_fn_names.empty()) atexit(stub_report);
    g_fn_names.reserve(4096); g_fn_names.push_back(name); g_fn_launches.push_back(0); *f = (CUfunction) (uintptr_t) g_fn_names.size(); return CUDA_SUCCESS;
}
static thread_local const char * g_last_symbol = nullptr;
CUresult cuLaunchKernelEx(const CUlaunchConfig *, CUfunction f, void **, void **) {
    const size_t i = (size_t) (uintptr_t) f; if (i >= 1 && i <= g_fn_launches.size()) { g_fn_launches[i - 1]++; g_last_symbol = g_fn_names[i - 1].c_str(); } return CUDA_SUCCESS;
}
// for hookspy (dry run only): the symbol of the manifest kernel this thread launched last, then cleared
const char * cudastub_take_last_symbol(void) { const char * s = g_last_symbol; g_last_symbol = nullptr; return s; }
// ---- cuBLAS / cuBLASLt: handles only
cublasStatus_t cublasCreate_v2(cublasHandle_t * h) { *h = (cublasHandle_t) malloc(8); return CUBLAS_STATUS_SUCCESS; }
const char * cublasGetStatusString(cublasStatus_t) { return "cudastub"; }
cublasStatus_t cublasLtCreate(cublasLtHandle_t * h) { *h = (cublasLtHandle_t) malloc(8); return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t * d, cublasComputeType_t, cudaDataType_t) { *d = (cublasLtMatmulDesc_t) calloc(1, 512); return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t * l, cudaDataType, uint64_t, uint64_t, int64_t) { *l = (cublasLtMatrixLayout_t) calloc(1, 512); return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t * p) { *p = (cublasLtMatmulPreference_t) calloc(1, 512); return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
        cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t r[], int * n) {
    memset(&r[0], 0, sizeof(r[0])); r[0].state = CUBLAS_STATUS_SUCCESS; *n = 1; return CUBLAS_STATUS_SUCCESS;
}
}
