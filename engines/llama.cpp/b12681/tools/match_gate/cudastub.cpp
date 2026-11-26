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
#include <algorithm>
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
// the dev box's card (RTX 5090, sm_120, measured: BoltBeam data/targets.json occupancy_scan): the SM count the hook's
// sm_count key reads, and the launch limits its loader checks a launches entry against (ext_load: block, grid, shared)
CUresult cuDeviceGetAttribute(int * v, CUdevice_attribute a, CUdevice) {
    switch (a) {
        case CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: *v = 1; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *v = 12; break;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: *v = 170; break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: *v = 2147483647; break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: *v = 65535; break;
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *v = 1024; break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: *v = 101376; break;
        case CU_DEVICE_ATTRIBUTE_WARP_SIZE: *v = 32; break;
        default: *v = 0;
    }
    return CUDA_SUCCESS;
}
CUresult cuGetErrorString(CUresult, const char ** s) { *s = "cudastub"; return CUDA_SUCCESS; }
CUresult cuMemGetAllocationGranularity(size_t * g, const CUmemAllocationProp *, CUmemAllocationGranularity_flags) { *g = (size_t) 2 << 20; return CUDA_SUCCESS; }
CUresult cuMemAddressReserve(CUdeviceptr * p, size_t n, size_t, CUdeviceptr, unsigned long long) { *p = (CUdeviceptr) (uintptr_t) stub_map(n); return *p ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY; }
CUresult cuMemAddressFree(CUdeviceptr p, size_t) { stub_unmap((void *) (uintptr_t) p); return CUDA_SUCCESS; }
CUresult cuMemCreate(CUmemGenericAllocationHandle * h, size_t, const CUmemAllocationProp *, unsigned long long) { *h = 1; return CUDA_SUCCESS; }
CUresult cuMemcpyDtoH_v2(void * d, CUdeviceptr s, size_t n) { stub_copy(d, (const void *) (uintptr_t) s, n, true); return CUDA_SUCCESS; }
CUresult cuMemcpyHtoD_v2(CUdeviceptr d, const void * s, size_t n) { stub_copy((void *) (uintptr_t) d, s, n, false); return CUDA_SUCCESS; }
// a list entry's scratch (ext-kernels.cpp: cuMemAlloc once a stream): without an address the hook runs the engine's own
// path and the dry run's engine never takes an entry with scratch, while the replay does
CUresult cuMemAlloc_v2(CUdeviceptr * p, size_t n) { *p = (CUdeviceptr) (uintptr_t) stub_map(n); return *p ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY; }
CUresult cuMemFree_v2(CUdeviceptr p) { stub_unmap((void *) (uintptr_t) p); return CUDA_SUCCESS; }
CUresult cuStreamIsCapturing(CUstream, CUstreamCaptureStatus * st) { *st = CU_STREAM_CAPTURE_STATUS_NONE; return CUDA_SUCCESS; }
// A module keeps a copy of its cubin (an ELF): a function's static shared bytes (.nv.shared.<name>), registers
// (.nv.info EIATTR_REGCOUNT) and launch bound (.nv.info.<name> EIATTR_MAX_THREADS) are read from it, so the hook's
// loader checks a launches entry against what the driver would say (cuFuncGetAttribute). A module that is not an ELF64
// (or a function it lacks) has no facts: its attributes read 0, and the loader drops the entry, as before.
struct stub_fn_facts { int smem = 0, regs = 0, max_threads = 0; };
static std::vector<stub_fn_facts> g_fn_facts;
static stub_fn_facts stub_elf_facts(const std::string & img, const char * name) {
    stub_fn_facts f;
    auto rd = [&](size_t o, size_t n) -> uint64_t { uint64_t v = 0; if (o + n <= img.size()) memcpy(&v, img.data() + o, n); return v; };
    if (img.size() < 64 || memcmp(img.data(), "\x7f" "ELF", 4) != 0 || img[4] != 2) return f;
    const uint64_t shoff = rd(0x28, 8), shent = rd(0x3a, 2), shnum = rd(0x3c, 2), shstr = rd(0x3e, 2);
    const uint64_t stroff = rd(shoff + shstr * shent + 0x18, 8);
    const std::string fn = name;
    for (uint64_t i = 0; i < shnum; ++i) {
        const uint64_t h = shoff + i * shent, off = rd(h + 0x18, 8), size = rd(h + 0x20, 8);
        const std::string sec = stroff + rd(h, 4) < img.size() ? std::string(img.c_str() + stroff + rd(h, 4)) : "";
        if (sec == ".nv.shared." + fn) f.smem = (int) std::max<uint64_t>(size, 1024) - 1024;   // the section holds the block's 1 KB reserved shared, the driver's attribute does not (19 of 19 cubins, measured)
        if (sec != ".nv.info" && sec != ".nv.info." + fn) continue;
        for (uint64_t o = off; o + 4 <= off + size;) {   // records: format, attribute, then a u16 value (HVAL) or a u16 size and that many bytes (SVAL)
            const int fmt = (int) rd(o, 1), attr = (int) rd(o + 1, 1);
            if (fmt == 4) {
                const uint64_t n = rd(o + 2, 2);
                if (sec == ".nv.info" && attr == 0x2f && n == 8) {   // EIATTR_REGCOUNT {symbol index, registers}: one row per function
                    const uint64_t symtab = [&] { for (uint64_t k = 0; k < shnum; ++k) if (rd(shoff + k * shent + 4, 4) == 2) return shoff + k * shent; return (uint64_t) 0; }();
                    const uint64_t sym = rd(o + 4, 4), symoff = rd(symtab + 0x18, 8), strtab = rd(shoff + rd(symtab + 0x28, 4) * shent + 0x18, 8);
                    if (symtab && fn == std::string(img.c_str() + std::min<uint64_t>(img.size() - 1, strtab + rd(symoff + sym * 24, 4)))) f.regs = (int) rd(o + 8, 4);
                }
                if (sec != ".nv.info" && (attr == 0x05 || attr == 0x10) && n == 12) {   // EIATTR_MAX_THREADS (launch bounds), EIATTR_REQNTID (the block it requires)
                    const int t = (int) (rd(o + 4, 4) * rd(o + 8, 4) * rd(o + 12, 4));
                    f.max_threads = f.max_threads > 0 ? std::min(f.max_threads, t) : t;
                }
                o += 4 + n;
            } else {
                o += 4;
            }
        }
    }
    if (f.regs > 0) {   // the block's registers: per thread rounded up to 8, 65,536 a block (sm_120), allocated a warp at a time
        const int per_warp = (f.regs + 7) / 8 * 8 * 32, by_regs = 65536 / per_warp * 32;
        f.max_threads = std::min(f.max_threads > 0 ? f.max_threads : 1024, std::min(1024, by_regs));
    } else {
        f.max_threads = 0;   // no register count read: unknown, the loader drops a launches entry on it
    }
    return f;
}
CUresult cuModuleLoadData(CUmodule * m, const void * image) {
    const char * p = (const char *) image; size_t n = 0;
    if (p && memcmp(p, "\x7f" "ELF", 4) == 0 && p[4] == 2) {   // the ELF's extent: its section headers and every section's bytes
        uint64_t shoff, shent = 0, shnum = 0; memcpy(&shoff, p + 0x28, 8); memcpy(&shent, p + 0x3a, 2); memcpy(&shnum, p + 0x3c, 2);
        n = shoff + shent * shnum;
        for (uint64_t i = 0; i < shnum; ++i) {
            uint32_t type; uint64_t off, size; memcpy(&type, p + shoff + i * shent + 4, 4); memcpy(&off, p + shoff + i * shent + 0x18, 8); memcpy(&size, p + shoff + i * shent + 0x20, 8);
            if (type != 8) n = std::max<uint64_t>(n, off + size);   // NOBITS take no bytes
        }
    }
    *m = (CUmodule) new std::string(p ? p : "", n); return CUDA_SUCCESS;
}
CUresult cuModuleGetFunction(CUfunction * f, CUmodule m, const char * name) {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_fn_names.empty()) atexit(stub_report);
    g_fn_names.reserve(4096); g_fn_names.push_back(name); g_fn_launches.push_back(0); g_fn_facts.push_back(stub_elf_facts(*(const std::string *) m, name));
    *f = (CUfunction) (uintptr_t) g_fn_names.size(); return CUDA_SUCCESS;
}
CUresult cuFuncGetAttribute(int * v, CUfunction_attribute a, CUfunction f) {
    const size_t i = (size_t) (uintptr_t) f; *v = 0;
    if (i < 1 || i > g_fn_facts.size()) return CUDA_SUCCESS;
    const stub_fn_facts & x = g_fn_facts[i - 1];
    *v = a == CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK ? x.max_threads : a == CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES ? x.smem : a == CU_FUNC_ATTRIBUTE_NUM_REGS ? x.regs : 0;
    return CUDA_SUCCESS;
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
