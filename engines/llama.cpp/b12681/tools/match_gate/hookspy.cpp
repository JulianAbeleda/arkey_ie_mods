// hookspy: LD_PRELOAD in front of an UNMODIFIED engine build. It interposes the kernels module's hook entry
// (ggml_cuda_ext_kernels_run, called through the PLT for every compute node of every CUDA split graph), calls the
// engine's own function, and records to HOOKSPY_OUT (JSON lines):
//   {"graph": ...}  once per distinct graph (ops, shapes, strides, flags, op_params, wiring, views, use counts and
//                   every tensor's ADDRESS as the engine's allocator placed it): schema boltbeam.engine_graph_dump.v2
//   {"pass": ...}   once per evaluation of a graph: which nodes the engine's matcher took and how many nodes each took
// It changes nothing the engine computes. Works the same in front of the real GPU (one short run gives the faithful
// dump) and in a cudastub dry run on the CPU. HOOKSPY_NO_PREFETCH=1 (dry run only): file mappings are not populated
// and WILLNEED advice is dropped, so a model file is mapped but never read.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

typedef int (*run_fn)(void *, ggml_cgraph *, int, bool (*)(void *, ggml_tensor *), void *);
static std::mutex g_mu;
static FILE *     g_out = nullptr;
struct pass_state { int last_i = 1 << 30; long gid = -1; long calls = 0; int n_nodes = 0; int64_t last_ne[4] = {0, 0, 0, 0}; std::vector<std::pair<int, int>> hits, gaps; int prev = -1; std::map<std::string, long> by; std::string pending; };
static std::map<const ggml_cgraph *, pass_state> g_pass;
static std::unordered_map<uint64_t, long> g_seen;   // graph signature -> gid
static long g_n_pass = 0;
// a graph is written when it is one of the first HOOKSPY_PER_SIZE of its node count, or when the engine's matcher did
// something on it that no written graph of that node count shows (a new set of taken nodes): decided when its pass ends
static const long g_per_size = getenv("HOOKSPY_PER_SIZE") ? atol(getenv("HOOKSPY_PER_SIZE")) : 4;
static std::map<int, long> g_written;                   // node count -> graphs written
static std::set<std::pair<int, uint64_t>> g_patterns;   // (node count, hash of the taken nodes) seen in a written graph

static void fnv(uint64_t & h, uint64_t v) { h = (h ^ v) * 1099511628211ull; }
static std::string esc(const char * s) { std::string o; for (; *s; ++s) { if (*s == '"' || *s == '\\') o += '\\'; if ((unsigned char) *s >= 32) o += *s; } return o; }

static void flush_pass(const ggml_cgraph * g, pass_state & p) {
    if (p.calls == 0 || g_out == nullptr) return;
    if (!p.pending.empty()) {
        uint64_t h = 1469598103934665603ull;
        for (auto & [i, n] : p.hits) { fnv(h, (uint64_t) i); fnv(h, (uint64_t) n); }
        const bool fresh = g_patterns.emplace(p.n_nodes, h).second;
        if (fresh || g_written[p.n_nodes] < g_per_size) { g_written[p.n_nodes]++; fputs(p.pending.c_str(), g_out); }
    }
    // kind: the graph's node count and the shape of its last node (the result: it carries the token count)
    fprintf(g_out, "{\"pass\": %ld, \"gid\": %ld, \"n_nodes\": %d, \"last_ne\": [%lld, %lld, %lld, %lld], \"calls\": %ld, \"by\": {", g_n_pass++, p.gid, p.n_nodes,
            (long long) p.last_ne[0], (long long) p.last_ne[1], (long long) p.last_ne[2], (long long) p.last_ne[3], p.calls);
    { bool first = true; for (auto & [sym, c] : p.by) { fprintf(g_out, "%s\"%s\": %ld", first ? "" : ", ", esc(sym.c_str()).c_str(), c); first = false; } }
    fprintf(g_out, "}, \"hits\": [");
    for (size_t k = 0; k < p.hits.size(); ++k) fprintf(g_out, "%s[%d, %d]", k ? ", " : "", p.hits[k].first, p.hits[k].second);
    if (p.prev + 1 < p.n_nodes) p.gaps.emplace_back(p.prev + 1, p.n_nodes);   // the tail the engine never called the hook on
    fprintf(g_out, "], \"gaps\": [");   // node ranges [a, b) the hook was not called on: views, a window taken, or a stock fusion that took them
    for (size_t k = 0; k < p.gaps.size(); ++k) fprintf(g_out, "%s[%d, %d]", k ? ", " : "", p.gaps[k].first, p.gaps[k].second);
    fprintf(g_out, "]}\n"); fflush(g_out);
    p.calls = 0; p.hits.clear(); p.gaps.clear(); p.prev = -1; p.by.clear(); p.pending.clear();
}
static long dump_graph(const ggml_cgraph * g, std::string & text) {
    std::unordered_map<const ggml_tensor *, int> id;
    std::vector<const ggml_tensor *> all;
    for (int k = 0; k < g->n_nodes; ++k) { id[g->nodes[k]] = k; all.push_back(g->nodes[k]); }
    for (size_t q = 0; q < all.size(); ++q) {   // every tensor a node reaches as a src or as the tensor it views
        const ggml_tensor * t = all[q];
        for (int s = -1; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * x = s < 0 ? t->view_src : t->src[s];
            if (x != nullptr && id.emplace(x, (int) all.size()).second) all.push_back(x);
        }
    }
    std::map<ggml_backend_buffer_t, int> bufs;
    uint64_t h = 1469598103934665603ull;
    for (const ggml_tensor * t : all) {
        if (t->buffer != nullptr) bufs.emplace(t->buffer, (int) bufs.size());
        const uint64_t base = t->buffer ? (uint64_t) (uintptr_t) ggml_backend_buffer_get_base(t->buffer) : 0;
        for (uint64_t v : { (uint64_t) t->op, (uint64_t) t->type, (uint64_t) t->flags, (uint64_t) t->view_offs, (uint64_t) (uintptr_t) t->data - base,
                            (uint64_t) (t->buffer ? bufs[t->buffer] : -1), (uint64_t) (t->view_src ? id[t->view_src] : -1) }) fnv(h, v);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) { fnv(h, (uint64_t) t->ne[d]); fnv(h, (uint64_t) t->nb[d]); }
        for (int s = 0; s < GGML_MAX_SRC; ++s) fnv(h, (uint64_t) (t->src[s] ? id[t->src[s]] : -1));
        for (size_t d = 0; d < sizeof(t->op_params) / sizeof(int32_t); ++d) fnv(h, (uint64_t) (uint32_t) t->op_params[d]);
    }
    auto it = g_seen.find(h);
    if (it != g_seen.end()) return it->second;
    const long gid = (long) g_seen.size(); g_seen[h] = gid;
    if (g_out == nullptr) return gid;
    char * mem = nullptr; size_t mem_n = 0; FILE * g_out = open_memstream(&mem, &mem_n);   // shadows the file: the text is kept until the pass ends
    fprintf(g_out, "{\"graph\": %ld, \"schema\": \"boltbeam.engine_graph_dump.v2\", \"sig\": \"%016llx\", \"n_nodes\": %d, \"buffers\": [", gid, (unsigned long long) h, g->n_nodes);
    std::vector<ggml_backend_buffer_t> bl(bufs.size());
    for (auto & [b, k] : bufs) bl[k] = b;
    for (size_t k = 0; k < bl.size(); ++k) fprintf(g_out, "%s{\"base\": %llu, \"size\": %zu, \"name\": \"%s\"}", k ? ", " : "", (unsigned long long) (uintptr_t) ggml_backend_buffer_get_base(bl[k]), ggml_backend_buffer_get_size(bl[k]), esc(ggml_backend_buffer_name(bl[k])).c_str());
    fprintf(g_out, "], \"tensors\": [");
    for (size_t q = 0; q < all.size(); ++q) {
        const ggml_tensor * t = all[q];
        fprintf(g_out, "%s{\"id\": %zu, \"op\": \"%s\", \"opi\": %d, \"type\": \"%s\", \"typei\": %d, \"ne\": [%lld, %lld, %lld, %lld], \"nb\": [%zu, %zu, %zu, %zu], \"flags\": %d, \"data\": %llu, \"buf\": %d, \"view_src\": %d, \"view_offs\": %zu, \"uses\": %d, \"name\": \"%s\", \"src\": [",
                q ? "," : "", q, ggml_op_name(t->op), (int) t->op, ggml_type_name(t->type), (int) t->type, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                t->nb[0], t->nb[1], t->nb[2], t->nb[3], (int) t->flags, (unsigned long long) (uintptr_t) t->data, t->buffer ? bufs[t->buffer] : -1, t->view_src ? id[t->view_src] : -1, t->view_offs,
                (int) q < g->n_nodes ? ggml_node_get_use_count(g, (int) q) : -1, esc(t->name).c_str());
        bool first = true;
        for (int s = 0; s < GGML_MAX_SRC; ++s) if (t->src[s]) { fprintf(g_out, "%s%d", first ? "" : ", ", id[t->src[s]]); first = false; } else break;
        fprintf(g_out, "], \"params\": \"");
        for (size_t d = 0; d < sizeof(t->op_params); ++d) fprintf(g_out, "%02x", ((const unsigned char *) t->op_params)[d]);
        fprintf(g_out, "\"}");
    }
    fprintf(g_out, "]}\n"); fclose(g_out);
    text.assign(mem, mem_n); free(mem);
    return gid;
}
static void at_exit() { std::lock_guard<std::mutex> l(g_mu); for (auto & [g, p] : g_pass) flush_pass(g, p); if (g_out) fclose(g_out); g_out = nullptr; }

int ggml_cuda_ext_kernels_run(void * stream, ggml_cgraph * cgraph, int i, bool (*stock)(void *, ggml_tensor *), void * stock_ctx) {
    static run_fn real = (run_fn) dlsym(RTLD_NEXT, "_Z25ggml_cuda_ext_kernels_runPvP11ggml_cgraphiPFbS_P11ggml_tensorES_");
    {
        std::lock_guard<std::mutex> l(g_mu);
        if (g_out == nullptr && getenv("HOOKSPY_OUT")) { g_out = fopen(getenv("HOOKSPY_OUT"), "a"); atexit(at_exit); }
        pass_state & p = g_pass[cgraph];
        if (i <= p.last_i) {          // the node index went back: a new evaluation of this graph object
            flush_pass(cgraph, p);
            p.gid = dump_graph(cgraph, p.pending); p.n_nodes = cgraph->n_nodes;
            if (cgraph->n_nodes > 0) memcpy(p.last_ne, cgraph->nodes[cgraph->n_nodes - 1]->ne, sizeof(p.last_ne));
        }
        if (i > p.prev + 1) p.gaps.emplace_back(p.prev + 1, i);
        p.last_i = p.prev = i; p.calls++;
    }
    const int n = real(stream, cgraph, i, stock, stock_ctx);
    if (n != 0) {
        // dry run: the stub says which manifest kernel was launched; in front of the real GPU the entry is found by the gate's replay
        static const char * (*take)(void) = (const char * (*)(void)) dlsym(RTLD_DEFAULT, "cudastub_take_last_symbol");
        const char * sym = take ? take() : nullptr;
        std::lock_guard<std::mutex> l(g_mu); pass_state & p = g_pass[cgraph]; p.hits.emplace_back(i, n); if (sym) p.by[sym]++;
    }
    return n;
}

// dry run only: a model file is mapped, never read
static const bool g_no_prefetch = getenv("HOOKSPY_NO_PREFETCH") != nullptr;
extern "C" void * mmap(void * addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (g_no_prefetch) flags &= ~MAP_POPULATE;
    return (void *) syscall(SYS_mmap, addr, len, prot, flags, fd, off);
}
extern "C" void * mmap64(void * addr, size_t len, int prot, int flags, int fd, off_t off) { return mmap(addr, len, prot, flags, fd, off); }
extern "C" int posix_madvise(void * addr, size_t len, int advice) {
    if (g_no_prefetch && advice == POSIX_MADV_WILLNEED) return 0;
    return (int) syscall(SYS_madvise, addr, len, advice) == 0 ? 0 : errno;
}
extern "C" int madvise(void * addr, size_t len, int advice) {
    if (g_no_prefetch && advice == MADV_WILLNEED) return 0;
    return (int) syscall(SYS_madvise, addr, len, advice);
}
