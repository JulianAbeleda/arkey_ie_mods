"""compare_spy.py A.jsonl B.jsonl: two hookspy records (e.g. a cudastub dry run and one real GPU run of the same build,
manifest and prompts). For every stored graph of A that B also stored (same ops, shapes and wiring), compare what the
allocator did: every tensor's offset inside its buffer, and what the engine's matcher took. Equal offsets = the dry run
reproduces the GPU's graph allocator for that graph. Absolute addresses differ by construction and are not compared."""
import hashlib, json, sys
def load(p):
    gs, hits = {}, {}
    for l in open(p):
        j = json.loads(l)
        if 'graph' in j:
            ts = j['tensors']; base = [b['base'] for b in j['buffers']]
            key = hashlib.sha256(json.dumps([(t['op'], t['type'], t['ne'], t['nb'], t['src'], t['view_src'], t['flags'], t['params']) for t in ts]).encode()).hexdigest()
            gs.setdefault(key, (j['graph'], j['n_nodes'], [(t['buf'], t['data'] - base[t['buf']] if t['buf'] >= 0 else None) for t in ts]))
        elif j['gid'] not in hits: hits[j['gid']] = j['hits']
    return gs, hits
(a, ha), (b, hb) = load(sys.argv[1]), load(sys.argv[2])
common = [k for k in a if k in b]
print(f"stored graphs: A {len(a)}, B {len(b)}, same ops / shapes / wiring / params in both: {len(common)}")
n_t = n_d = n_h = 0
for k in common:
    (ga, n, oa), (gb, _, ob) = a[k], b[k]
    d = sum(1 for x, y in zip(oa, ob) if x != y); n_t += len(oa); n_d += d
    same_hits = ha.get(ga) == hb.get(gb); n_h += not same_hits
    print(f"  {n} nodes (A graph {ga}, B graph {gb}): {len(oa) - d} of {len(oa)} tensors at the same offset in the same buffer; matcher took the same windows: {same_hits}")
print(f"TOTAL {n_t - n_d} of {n_t} tensor offsets equal; graphs where the matcher differs: {n_h} of {len(common)}")
