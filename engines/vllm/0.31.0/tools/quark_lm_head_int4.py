"""quark_lm_head_int4.py SRC DST: a copy of a Quark W4A16 checkpoint (one model.safetensors) with lm_head quantized the same way as
its linear layers: int4 symmetric, groups of 128 along the input, scale = amax / 7.5, round half to even, clamped to
[-8, 7] (the value histogram of the checkpoint's own layers), packed 8 per int32 in AWQ nibble order ("reorder"), zero
points 0. The safetensors file is streamed: every other tensor's bytes are copied as they are."""
import json, os, pathlib, struct, sys
import torch
from safetensors import safe_open

REV = [0, 4, 1, 5, 2, 6, 3, 7]  # vllm auto_awq._REVERSE_AWQ_PACK_ORDER: column i of 8 sits at nibble REV[i]
src, dst = pathlib.Path(sys.argv[1]).expanduser(), pathlib.Path(sys.argv[2]).expanduser()
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
  if f.name not in ("model.safetensors", "config.json", "SHA256SUMS", "crc32.txt") and not (dst / f.name).exists():
    os.symlink(f, dst / f.name)

with safe_open(src / "model.safetensors", "pt", device="cuda") as f:
  w = f.get_tensor("lm_head.weight")  # [vocab, hidden]
inp, out = w.shape[1], w.shape[0]
packs, scales, num, den = [], [], 0.0, 0.0
for c in range(0, out, 16384):  # vocab chunks: the int64 work for all 1.27e9 weights at once is ~10 GB
  wt = w[c:c + 16384].t().float()  # [hidden (in), chunk (out)]
  co = wt.shape[1]
  g = wt.view(inp // 128, 128, co)
  sc = (g.abs().amax(dim=1) / 7.5).clamp_min(1e-8)  # [in/128, chunk]
  q = torch.round(g / sc[:, None, :]).clamp(-8, 7)
  num += ((q * sc[:, None, :]).view(inp, co) - wt).pow(2).sum().item(); den += wt.pow(2).sum().item()
  nib = (q.to(torch.int64).view(inp, co) & 0xF).view(inp, co // 8, 8)
  pk = torch.zeros(inp, co // 8, dtype=torch.int64, device=w.device)
  for i in range(8):
    pk |= nib[:, :, i] << (REV[i] * 4)
  packs.append(pk.to(torch.int32).cpu()); scales.append(sc.to(torch.bfloat16).cpu())
packed, scale, err = torch.cat(packs, 1), torch.cat(scales, 1), (num / den) ** 0.5
new = {"lm_head.weight": packed.contiguous(), "lm_head.weight_scale": scale.contiguous(),
       "lm_head.weight_zero_point": torch.zeros(inp // 128, out // 8, dtype=torch.int32)}
print(f"lm_head {tuple(w.shape)} -> weight {tuple(packed.shape)}, scale {tuple(scale.shape)}; relative error {err:.4f}")

# stream the file: header, then each tensor's bytes (old ones copied from the source file)
raw = (src / "model.safetensors").open("rb")
n = struct.unpack("<Q", raw.read(8))[0]
head = json.loads(raw.read(n)); base = 8 + n
meta = head.pop("__metadata__", None)
DT = {torch.int32: "I32", torch.bfloat16: "BF16"}
items, at = [], 0
for name, t in sorted(head.items(), key=lambda kv: kv[1]["data_offsets"][0]):
  if name == "lm_head.weight":
    continue
  a, b = t["data_offsets"]; items.append((name, {**t, "data_offsets": [at, at + b - a]}, ("copy", base + a, b - a))); at += b - a
for name, t in new.items():
  nb = t.numel() * t.element_size()
  items.append((name, {"dtype": DT[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + nb]}, ("new", t))); at += nb
header = {**({"__metadata__": meta} if meta else {}), **{name: h for name, h, _ in items}}
hb = json.dumps(header, separators=(",", ":")).encode()
hb += b" " * (-len(hb) % 8)
with (dst / "model.safetensors").open("wb") as o:
  o.write(struct.pack("<Q", len(hb))); o.write(hb)
  for _, _, how in items:
    if how[0] == "copy":
      raw.seek(how[1]); left = how[2]
      while left:
        chunk = raw.read(min(left, 64 << 20)); o.write(chunk); left -= len(chunk)
    else:
      t = how[1]; o.write((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes())

cfg = json.loads((src / "config.json").read_text())
cfg["quantization_config"]["exclude"] = [e for e in cfg["quantization_config"]["exclude"] if e != "lm_head"]
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
print(f"wrote {dst / 'model.safetensors'}: {(dst / 'model.safetensors').stat().st_size / 1e9:.2f} GB")
