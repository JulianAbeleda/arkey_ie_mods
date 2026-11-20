"""stamp.py OUT MATCHER_SOURCE: what the report must say about the run that made it, added to OUT/gate.json after hook_gate wrote
it: recording_sha256 (the sha256 of the recording beside it, spy.jsonl and turns.json: a reader takes the report only with that
recording), matcher_source_sha256 (the engine source the replay was built from), produced_by "dry run" and device null (a run on
the do-nothing CUDA runtime has no device; only a run on the GPU could name one, and this tool makes none)."""
import hashlib, json, pathlib, sys

out, src = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
rep = json.loads((out / "gate.json").read_text())
rep.update(recording_sha256={n: sha(out / n) for n in ("spy.jsonl", "turns.json")}, matcher_source_sha256=sha(src), produced_by="dry run", device=None)
(out / "gate.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
