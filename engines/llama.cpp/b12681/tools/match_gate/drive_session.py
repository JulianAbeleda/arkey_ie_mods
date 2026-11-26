"""drive_session.py OUT SESSION.json FIRST-LAST [REQUEST.json]: drive the dry-run server of OUT (dryrun.sh) through turns FIRST..LAST
of a recorded session as a served session runs them (BoltBeam's session-ab with --history recorded): the chat endpoint, the
messages system, then each turn's user and recorded assistant text, max_tokens = the turn's reply_tokens, and every field of
REQUEST.json (the served requests' own: a field such as ignore_eos adds a sampler stage to the step graph, so a run without it
records another graph). Turns before FIRST are history only: the first request carries them, as session-ab's fill does.
Replies are meaningless (the stub device reads back zeros); only the graphs and what the hook did with them are kept.
Each turn is two requests: the messages with max_tokens 0 (the prompt alone), then the reply; the recorder's pass count between
them marks where that turn's DECODE steps begin. Writes OUT/turns.json: per turn the tokens evaluated / predicted and
decode_passes = [first, end) in the recorder's pass numbering (a pass is written when the next one of its graph starts, so the
bounds are 1-2 passes late). Then stops the server by its PID."""
import json, os, signal, sys, time, urllib.request

out, sess, (a, b) = sys.argv[1], json.load(open(sys.argv[2])), (int(x) for x in sys.argv[3].split("-"))
extra = json.load(open(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[4] else {}
url = f"http://127.0.0.1:{os.environ.get('DRY_PORT', '39517')}"
while not os.path.exists(out + "/server.pid"):   # dryrun.sh's memcap --wait queues the server: its 300 s start from here
    if os.path.exists(out + "/server.log") and "memcap: refused" in open(out + "/server.log").read():
        sys.exit("server not started: " + open(out + "/server.log").read()[-600:])
    time.sleep(2)
for _ in range(150):
    try:
        urllib.request.urlopen(url + "/health", timeout=2)
        break
    except Exception:
        time.sleep(2)
        if os.path.exists(out + "/server.pid") and not os.path.exists(f"/proc/{int(open(out + '/server.pid').read())}"):
            sys.exit("server died: " + open(out + "/server.log").read()[-600:])
else:
    sys.exit("server not healthy")
pid = int(open(out + "/server.pid").read())


def passes():
    try:
        return sum(1 for line in open(out + "/spy.jsonl") if line.startswith('{"pass"'))
    except FileNotFoundError:
        return 0


def post(msgs, n):
    """One chat request with the served fields; a streamed reply is read to its end. The server's timings and usage."""
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps({"messages": msgs, **extra, "max_tokens": n}).encode(), {"Content-Type": "application/json"})
    got = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if line.startswith("data:") and line != "data: [DONE]":
                line = line[5:]
            elif not line.startswith("{"):
                continue
            j = json.loads(line)
            got.update({k: j[k] for k in ("timings", "usage") if j.get(k)})
    return got


msgs, rows = [{"role": "system", "content": sess.get("system", "")}], []
for t in sess["turns"]:
    if t["i"] > b:
        break
    msgs.append({"role": "user", "content": t["user"]})
    if t["i"] >= a:
        t0 = time.time()
        post(msgs, 0)                  # the prompt alone: every pass before this point of the record is prefill
        p0 = passes()
        j = post(msgs, t["reply_tokens"]).get("timings", {})   # the same prompt from the cache: from here on decode steps (and the draft's)
        rows.append({"turn": t["i"], "evaluated": j.get("prompt_n"), "predicted": j.get("predicted_n"), "cached": j.get("cache_n"), "wall_s": round(time.time() - t0, 2),
                     "decode_passes": [p0, passes()], "passes_after": passes(), "rss_mb": int(open(f"/proc/{pid}/status").read().split("VmRSS:")[1].split()[0]) // 1024})
        print(rows[-1], flush=True)
    msgs.append({"role": "assistant", "content": t["assistant"]})
json.dump(rows, open(out + "/turns.json", "w"), indent=1)
os.kill(pid, signal.SIGTERM)
for _ in range(20):
    if not os.path.exists(f"/proc/{pid}"):
        break
    time.sleep(0.5)
