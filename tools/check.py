#!/usr/bin/env python3
"""Check every engines/<engine>/<version>/ folder. Standard library only. Names no engine: engine facts are rows in
engines.json, mod facts are rows in mods.json, machine paths are in the untracked local config (local.json).

Repo
  0 layout    every folder and file under engines/, benchmarks/ and at the top level is known; engines.json
              versions equal the folders; mod names are unique per folder and ids (engine/version/name) unique overall
Per folder
  1 base      the base is the recorded one and the touched files are unmodified (engines.json base_identity.kind)
  2 series    every patch applies in order on a clean copy of the base; the series reverts to the base bytes
  3 deps      every patch applies on the base plus only the patches its depends_on.apply names (transitively)
  4 lines     added / removed lines and the file list equal the row; the budget follows the folder's budget_rule,
              and a change of lines or budget against HEAD needs a budget_raised entry {from, to, why, date}
  5 switch    the switch, check and marker are in the patch's added lines (per the engine's switch_presence
              templates); every switch-shaped or env-read literal a patch adds is declared; no archived switch or
              archived sha256 is in the series or parked; the count of hunks that remove stock lines equals the row
  5b rows     known keys only; a series mod has a one-line title (parked may); output label in {identical, quality-gated, untested} with its evidence (identical:
              a counted match, quality-gated: a gate); a rule_exceptions entry for every missing check, missing
              switch or switch that is on by default; no model name and no other engine's name in a row or a patch
              header or added line (engines.json lint); switch.default_run in {on, off} with its why, on only for
              identical, and stated by every switched row of a version once one row states it; default_run_where
              (on only) names known device facts (compute_capability, sm_count) with integer values
  6 tree      base + the listed patches is byte-equal to each tested tree on the touched files, no other base file
              differs, and the tree holds no extra file the base does not ignore; a folder with mods has tested
              trees or an explicit tested_trees_none {why}
  7 archive   every archived patch's sha256 and lines equal its row; archived patches are not budgeted and never in
              the series, parked or index.json's active list (a tested tree may apply one, for byte equality only);
              archive-ledger.json is append-only and keeps every archived patch ever recorded
  8 files     every file in patches/, parked/, archive/ and tools/ has a row; also_applies_to is checked by applying;
              a kernels row's change of lines or budget against HEAD needs a budget_raised entry, as in 4
  9 exports   index.json and each archive/README.md equal what the rows generate (--write regenerates them and
              appends new archived rows to the ledger); index.json requires is engines.json requires (the oldest
              consumer version that reads it: requires.boltbeam a dotted version, with a why)
  10 stack    the whole series against stack_budget.json: total lines, the most mods on one file, the most overlapping
              hunks in one file, the longest depends_on.apply chain, hunks stacked on another mod's added lines; a
              crossing names the files, functions and mods and the extension point to introduce; a cap is derived
              (measured + ceil(measured / series mods)) or an Owner-approved raise to the measured value; a stack
              health summary prints on every run; a cap whose metric dropped (a mod retired or shrunk) fails until it
              is lowered to measured + ceil(measured / series mods) (the ratchet: freed room is not silently reused)
Modules (a version's mods.json modules list): one patch per engine extension point; a mod row that names a module is a
  feature of it (switch, check, label, evidence) and holds no patch. Checks 2, 4, 6 and 10 run on the module patches,
  check 3 becomes: every module applies alone on the base; check 5's declared literals are those of the module's rows
Retired (engines.json retired_versions, or a version's mods.json retired list): linted for format (checks 4, 5, 5b,
  7, 8) and kept in the repo; never applied, never in the stack measurement, never in index.json's current lists
  (index.json retired names them)

What this does NOT prove: that a hunk which removes or rewrites stock lines keeps the stock behaviour with the switch
off. That is a run-time fact. Its evidence is each row's output.evidence and check 6. Nothing here compiles or runs
the engine.

Exit 0 = no failure. A missing local checkout, tested tree, history or unrecorded identity is a named SKIP; with
--strict any SKIP fails.
"""
import argparse, base64, csv, hashlib, json, math, os, re, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails, skips = [], []
LOCAL, GIT = {}, None
LABELS = {"identical", "quality-gated", "untested"}
ROW_KEYS = {"name", "title", "patch", "engine", "base", "switch", "switch_null_why", "check", "reads", "marker", "files", "lines", "budget",
            "budget_raised", "depends_on", "output", "gain", "exercised_by", "limits", "rule_exceptions", "static", "scrubbed_from",
            "parked_why", "applies_after", "retired", "module"}
ARCH_KEYS = {"name", "kind", "date", "patch", "switch", "written_on", "why", "evidence", "files", "lines", "sha256", "scrubbed_from"}
SPEC_KEYS = {"schema", "engine", "version", "base", "about", "budget_rule", "modules", "mods", "parked", "archived", "retired", "tools", "kernels", "engine_flags",
             "tested_trees", "tested_trees_none", "verified", "also_applies_to"}
MODULE_FACTS = ("patch", "files", "lines", "budget", "budget_raised", "static")  # what a module holds for its feature rows
MODULE_KEYS = {"name", "title", "point", *MODULE_FACTS}
GAIN_KEYS = {"value", "unit", "vs", "conditions", "run", "basis"}
EVID_KEYS = {"run", "numbers", "match", "gate"}
TOP_FILES = {".git", ".githooks", ".gitignore", "DETAILS.md", "LICENSE", "LICENSES", "README.md", "engines", "engines.json", "index.json",
             "tools", "local.json", "local.example.json", "archive-ledger.json", "__pycache__", "benchmarks"}
VERSION_ENTRIES = {"mods.json", "stack_budget.json", "patches", "parked", "archive", "retired", "tools", "kernels"}
RETIRED_KEYS = {"date", "why"}  # a retired mod's row "retired", and an engines.json retired_versions entry
KERNEL_KEYS = {"name", "dir", "hook", "files", "source", "lines", "budget", "budget_raised", "output", "gain", "exercised_by", "limits"}
INDEX_MOD_KEYS = {"id", "engine", "version", "name", "title", "state", "order", "applier", "base", "switch", "switch_null_why", "check", "check_needs",
                  "marker", "knobs", "output", "patch", "sha256", "lines", "budget", "files", "depends_on", "rule_exceptions", "module"}
TITLE_MAX = 80  # a mod's title: what it changes, in one line (a consumer's list shows it first)
BUDGET_RULES = {"measured": lambda n: n, "ceil_105_10": lambda n: int(math.ceil(n * 1.05 / 10) * 10)}


def say(tag, msg):
    print(f"  {tag:5} {msg}")
    if tag == "FAIL":
        fails.append(msg)
    if tag == "SKIP":
        skips.append(msg)


def run(cmd, cwd, stdin=None):
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=os.path.dirname(cwd), LC_ALL="C")
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, input=stdin)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def parse(path):
    """files, added, removed, hunks that remove lines, added text, created files (header: header())."""
    files, created, added, a, r, mod, rm, old, hdr_lines, in_hdr = [], set(), [], 0, 0, 0, False, None, [], True
    for line in open(path, encoding="utf-8", errors="surrogateescape"):
        if in_hdr and (line.startswith("diff --git") or line.startswith("--- ")):
            in_hdr = False
        if in_hdr:
            hdr_lines.append(line)
            continue
        if line.startswith("--- "):
            old = line[4:].split("\t")[0].strip()
        elif line.startswith("+++ "):
            f = line[4:].split("\t")[0].strip()
            f = f[2:] if f.startswith("b/") else f
            files.append(f)
            if old == "/dev/null":
                created.add(f)
        elif line.startswith("@@"):
            mod, rm = mod + rm, False
        elif line.startswith("+"):
            a += 1
            added.append(line[1:])
        elif line.startswith("-"):
            r, rm = r + 1, True
    return files, a, r, mod + rm, "".join(added), created


def header(path):
    """the text before the first diff: free text that is not applied."""
    out = []
    for line in open(path, encoding="utf-8", errors="surrogateescape"):
        if line.startswith("diff --git") or line.startswith("--- "):
            break
        out.append(line)
    return "".join(out)


def local_path(ref):
    """A path reference: '<root>/rest' with <root> in local.json roots, or None when this machine has no such root."""
    if not ref:
        return None
    head, _, rest = ref.partition("/")
    root = (LOCAL.get("roots") or {}).get(head)
    return os.path.join(os.path.expanduser(root), rest) if root else None


def checkout_of(engine, version):
    p = (LOCAL.get("checkouts") or {}).get(f"{engine}/{version}")
    return os.path.expanduser(p) if p else None


def head_json(rel):
    if not GIT:
        return None
    p = run(["git", "show", f"HEAD:{rel}"], GIT)
    return json.loads(p.stdout) if p.returncode == 0 else None


class GitApply:
    """applier 'git_apply': patches against a source tree."""

    def __init__(self, base, files):
        self.base, self.files = base, files

    def fresh(self):
        tmp = tempfile.mkdtemp(prefix="ie_mods_")
        for f in self.files:
            src = os.path.join(self.base, f)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(os.path.join(tmp, f)), exist_ok=True)
                shutil.copyfile(src, os.path.join(tmp, f))
        return tmp

    def apply(self, tmp, patch, revert=False):
        p = run(["git", "apply"] + (["-R"] if revert else []) + [patch], tmp)
        return p.returncode == 0, p.stderr.strip().splitlines()[-1:] or [""]


class PatchP1(GitApply):
    """applier 'patch_p1': patch -p1 in an installed package folder (or a source checkout); dry run first."""

    def apply(self, tmp, patch, revert=False):
        cmd = ["patch", "-p1", "-s", "--forward", "--no-backup-if-mismatch", "-r", "-", "-i", patch] + (["-R"] if revert else [])
        p = run(cmd + ["--dry-run"], tmp)
        if p.returncode == 0:
            p = run(cmd, tmp)
        return p.returncode == 0, (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]


APPLIERS = {"git_apply": GitApply, "patch_p1": PatchP1}


class GitCommit:
    """base_identity 'git_commit': HEAD of the checkout equals base.commit; the touched files are clean."""

    @staticmethod
    def check(base, checkout, files):
        head = run(["git", "rev-parse", "HEAD"], checkout).stdout.strip()
        if head != base["commit"]:
            return [("FAIL", f"base    checkout {checkout} is at {head[:9] or '?'}, row says {base['commit'][:9]}")]
        present = [f for f in files if os.path.exists(os.path.join(checkout, f))]
        dirty = run(["git", "status", "--porcelain", "--"] + present, checkout).stdout.strip()
        if dirty:
            return [("FAIL", "base    touched files modified in the base checkout:\n" + dirty)]
        return [("ok", f"base    {head[:9]}, {len(present)} touched files clean")]

    @staticmethod
    def tracked(base, checkout):
        return [f for f in run(["git", "ls-files"], checkout).stdout.split("\n") if f]

    @staticmethod
    def ignored(checkout, paths):
        if not paths:
            return set()
        p = run(["git", "check-ignore", "--no-index", "--stdin"], checkout, stdin="\n".join(paths) + "\n")
        return set(p.stdout.split("\n"))


class PipWheel:
    """base_identity 'pip_version_wheel_hash': version from METADATA; RECORD sha256 (the exact build) when recorded;
    every touched file's sha256 equals its RECORD entry; the wheel's sha256 when a wheel was kept."""

    @staticmethod
    def info(base, site):
        return os.path.join(site, f"{base['package']}-{base['version']}.dist-info")

    @staticmethod
    def record(base, site):
        return {row[0]: row[1] for row in csv.reader(open(os.path.join(PipWheel.info(base, site), "RECORD"), encoding="utf-8")) if row}

    @staticmethod
    def check(base, site, files):
        out, info = [], PipWheel.info(base, site)
        if not os.path.isdir(info):
            return [("FAIL", f"base    {info} not found: {base['package']} {base['version']} is not installed at {site}")]
        got = re.search(r"^Version: (\S+)", open(os.path.join(info, "METADATA"), encoding="utf-8").read(), re.M)
        if not got or got.group(1) != base["version"]:
            return [("FAIL", f"base    METADATA version {got and got.group(1)}, row says {base['version']}")]
        h = sha256(os.path.join(info, "RECORD"))
        if base.get("record_sha256"):
            ok = h == base["record_sha256"]
            out.append(("ok" if ok else "FAIL", f"base    {base['package']} {base['version']}, RECORD sha256 {h[:12]}" + ("" if ok else f", row says {base['record_sha256'][:12]}")))
        else:
            out.append(("SKIP", f"base    {base['package']} {base['version']}: no RECORD sha256 recorded, the exact build is unverified"))
        record, dirty = PipWheel.record(base, site), []
        for f in files:
            want = record.get(f, "")
            p = os.path.join(site, f)
            have = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(open(p, "rb").read()).digest()).rstrip(b"=").decode() if os.path.exists(p) else None
            if not want.startswith("sha256=") or have != want:
                dirty.append(f)
        out.append(("FAIL", "base    touched files differ from the installed RECORD: " + ", ".join(dirty)) if dirty
                   else ("ok", f"base    {len(files)} touched files equal the installed RECORD"))
        if not base.get("wheel_sha256"):
            out.append(("SKIP", f"base    wheel sha256 not recorded ({base.get('wheel_why', 'no wheel file kept')})"))
        return out

    @staticmethod
    def tracked(base, site):
        return [f for f in PipWheel.record(base, site) if not f.startswith("..")]

    @staticmethod
    def ignored(site, paths):
        return {p for p in paths if "__pycache__" in p.split("/") or p.endswith(".pyc")}


BASES = {"git_commit": GitCommit, "pip_version_wheel_hash": PipWheel}


def load_spec(path):
    """mods.json with its modules expanded. A module is the code unit: one patch on one extension point, which applies
    alone on the base. A row that names a module is a feature of it (a switch, its check, label and evidence) and takes
    the module's patch, files, lines, budget and static, so every reader of a row keeps working. Returns (spec, bad)."""
    spec = json.load(open(path))
    units, bad = spec.get("modules", []), []
    by = {u.get("name"): u for u in units}
    bad += [f"module {u.get('name')}: keys must be {sorted(MODULE_KEYS - {'budget_raised'})} (+ budget_raised)" for u in units
            if set(u) - MODULE_KEYS or (MODULE_KEYS - {"budget_raised"}) - set(u)]
    bad += ["module names not unique"] * (len(by) != len(units))
    bad += [f"module {u['name']}: title must be one line of 1-{TITLE_MAX} characters" for u in units
            if not isinstance(u.get("title"), str) or not u["title"].strip() or "\n" in u["title"] or len(u["title"]) > TITLE_MAX]
    order = []
    for m in spec.get("mods", []) + spec.get("parked", []):
        if "module" not in m:
            bad += [f"{m['name']}: a folder with modules has every series row in one"] * bool(units and m in spec.get("mods", []))
            continue
        u = by.get(m["module"])
        if not u or set(m) & set(MODULE_FACTS):
            bad.append(f"{m['name']}: " + (f"module {m['module']!r} is not in modules" if not u else f"a feature row keeps {sorted(set(m) & set(MODULE_FACTS))} in its module"))
            m.update({"patch": "patches/none", "files": [], "lines": {"added": 0, "removed": 0}, "budget": {"added": 0, "removed": 0}, "static": {"hunks_removing_stock_lines": 0}})
            continue
        if m not in spec.get("mods", []):
            bad.append(f"{m['name']}: a parked row cannot be in a module (the module's patch is in every build): park a patch of its own")
        if (m.get("depends_on") or {}).get("apply"):
            bad.append(f"{m['name']}: a feature row has no depends_on.apply (its module applies alone on the base)")
        m.update({k: u[k] for k in MODULE_FACTS if k in u})
        order += [m["module"]] * (m in spec.get("mods", []) and m["module"] not in order[-1:])
    if units and order != [u["name"] for u in units]:
        bad.append(f"series rows grouped by module give {order}, modules lists {[u['name'] for u in units]}: same order, every module used")
    return spec, bad


def units_of(spec):
    """the series' code units in apply order: its modules as rows (name, patch, lines, budget, no apply dependency), or
    its mods when the folder has no modules."""
    if not spec.get("modules"):
        return spec.get("mods", [])
    return [{**u, "depends_on": {"apply": [], "run": []}} for u in spec["modules"]]


def same(a, b):
    if os.path.exists(a) != os.path.exists(b):
        return False
    return not os.path.exists(a) or open(a, "rb").read() == open(b, "rb").read()


def sw_name(m):
    return (m.get("switch") or {}).get("name")


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


class Lint:
    def __init__(self, reg):
        lint = reg.get("lint") or {}
        self.model = re.compile(lint["model_name_regex"]) if lint.get("model_name_regex") else None
        self.private = re.compile(lint["private_regex"]) if lint.get("private_regex") else None
        self.names = {e: [n.lower() for n in [e] + (row.get("aliases") or [])] for e, row in reg["engines"].items()}
        self.others = [n.lower() for n in lint.get("other_engines", [])]
        if not self.model or not self.private:
            say("FAIL", "lint    engines.json lint.model_name_regex and lint.private_regex are required")

    def foreign(self, engine, text, allow=()):
        t = text.lower()
        names = ([n for e, ns in self.names.items() if e != engine for n in ns] if engine else []) + self.others
        own = self.names.get(engine, []) + list(allow)
        return sorted({n for n in names if n not in own and re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", t)})

    def text(self, engine, text, models=True, allow=()):
        bad = []
        if models and self.model and (m := self.model.search(text)):
            bad.append(f"model name '{m.group(0)}'")
        if self.private and (m := self.private.search(text)):
            bad.append(f"private detail '{m.group(0)}'")
        if f := self.foreign(engine, text, allow):
            bad.append(f"another engine named: {f}")
        return bad


def check_row(m, spec, engine_row, lint, rel):
    bad = []
    extra = set(m) - ROW_KEYS
    if extra:
        bad.append(f"unknown keys {sorted(extra)}")
    if m.get("engine") != spec["engine"] or m.get("base") != spec["version"]:
        bad.append("row engine/base differ from the folder")
    out = m.get("output") or {}
    ev = out.get("evidence") or {}
    lab = out.get("label")
    if lab not in LABELS:
        bad.append(f"output label {lab!r} not in {sorted(LABELS)}")
    if set(ev) - EVID_KEYS:
        bad.append(f"unknown evidence keys {sorted(set(ev) - EVID_KEYS)}")
    if not ev.get("numbers"):
        bad.append("output.evidence.numbers empty: say what was measured")
    if lab in ("identical", "quality-gated") and not ev.get("run"):
        bad.append(f"{lab} needs output.evidence.run")
    if lab == "identical":
        mt = ev.get("match") or {}
        if not (mt.get("what") and isinstance(mt.get("of"), int) and mt["of"] > 0 and mt.get("equal") == mt["of"]):
            bad.append("identical needs output.evidence.match {what, equal, of} with equal == of > 0")
    if lab == "quality-gated":
        g = ev.get("gate") or {}
        if not (g.get("name") and g.get("run") and str(g.get("verdict", "")).startswith("pass")):
            bad.append("quality-gated needs output.evidence.gate {name, run, verdict: pass...}")
    if m.get("gain") is not None and set(m["gain"]) - GAIN_KEYS:
        bad.append(f"unknown gain keys {sorted(set(m['gain']) - GAIN_KEYS)}")
    need = set()
    if m.get("check") is None:
        need.add("check")
    if m.get("switch") is None:
        need.add("switch")
        if not m.get("switch_null_why"):
            bad.append("switch null without switch_null_why")
    elif m["switch"].get("default") == "on":
        need.add("default_off")
    have = {e.get("rule") for e in m.get("rule_exceptions", []) if isinstance(e, dict) and e.get("why")}
    if need - have:
        bad.append(f"rule_exceptions missing {sorted(need - have)}")
    if m.get("switch") and m["switch"].get("kind") not in (engine_row.get("switch_kinds") or {}):
        bad.append(f"switch kind {m['switch'].get('kind')} not in the engine's switch_kinds")
    bad += default_run_bad(m, spec)
    if not m.get("marker"):
        bad.append("no marker")
    title = m.get("title")
    if title is not None and (not isinstance(title, str) or not title.strip() or "\n" in title or len(title) > TITLE_MAX):
        bad.append(f"title must be one line of 1-{TITLE_MAX} characters")
    bad += [f"row text: {x}" for s in strings({k: v for k, v in m.items() if k not in ("files", "patch", "scrubbed_from")}) for x in lint.text(spec["engine"], s)][:3]
    return bad


DEFAULT_RUN = ("on", "off")  # switch.default_run: what a consumer that sets switches for a build does with this one (engines.json rules)
WHERE_FACTS = ("compute_capability", "sm_count")  # switch.default_run_where: the device facts an "on" covers (kernel manifests' sm_count)


def default_run_bad(m, spec):
    """switch.default_run: once any row of a version states it, every switched series or parked row does, with its why;
    "on" only for an identical row (a consumer never turns on by itself a switch whose output may differ)."""
    sw = m.get("switch") or {}
    stated = any("default_run" in (r.get("switch") or {}) for r in spec.get("mods", []) + spec.get("parked", []))
    if not m.get("switch"):
        return ["default_run on a row with no switch"] if "default_run" in sw else []
    if "default_run" not in sw:  # a retired row is linted, never run: it need not state it
        live = any(r is m for r in spec.get("mods", []) + spec.get("parked", []))
        return ["switch.default_run missing (another row of this version states it)"] if stated and live else []
    bad = [] if sw["default_run"] in DEFAULT_RUN else [f"switch.default_run {sw['default_run']!r} not one of {list(DEFAULT_RUN)}"]
    if not str(sw.get("default_run_why") or "").strip():
        bad.append("switch.default_run without switch.default_run_why")
    if sw["default_run"] == "on" and (m.get("output") or {}).get("label") != "identical":
        bad.append("switch.default_run on for a row that is not identical")
    if "default_run_where" in sw:
        w = sw["default_run_where"]
        if sw["default_run"] != "on":
            bad.append("switch.default_run_where on a row whose default_run is not on")
        if not isinstance(w, dict) or not w or set(w) - set(WHERE_FACTS) or not all(isinstance(v, int) and not isinstance(v, bool) for v in w.values()):
            bad.append(f"switch.default_run_where must name one or more of {list(WHERE_FACTS)} with integer values")
    return bad


def retired_version(engine_row, version):
    """the engines.json retired_versions entry {date, why} of a version, or None (an active version)."""
    return (engine_row.get("retired_versions") or {}).get(version)


def retired_bad(r):
    if not isinstance(r, dict) or set(r) != RETIRED_KEYS or not DATE_RE.match(str(r.get("date", ""))) or not str(r.get("why") or "").strip():
        return ["needs exactly {date: YYYY-MM-DD, why}"]
    return []


def present(engine_row, kind, name, text):
    tmpl = (engine_row.get("switch_presence") or {}).get(kind)
    if not tmpl:
        return None
    return any(t.format(name=name) in text for t in tmpl)


def check_folder(folder, engine_row, reg, lint, ledger, strict):
    rel = os.path.relpath(folder, ROOT)
    print(f"\n{rel}")
    spec, mbad = load_spec(os.path.join(folder, "mods.json"))
    if set(spec) - SPEC_KEYS:
        say("FAIL", f"spec    unknown keys {sorted(set(spec) - SPEC_KEYS)}")
    for b in mbad:
        say("FAIL", f"module  {b}")
    if mbad:
        return
    rows = spec.get("mods", []) + spec.get("parked", [])
    retired = spec.get("retired", [])
    retired_v = retired_version(engine_row, spec["version"])
    names = [m["name"] for m in rows]
    allnames = names + [m["name"] for m in retired] + [z["name"] for z in spec.get("archived", [])]
    if len(set(allnames)) != len(allnames):
        say("FAIL", f"names   not unique in {rel}: {sorted({n for n in allnames if allnames.count(n) > 1})}")
    if spec.get("engine") != os.path.basename(os.path.dirname(folder)) or spec.get("version") != os.path.basename(folder):
        say("FAIL", f"spec    engine/version {spec.get('engine')}/{spec.get('version')} differ from the folder")
    check_files(folder, spec)
    check_archive(folder, spec, lint)
    ledger_sha = {e["sha256"]: e["id"] for e in ledger}
    ledger_sw = {e["switch"]: e["id"] for e in ledger if e.get("switch")}
    if not rows and not retired:
        say("ok", "archive only: no series, nothing to apply")
        return
    kind = engine_row.get("applier")
    bkind = (engine_row.get("base_identity") or {}).get("kind")
    for key in ("switch_literal_regex", "env_read_regex", "switch_presence"):
        if key not in engine_row:
            say("FAIL", f"engine  engines.json row for {spec['engine']} has no {key}")
    if kind not in APPLIERS or not reg["appliers"].get(kind, {}).get("implemented") or bkind not in BASES:
        say("FAIL", f"engine  no applier or base check for this engine (applier {kind}, base {bkind})")
        return
    units = units_of(spec)
    info = {m["name"]: parse(os.path.join(folder, m["patch"])) for m in rows + retired}
    path = lambda m: os.path.join(folder, m["patch"])
    by_name = {m["name"]: m for m in rows + retired}
    old = head_json(f"{rel}/mods.json")
    old_rows = {m["name"]: m for k in ("mods", "parked", "retired") for m in (old or {}).get(k, []) if "module" not in m} if old else {}
    old_units = {u["name"]: u for u in (old or {}).get("modules", [])}
    peers = lambda m: [x for x in rows if m.get("module") and x.get("module") == m["module"]]  # the rows one patch serves
    if GIT is None:
        say("SKIP", "history no git history here: budget changes against HEAD unchecked")

    def closure(n, seen):
        for d in by_name[n]["depends_on"]["apply"]:
            if d in by_name and d not in seen:
                closure(d, seen)
                seen.append(d)
        return seen

    # 4 lines and budgets, 5 switches, 5b rows (no checkout needed)
    rule = BUDGET_RULES.get(spec.get("budget_rule"))
    if not rule:
        say("FAIL", f"budget  budget_rule {spec.get('budget_rule')!r} not in {sorted(BUDGET_RULES)}")
    regexes = [re.compile(engine_row[k]) for k in ("switch_literal_regex", "env_read_regex") if engine_row.get(k)]
    tot_a = tot_r = 0
    retired_names = {m["name"] for m in retired}
    for m in rows + retired:
        files, a, r, mod, text, _ = info[m["name"]]
        hdr = header(path(m))
        series = m in spec["mods"]
        first = not m.get("module") or peers(m)[0] is m  # a module's lines are checked and counted once, at its first row
        tot_a, tot_r = tot_a + a * series * first, tot_r + r * series * first
        b = m["budget"]
        bad = []
        if (a, r) != (m["lines"]["added"], m["lines"]["removed"]):
            bad.append(f"row says +{m['lines']['added']} -{m['lines']['removed']}")
        if a > b["added"] or r > b["removed"]:
            bad.append("OVER BUDGET")
        raised = (m.get("budget_raised") or [])
        last = raised[-1] if raised else None
        if rule and (b["added"], b["removed"]) != (rule(m["lines"]["added"]), m["lines"]["removed"]):
            if not (last and last.get("to") == b and last.get("why") and last.get("date")):
                bad.append(f"budget {b} is not budget_rule {spec.get('budget_rule')} of the lines and no budget_raised entry ends at it")
        o = old_units.get(m["module"]) if m.get("module") else old_rows.get(m["name"])
        if o and (o["lines"] != m["lines"] or o["budget"] != m["budget"]):
            if not (last and last.get("from") == o["budget"] and last.get("to") == m["budget"] and last.get("why") and last.get("date")):
                bad.append(f"lines/budget changed against HEAD ({o['lines']} {o['budget']} -> {m['lines']} {m['budget']}) without budget_raised {{from, to, why, date}}")
        if sorted(files) != sorted(m["files"]):
            bad.append("file list differs from the row")
        if first:
            say("FAIL" if bad else "ok", f"lines   {m.get('module', m['name']):22} +{a:<4} -{r:<3} budget +{b['added']:<4} -{b['removed']:<3} files {len(files)}"
                + (f" (module: {', '.join(x['name'] for x in peers(m))})" if m.get("module") else "") + ("  " + "; ".join(bad) if bad else ""))
        sw, ck = sw_name(m), (m.get("check") or {}).get("name")
        bad = check_row(m, spec, engine_row, lint, rel) + ["a series mod needs a title"] * (m in spec.get("mods", []) and not m.get("title"))
        if m in retired:
            bad += [f"retired: {x}" for x in retired_bad(m.get("retired"))]
            bad += [f"a retired patch lives in retired/, not {m['patch']}"] * (not m["patch"].startswith("retired/"))
        else:
            bad += ["only a row in the retired list has a retired entry"] * ("retired" in m)
            bad += [f"depends on retired {sorted(set(m['depends_on']['apply'] + m['depends_on']['run']) & retired_names)}: "
                    f"un-retire it or drop the dependency"] * bool(set(m["depends_on"]["apply"] + m["depends_on"]["run"]) & retired_names)
        for nm, k in ((sw, (m.get("switch") or {}).get("kind")), (ck, (m.get("check") or {}).get("kind"))):
            if nm:
                p = present(engine_row, k, nm, text)
                if p is None:
                    bad.append(f"no switch_presence template for kind {k}")
                elif not p:
                    bad.append(f"{nm} not in the added lines")
        if m.get("marker") and m["marker"] not in text:
            bad.append(f"marker {m['marker']!r} not in the added lines")
        allowed = {sw, ck} | set(m.get("reads", [])) | set((m.get("check") or {}).get("needs", []))
        for x in peers(m):  # the patch is the module's: a literal is declared when one of its rows declares it
            allowed |= {sw_name(x), (x.get("check") or {}).get("name")} | set(x.get("reads", [])) | set((x.get("check") or {}).get("needs", []))
        for d in closure(m["name"], []):
            allowed |= {sw_name(by_name[d]), (by_name[d].get("check") or {}).get("name")}
        found = {x if isinstance(x, str) else next((y for y in x if y), '') for rx in regexes for x in rx.findall(text)} - {''}
        if found - allowed:
            bad.append(f"undeclared switch-shaped or env-read literals: {sorted(found - allowed)}")
        h = sha256(path(m))
        if h in ledger_sha:
            bad.append(f"sha256 equals archived {ledger_sha[h]}")
        if sw in ledger_sw:
            bad.append(f"switch {sw} belongs to archived {ledger_sw[sw]}")
        bad += [f"archived switch {x} is in the patch" for x in ledger_sw if f'"{x}"' in text or f"'{x}'" in text]
        bad += [f"patch header or added lines: {x}" for x in lint.text(spec["engine"], hdr + text)]
        if mod != m["static"]["hunks_removing_stock_lines"]:
            bad.append(f"{mod} hunks remove stock lines, row says {m['static']['hunks_removing_stock_lines']}")
        swtxt = f"no switch ({m.get('switch_null_why', '').split(';')[0]})" if sw is None else sw + (" (default ON)" if m["switch"].get("default") == "on" else "")
        note = "no check mode" if not ck else f"check {ck}"
        guard = "additive only" if mod == 0 else f"{mod} hunks rewrite stock lines"
        exc = f"; exceptions: {', '.join(e['rule'] for e in m.get('rule_exceptions', []))}" if m.get("rule_exceptions") else ""
        say("FAIL" if bad else "ok", f"switch  {m['name']:22} {swtxt}; {note}; {guard}; output {m['output']['label']}{exc}" + ("  " + "; ".join(bad) if bad else ""))
    print(f"  total   series: {len(units)} patches" + (f" (modules), {len(spec['mods'])} features" if spec.get("modules") else "") + f", +{tot_a} -{tot_r} lines; parked: {len(spec.get('parked', []))}; "
          f"retired: {len(retired)}; archived: {len(spec.get('archived', []))}")
    if retired_v:
        say("ok", f"retired version ({retired_v.get('date') if isinstance(retired_v, dict) else '?'}): linted only; not applied, not in the stack measurement, not in "
                  f"index.json's current lists")
        if os.path.exists(os.path.join(folder, "stack_budget.json")):
            say("FAIL", "stack   a retired version keeps no stack_budget.json (git history has it; un-retiring derives caps again)")
        return
    if not rows:
        return

    for t in spec.get("tested_trees", []):
        gone = [p for p in t.get("apply", []) if not os.path.isfile(os.path.join(folder, p))]
        if gone:
            say("FAIL", f"tree    {t.get('id')}: apply lists {gone}, not files of this folder (a retired or archived patch moved: update the path)")
    if not spec.get("tested_trees") and not (spec.get("tested_trees_none") or {}).get("why"):
        say("FAIL", "tree    a folder with mods needs tested_trees or tested_trees_none {why}")

    # 10 stack (reads the base checkout's text when there is one, to place hunks exactly)
    touched = sorted({f for i in info.values() for f in i[0]})
    checkout = checkout_of(spec["engine"], spec["version"])
    check_stack(folder, spec, checkout if checkout and os.path.isdir(checkout) else None, head_json(f"{rel}/stack_budget.json"))

    # needs the base checkout from here on
    if not checkout or not os.path.isdir(checkout):
        say("SKIP", f"base    setup: no base checkout for {spec['engine']}/{spec['version']} (local.json checkouts{': ' + checkout if checkout else ''}); apply, deps and tree checks not run")
        return
    B = BASES[bkind]
    res = B.check(spec["base"], checkout, touched)
    for tag, msg in res:
        say(tag, msg)
    if any(t == "FAIL" for t, _ in res):
        return
    ap = APPLIERS[kind](checkout, touched)

    # 2 series, in order, then revert
    tmp, applied = ap.fresh(), []
    for m in units:
        ok, err = ap.apply(tmp, path(m))
        say("ok" if ok else "FAIL", f"apply   {m['patch']}" + ("" if ok else f": {err[0]}"))
        if not ok:
            break
        applied.append(m)
    else:
        for m in reversed(applied):
            ap.apply(tmp, path(m), revert=True)
        back = [f for f in touched if not same(os.path.join(tmp, f), os.path.join(checkout, f))]
        say("FAIL" if back else "ok", f"revert  {'files differ from the base after reverting: ' + ', '.join(back) if back else 'series reverts to the base bytes'}")
    shutil.rmtree(tmp)

    # 3 depends_on.apply is sufficient
    for m in rows:
        bad = [d for d in m["depends_on"]["apply"] + m["depends_on"]["run"] if d not in names or names.index(d) >= names.index(m["name"])]
        if bad:
            say("FAIL", f"deps    {m['name']}: depends on {bad}, not an earlier mod")
            continue
        if m.get("module"):
            continue  # its module applies alone on the base (below)
        chain = (list(names[:len(spec["mods"])]) if m.get("applies_after") else closure(m["name"], [])) + [m["name"]]
        tmp = ap.fresh()
        res = [ap.apply(tmp, path(by_name[n])) for n in chain]
        shutil.rmtree(tmp)
        say("ok" if all(r[0] for r in res) else "FAIL", f"deps    {m['name']} applies on base + {chain[:-1] or 'nothing'}")

    for u in units if spec.get("modules") else []:
        tmp = ap.fresh()
        ok, err = ap.apply(tmp, path(u))
        shutil.rmtree(tmp)
        say("ok" if ok else "FAIL", f"deps    module {u['name']} applies alone on the base" + ("" if ok else f": {err[0]}"))

    # 8 also_applies_to: each series and parked patch alone on the other base
    for t in spec.get("also_applies_to", []):
        other = checkout_of(spec["engine"], t["version"])
        if not other or not os.path.isdir(other):
            say("SKIP", f"also    setup: no checkout for {spec['engine']}/{t['version']} in local.json")
            continue
        bad = []
        for m in rows:
            a2 = APPLIERS[kind](other, info[m["name"]][0])
            tmp = a2.fresh()
            if not a2.apply(tmp, path(m))[0]:
                bad.append(m["name"])
            shutil.rmtree(tmp)
        say("FAIL" if bad else "ok", f"also    each of {len(rows)} patches applies alone on {spec['engine']}/{t['version']}" + (f": not {bad}" if bad else ""))

    # 6 byte equality with the tested trees
    if not spec.get("tested_trees"):
        say("SKIP", f"tree    no tested tree: {(spec.get('tested_trees_none') or {}).get('why', 'none recorded')}")
    tracked = None
    for t in spec.get("tested_trees", []):
        tree = local_path(t["path"])
        if not tree or not os.path.exists(tree):
            say("SKIP", f"tree    {t['id']} ({t['path']}): not on this machine (local.json roots)")
            continue
        patches = [os.path.join(folder, p) for p in t["apply"]]
        files = sorted({f for p in patches for f in parse(p)[0]})
        tmp = APPLIERS[kind](checkout, files).fresh()
        res = [(p, ap.apply(tmp, p)) for p in patches]
        badp = [os.path.basename(p) for p, (ok, _) in res if not ok]
        diff = [f for f in files if not same(os.path.join(tmp, f), os.path.join(tree, f))]
        shutil.rmtree(tmp)
        tracked = tracked if tracked is not None else B.tracked(spec["base"], checkout)
        known = set(tracked) | set(files)
        other = [f for f in tracked if f not in files and not same(os.path.join(checkout, f), os.path.join(tree, f))]
        on_disk = []
        for d, dirs, fs in os.walk(tree):
            dirs[:] = [x for x in dirs if x != ".git"]
            on_disk += [os.path.relpath(os.path.join(d, f), tree) for f in fs]
        cand = [f for f in on_disk if f not in known]
        extra = sorted(set(cand) - B.ignored(checkout, cand))
        if other:
            say("FAIL", f"tree    {t['id']}: {len(other)} base files no patch touches differ: {', '.join(other[:8])}")
        if extra:
            say("FAIL", f"tree    {t['id']}: {len(extra)} files the base neither has nor ignores: {', '.join(extra[:8])}")
        if badp:
            say("FAIL", f"tree    {t['id']}: did not apply: {badp}")
        elif diff:
            say("FAIL", f"tree    {t['id']}: {len(diff)} of {len(files)} files differ: {', '.join(diff)}")
        elif not other and not extra:
            say("ok", f"tree    {t['id']}: byte-equal on the {len(files)} touched files ({t['contains']}); the other {len(tracked) - len(set(files) & set(tracked))} base files equal; no extra files")


# 10 stack: the health of the whole series, not one patch. A cap per metric in stack_budget.json beside mods.json.

STACK_METRICS = {  # name: (what, the refactor a crossing suggests)
    "total_lines": ("added + removed lines over the series",
                    "fold mods that patch the same code into one extension point (a hook patch) plus rows, or archive a weak mod"),
    "mods_per_file": ("the most mods that patch one engine file",
                      "introduce one extension point in that file (a hook patch) and turn the mods into rows it reads"),
    "overlap_hunks_per_file": ("the most hunks in one file whose lines (context included) another mod's hunk already covers",
                               "the mods edit the same region: replace them with one hook there plus rows"),
    "depth": ("the longest depends_on.apply chain, in mods",
              "a mod that needs another mod's code to apply is a hook in disguise: make the base mod an extension point"),
    "stacked_hunks": ("hunks whose context or removed lines are another mod's added lines (mods stacked on mods)",
                      "each one breaks when the mod under it changes: merge the pair or expose what the upper mod needs as a hook"),
}
STACK_KEYS = {"schema", "about", "margin_rule", "caps"}
CAP_KEYS = {"cap", "set", "raised"}
SET_KEYS = {"measured", "mods", "date", "after_raises"}  # after_raises: the raised entries before a re-derived (lowered) set
RAISE_KEYS = {"from", "to", "measured", "why", "date", "approved"}
APPROVED_RE = re.compile(r"Owner-approved \d{4}-\d{2}-\d{2}$")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}$")


def stack_margin(measured, mods):
    """the cap a measurement sets: room for one more average mod. cap = measured + ceil(measured / series mods)."""
    return measured + (-(-measured // mods) if mods else 0)


def hunks(path):
    """[(file, [(old_start, old_len, new_start, new_len, function, [(kind, text)])])] in patch order; a hunk's body is
    read by its header's counts, so a removed line that starts with '-- ' is not taken for a file header."""
    out, cur, h, left, old = [], None, None, [0, 0], None
    with open(path, encoding="utf-8", errors="surrogateescape") as fh:
        text = fh.readlines()
    for line in text:
        if h is not None and line.startswith("\\"):  # '\ No newline at end of file': not a line, the hunk goes on
            continue
        if h is not None and (left[0] or left[1]) and (line[:1] in (" ", "+", "-") or line in ("\n", "\r\n")):
            k = line[0] if line[:1] in (" ", "+", "-") else " "  # an empty line is context whose space was stripped
            h[5].append((k, line[1:].rstrip("\n") if k == line[:1] else ""))
            left[0] -= k != "+"
            left[1] -= k != "-"
            continue
        h = None
        if line.startswith("--- "):
            old = line[4:].split("\t")[0].strip()
        elif line.startswith("+++ "):
            f = line[4:].split("\t")[0].strip()
            f = old if f == "/dev/null" else f  # a deleted file keeps its path
            cur = (f[2:] if f.startswith(("a/", "b/")) else f, [])
            out.append(cur)
        elif line.startswith("@@") and cur is not None:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)", line)
            h = (int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1), m.group(5).strip(), [])
            left = [h[1], h[3]]
            cur[1].append(h)
    return out


def stack_measure(mods, folder, base=None):
    """Apply the series' hunks to a model of each file: per line its text (when known), the mod that added it and the
    mods whose hunks covered it. Hunks are placed like git apply: at their line, else the nearest offset where every
    known line matches (the base checkout, when given, makes every base line known)."""
    model, complete, per_file, problems = {}, {}, {}, []
    stacked, by_func = [], {}
    for m in mods:
        for f, hs in hunks(os.path.join(folder, m["patch"])):
            if f not in model:
                p = os.path.join(base, f) if base else None
                whole = bool(p and os.path.isfile(p)) or (hs and hs[0][0] == 0 and hs[0][1] == 0)  # every line known
                known = []
                if p and os.path.isfile(p):
                    with open(p, encoding="utf-8", errors="surrogateescape") as fh:
                        known = fh.read().split("\n")
                if known and known[-1] == "":
                    known.pop()
                model[f] = [[t, None, set()] for t in known]
                complete[f] = whole
            lines = model[f]
            st = per_file.setdefault(f, {"mods": [], "overlap": 0})
            if m["name"] not in st["mods"]:
                st["mods"].append(m["name"])
            delta = shift = 0
            for os_, ol, ns, nl, func, body in hs:
                old = [(k, t) for k, t in body if k != "+"]
                want = (os_ - 1 if ol else os_) + delta + shift
                place = None
                for off in sorted(range(-len(lines) - 1, len(lines) + 2), key=abs):
                    q = want + off
                    if q < 0 or (complete[f] and q + len(old) > len(lines)):
                        continue
                    while len(lines) < q + len(old):
                        lines.append([None, None, set()])
                    if all(lines[q + i][0] is None or lines[q + i][0] == t for i, (_, t) in enumerate(old)):
                        place = q
                        break
                if place is None:
                    problems.append(f"{m['patch']}: cannot place the hunk at {f}:{os_} on the series before it")
                    continue
                shift += place - want
                region = lines[place:place + len(old)]
                others = {x for ln in region for x in ln[2]} - {m["name"]}
                owners = {ln[1] for ln in region} - {None, m["name"]}
                if others:
                    st["overlap"] += 1
                if owners:
                    stacked.append((m["name"], f, func, sorted(owners)))
                by_func.setdefault((f, func or "(file scope)"), set()).add(m["name"])
                new, i = [], 0
                for k, t in body:
                    if k == " ":
                        new.append([t, region[i][1], region[i][2] | {m["name"]}])
                    if k != "+":
                        i += 1
                    if k == "+":
                        new.append([t, m["name"], {m["name"]}])
                lines[place:place + len(old)] = new
                delta += len(new) - len(old)
    return per_file, stacked, by_func, problems


def stack_depth(mods):
    by = {m["name"]: m for m in mods}
    memo = {}

    def chain(n, seen=()):
        if n not in memo:
            deps = [d for d in by[n]["depends_on"]["apply"] if d in by and d not in seen]
            memo[n] = max((chain(d, seen + (n,)) for d in deps), key=len, default=[]) + [n]
        return memo[n]
    return max((chain(m["name"]) for m in mods), key=len, default=[])


def stack_values(spec, folder, base=None):
    mods = units_of(spec)
    per_file, stacked, by_func, problems = stack_measure(mods, folder, base)
    total = sum(m["lines"]["added"] + m["lines"]["removed"] for m in mods)
    hot = sorted(per_file.items(), key=lambda kv: (-len(kv[1]["mods"]), -kv[1]["overlap"], kv[0]))
    chain = stack_depth(mods)
    vals = {"total_lines": total,
            "mods_per_file": max((len(v["mods"]) for v in per_file.values()), default=0),
            "overlap_hunks_per_file": max((v["overlap"] for v in per_file.values()), default=0),
            "depth": len(chain),
            "stacked_hunks": len(stacked)}
    return vals, {"per_file": per_file, "hot": hot, "chain": chain, "stacked": stacked, "by_func": by_func, "problems": problems}


def stack_authorized(name, e):
    """the cap an entry may hold and the measurement it rests on, or the reasons it holds none: the last Owner-approved
    raise after the set, else the derived set. A set re-derived lower (the ratchet) records after_raises, the count of
    raised entries before it: those stay (append-only) as history and no longer chain."""
    bad = [f"{name}: unknown keys {sorted(set(e) - CAP_KEYS)}"] if set(e) - CAP_KEYS else []
    s, raised = e.get("set") or {}, e.get("raised") or []
    k = s.get("after_raises", 0)
    if set(s) - SET_KEYS or not isinstance(s.get("measured"), int) or not isinstance(s.get("mods"), int) or not DATE_RE.match(str(s.get("date", ""))) \
            or not isinstance(k, int) or not 0 <= k <= len(raised):
        return None, None, bad + [f"{name}: set needs {{measured, mods, date}} (the measurement the derived cap comes from), "
                                  f"after_raises 0..{len(raised)} when given"]
    cap, basis = stack_margin(s["measured"], s["mods"]), s["measured"]
    for r in raised[k:]:
        miss = [k for k in ("from", "to", "measured", "why", "date", "approved") if r.get(k) in (None, "")]
        if set(r) - RAISE_KEYS or miss:
            bad.append(f"{name}: a raise needs {{from, to, measured, why, date, approved}}" + (f" (missing {miss})" if miss else ""))
            continue
        if not APPROVED_RE.match(str(r["approved"])):
            bad.append(f"{name}: raise approved {r['approved']!r} is not 'Owner-approved YYYY-MM-DD'")
        if r["from"] != cap:
            bad.append(f"{name}: raise from {r['from']}, the cap before it is {cap}")
        if r["to"] != r["measured"]:
            bad.append(f"{name}: raise to {r['to']} is not its measured value {r['measured']} (a cap rises only to what was measured after review cuts)")
        cap, basis = r["to"], r["measured"]
    return cap, basis, bad


def stack_lower_hint(rel, lows, n, raised):
    """the exact values that lower drifted caps: set re-derived from today's measurement, cap its margin."""
    today = __import__("datetime").date.today().isoformat()
    return "; ".join(f"{rel}/stack_budget.json caps.{name}: \"cap\": {stack_margin(val, n)}, \"set\": "
                     + json.dumps({"measured": val, "mods": n, "date": today, **({"after_raises": len(raised[name])} if raised[name] else {})})
                     + f" (was cap {cap}, measured {basis})" for name, val, cap, basis in lows)


def stack_raise_hint(rel, name, val, cap):
    return (f"to accept it instead: the owner approves, then {rel}/stack_budget.json caps.{name}.raised gets "
            f"{{\"from\": {cap}, \"to\": {val}, \"measured\": {val}, \"why\": \"...\", \"date\": \"YYYY-MM-DD\", "
            f"\"approved\": \"Owner-approved YYYY-MM-DD\"}} and cap {val}")


def check_stack(folder, spec, base=None, head=None):
    """10 stack: per version folder, the series' total lines, file hot spots, dependency depth and mods stacked on
    mods against the caps in stack_budget.json; a cap rises only by an Owner-approved raise to the measured value."""
    rel = os.path.relpath(folder, ROOT)
    mods, n = units_of(spec), len(spec.get("mods", []))  # measured on the code units; the margin is one more average row
    p = os.path.join(folder, "stack_budget.json")
    if not mods:
        if os.path.exists(p):
            say("FAIL", "stack   stack_budget.json in a folder with no series")
        return
    vals, why = stack_values(spec, folder, base)
    for x in why["problems"]:
        say("FAIL", f"stack   {x}")
    if not os.path.exists(p):
        say("FAIL", f"stack   {rel}/stack_budget.json missing; derived caps for today's series (cap = measured + ceil(measured / "
                    f"{n} mods)): " + ", ".join(f"{k} {v} -> {stack_margin(v, n)}" for k, v in vals.items()))
        return
    try:
        with open(p) as fh:
            sb = json.load(fh)
    except ValueError as x:
        say("FAIL", f"stack   {rel}/stack_budget.json is not JSON: {x}")
        return
    caps = sb.get("caps") or {}
    bad = [f"unknown keys {sorted(set(sb) - STACK_KEYS)}"] * bool(set(sb) - STACK_KEYS)
    bad += [f"caps must be exactly {sorted(STACK_METRICS)}"] * (set(caps) != set(STACK_METRICS))
    head_caps = (head or {}).get("caps") or {}
    lows = []
    for name in sorted(set(caps) & set(STACK_METRICS)):
        e = caps[name]
        auth, basis, b = stack_authorized(name, e)
        bad += b
        if auth is not None and e.get("cap") == auth and vals[name] < basis and stack_margin(vals[name], n) < auth:
            lows.append((name, vals[name], auth, basis))
        h = head_caps.get(name)
        s = e.get("set") or {}
        if s != (h or {}).get("set") and (s.get("measured"), s.get("mods")) != (vals[name], n):
            bad.append(f"{name}: set {{measured {s.get('measured')}, mods {s.get('mods')}}} is new against HEAD and is not "
                       f"today's measurement {{measured {vals[name]}, mods {n}}}")
        new = (e.get("raised") or [])[len((h or {}).get("raised") or []):]
        bad += [f"{name}: new raise measured {r.get('measured')} is not today's value {vals[name]}" for r in new
                if isinstance(r, dict) and r.get("measured") != vals[name]]
        if auth is not None and e.get("cap") != auth:
            bad.append(f"{name}: cap {e.get('cap')} is not the derived or last raised value {auth}")
        h = head_caps.get(name)
        if h and isinstance(e.get("cap"), int) and isinstance(h.get("cap"), int) and e["cap"] > h["cap"]:
            hr, r = h.get("raised") or [], e.get("raised") or []
            if not (len(r) > len(hr) and r[:len(hr)] == hr and r[-1].get("from") == h["cap"]):
                bad.append(f"{name}: cap {h['cap']} -> {e['cap']} against HEAD without a new Owner-approved raise from {h['cap']}")
        if h and (h.get("raised") or []) != (e.get("raised") or [])[:len(h.get("raised") or [])]:
            bad.append(f"{name}: raised entries are append-only against HEAD")
    for b in bad:
        say("FAIL", f"stack   {rel}/stack_budget.json {b}")
    if bad:
        return
    PLACED = ("overlap_hunks_per_file", "stacked_hunks")  # hunk placement: exact only with the base checkout
    if base is None and any(n in PLACED for n, *_ in lows):
        say("SKIP", f"stack   ratchet: {', '.join(n for n, *_ in lows if n in PLACED)} dropped, but without the base checkout hunk "
                    f"placement is approximate; run with the checkout to lower: " + stack_lower_hint(rel, [x for x in lows if x[0] in PLACED], n, {n: caps[n].get("raised") or [] for n in caps}))
        lows = [x for x in lows if x[0] not in PLACED]
    if lows:
        say("FAIL", f"stack   ratchet: {', '.join(n for n, *_ in lows)} dropped below the measurement the cap rests on (a mod "
                    f"retired or shrunk); freed room is not kept as headroom. Lower to measured + ceil(measured / {n} mods): "
                    + stack_lower_hint(rel, lows, n, {n: caps[n].get("raised") or [] for n in caps}))
    hot, top = why["hot"], why["hot"][0] if why["hot"] else (None, {"mods": []})

    def funcs(f):
        fs = sorted(((len(ms), fn, sorted(ms)) for (ff, fn), ms in why["by_func"].items() if ff == f and len(ms) > 1), reverse=True)
        return "; ".join(f"{n} mods patch {f}/{fn} ({', '.join(ms)})" for n, fn, ms in fs[:3]) or f"no one function: the mods spread over {f}"

    for name, (what, fix) in STACK_METRICS.items():
        val, cap = vals[name], caps[name]["cap"]
        if val <= cap:
            continue
        if name == "mods_per_file":
            over = [(f, v) for f, v in hot if len(v["mods"]) > cap]
            detail = " | ".join(f"refactor: {len(v['mods'])} mods patch {f} ({', '.join(v['mods'])}); {funcs(f)}; introduce an extension point" for f, v in over)
        elif name == "overlap_hunks_per_file":
            over = sorted(((f, v) for f, v in why["per_file"].items() if v["overlap"] > cap), key=lambda kv: -kv[1]["overlap"])
            detail = " | ".join(f"refactor: {v['overlap']} overlapping hunks in {f} from {len(v['mods'])} mods ({', '.join(v['mods'])}); {funcs(f)}; introduce an extension point" for f, v in over)
        elif name == "depth":
            detail = "chain " + " -> ".join(reversed(why["chain"])) + f"; make {why['chain'][0]} an extension point the others are rows of"
        elif name == "stacked_hunks":
            pairs = {}
            for upper, f, fn, lows in why["stacked"]:
                for lo in lows:
                    pairs.setdefault((upper, lo), set()).add(f"{f}/{fn or '(file scope)'}")
            detail = "; ".join(f"{u} on {lo} at {', '.join(sorted(w))}" for (u, lo), w in sorted(pairs.items()))
        else:
            detail = f"hottest file {top[0]} ({len(top[1]['mods'])} mods: {', '.join(top[1]['mods'])})" if top[0] else ""
        say("FAIL", f"stack   {name} {val} > cap {cap} ({what}). Suggested refactor: {fix}. {detail}. {stack_raise_hint(rel, name, val, cap)}")
    # the summary: printed on every run, never fails
    print(f"  stack health (" + (f"{len(mods)} modules, " if spec.get("modules") else "") + f"{n} series mods; cap = measured + ceil(measured / mods) unless Owner-approved)")
    for name in STACK_METRICS:
        val, cap = vals[name], caps[name]["cap"]
        print(f"    {name:23} {val:>6} / cap {cap:<6} headroom {cap - val}")
    for f, v in hot[:3]:
        print(f"    hot spot  {f}: {len(v['mods'])} mods ({', '.join(v['mods'])}), {v['overlap']} overlapping hunks")
    print(f"    deepest   {' -> '.join(reversed(why['chain'])) if why['chain'] else 'none'}")
    tight = min(mods, key=lambda m: m["budget"]["added"] - m["lines"]["added"])
    print(f"    per-mod   budgets checked under lines (budget_rule); tightest {tight['name']} +{tight['lines']['added']} of +{tight['budget']['added']}")
    if not any(vals[n] > caps[n]["cap"] for n in STACK_METRICS):
        say("ok", f"stack   every metric within its cap")


def check_archive(folder, spec, lint):
    for z in spec.get("archived", []):
        p = os.path.join(folder, z["patch"])
        bad = [f"unknown keys {sorted(set(z) - ARCH_KEYS)}"] if set(z) - ARCH_KEYS else []
        if not z["patch"].startswith("archive/"):
            bad.append(f"patch {z['patch']} is not under archive/")
        if not os.path.exists(p):
            say("FAIL", f"archive {z['name']}: {z['patch']} missing")
            continue
        files, a, r, _, text, _ = parse(p)
        hdr = header(p)
        if sha256(p) != z["sha256"]:
            bad.append(f"sha256 {sha256(p)[:12]}, row says {z['sha256'][:12]}")
        if (a, r) != (z["lines"]["added"], z["lines"]["removed"]):
            bad.append(f"+{a} -{r}, row says +{z['lines']['added']} -{z['lines']['removed']}")
        bad += [f"row text: {x}" for s in strings(z) for x in lint.text(spec["engine"], s)][:3]
        bad += [f"patch: {x}" for x in lint.text(spec["engine"], hdr + text)]
        say("FAIL" if bad else "ok", f"archive {z['name']:22} +{a:<4} -{r:<3} sha256 {z['sha256'][:12]} ({z['kind']}, {z['date']})" + ("  " + "; ".join(bad) if bad else ""))


def check_files(folder, spec):
    rows = spec.get("mods", []) + spec.get("parked", []) + spec.get("retired", []) + spec.get("archived", [])
    known = {m["patch"] for m in rows} | {t["path"] for t in spec.get("tools", [])} | {"archive/README.md"} | \
        {os.path.join(k["dir"], f) for k in spec.get("kernels", []) for f in k["files"]}
    for e in sorted(os.listdir(folder)):
        if e not in VERSION_ENTRIES:
            say("FAIL", f"files   {e} is not part of a version folder ({sorted(VERSION_ENTRIES)})")
    for sub in ("patches", "parked", "archive", "retired", "tools", "kernels"):
        d = os.path.join(folder, sub)
        for dd, dirs, fs in os.walk(d):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for f in fs:
                r = os.path.relpath(os.path.join(dd, f), folder)
                if r not in known:
                    say("FAIL", f"files   {r} has no row in mods.json")
    check_kernels(folder, spec)
    for t in spec.get("tools", []):
        p = os.path.join(folder, t["path"])
        ok = os.path.exists(p) and sha256(p) == t["sha256"]
        say("ok" if ok else "FAIL", f"tool    {t['path']} sha256 {t['sha256'][:12]}" + ("" if ok else ": missing or changed"))


def check_kernels(folder, spec, old=None):
    """kernels/<name>/: an engine kernel kept as an artifact (a source, its manifest entry for an install hook mod, and
    built binaries if kept), with the mods' rules: the hook mod's switch turns it on and its check mode checks it; an
    output label with evidence (identical needs a counted match); a line budget on the source; every file's sha256;
    a change of lines or budget against HEAD (old: its mods.json) needs a budget_raised entry {from, to, why, date}."""
    by = {m["name"]: m for m in spec.get("mods", [])}
    old = head_json(f"{os.path.relpath(folder, ROOT)}/mods.json") if old is None else old
    old_k = {k["name"]: k for k in (old or {}).get("kernels", [])}
    for k in spec.get("kernels", []):
        bad = [f"unknown keys {sorted(set(k) - KERNEL_KEYS)}"] if set(k) - KERNEL_KEYS else []
        if not k["dir"].startswith("kernels/") or k["dir"].rstrip("/").split("/")[-1] != k["name"]:
            bad.append("dir must be kernels/<name>/")
        if k["hook"] not in by:
            bad.append(f"hook {k['hook']} is not a series mod of this version")
        for f, h in k["files"].items():
            p = os.path.join(folder, k["dir"], f)
            if not os.path.exists(p) or sha256(p) != h:
                bad.append(f"{f} missing or changed")
        srcs = [k["source"]] if isinstance(k["source"], str) else k["source"]   # a manifest may install several kernels
        paths = [os.path.join(folder, k["dir"], x) for x in srcs]
        n = sum(sum(1 for _ in open(x, encoding="utf-8", errors="replace")) for x in paths) if paths and all(map(os.path.exists, paths)) else -1
        if any(x not in k["files"] for x in srcs):
            bad.append("every source must be one of the row's files")
        if n != k["lines"] or n > k["budget"]:
            bad.append(f"source {n} lines, row says {k['lines']}, budget {k['budget']}")
        o, last = old_k.get(k["name"]), (k.get("budget_raised") or [None])[-1]
        if o and (o["lines"], o["budget"]) != (k["lines"], k["budget"]):
            if not (isinstance(last, dict) and last.get("from") == o["budget"] and last.get("to") == k["budget"] and last.get("why") and last.get("date")):
                bad.append(f"lines/budget changed against HEAD ({o['lines']} {o['budget']} -> {k['lines']} {k['budget']}) without budget_raised {{from, to, why, date}}")
        ev = k["output"].get("evidence") or {}
        if k["output"].get("label") not in LABELS or not ev.get("numbers") or (k["output"]["label"] == "identical" and not (ev.get("match") or {}).get("of")):
            bad.append("output label with evidence (identical: a counted match)")
        say("FAIL" if bad else "ok", f"kernel  {k['name']:22} via {k['hook']}" + ("  " + "; ".join(bad) if bad else ""))


def benchmark_files(bad=None):
    """benchmarks/<name>/ holds README.md and .json files only (a benchmark record: method, results, summaries).
    A file named vs-<engine>.md or vs-<engine>.json is a comparison against that one engine; files in a folder that
    holds one may name that engine (so the folder's README can link it), and no other."""
    top, out = os.path.join(ROOT, "benchmarks"), []
    for d in sorted(os.listdir(top)) if os.path.isdir(top) else []:
        if not os.path.isdir(os.path.join(top, d)):
            bad is not None and bad.append(f"benchmarks/{d} is not a benchmark folder")
            continue
        names = sorted(os.listdir(os.path.join(top, d)))
        if "README.md" not in names and bad is not None:
            bad.append(f"benchmarks/{d} has no README.md")
        for f in names:
            if f in ("README.md",) or f.endswith(".json") or (f.startswith("vs-") and f.endswith(".md")):
                out.append(f"benchmarks/{d}/{f}")
            elif bad is not None:
                bad.append(f"benchmarks/{d}/{f} is not README.md, vs-<engine>.md or .json")
    return out


def check_docs(reg, lint):
    """README.md, DETAILS.md, LICENSE, engines/README.md and every folder-level string of mods.json: no private detail, no engine
    outside engines.json (lint.doc_allow lists exact lines that may still match). Model names are allowed in README.md
    and benchmarks/ (owner, 2026-11-17: they describe what was measured); mods, patches and mods.json stay model-agnostic."""
    print("\ndocs")
    allow = set((reg.get("lint") or {}).get("doc_allow", []))
    # the top README may link a published comparison: it may name an engine that has a benchmarks/*/vs-<engine> file
    compared = sorted({os.path.basename(r)[3:].rsplit(".", 1)[0].lower() for r in benchmark_files()
                       if os.path.basename(r).startswith("vs-")})
    n = 0
    for rel in ("README.md", "DETAILS.md", "LICENSE", "engines/README.md"):
        p = os.path.join(ROOT, rel)
        for i, line in enumerate(open(p, encoding="utf-8") if os.path.exists(p) else [], 1):
            if line.strip() in allow:
                continue
            for x in lint.text(None, line, models=rel == "LICENSE" or rel.startswith("engines/"),
                               allow=compared if rel == "README.md" else ()):
                say("FAIL", f"docs    {rel}:{i}: {x}")
                n += 1
    for rel in benchmark_files():
        p = os.path.join(ROOT, rel)
        lines = [s_ for s_ in strings(json.load(open(p)))] if rel.endswith(".json") else open(p, encoding="utf-8")
        here = os.path.dirname(p)
        allow = sorted({f[3:].rsplit(".", 1)[0].lower() for f in os.listdir(here) if f.startswith("vs-")})
        for i, line in enumerate(lines, 1):
            for x in lint.text(None, line, models=False, allow=allow):
                say("FAIL", f"docs    {rel}:{i}: {x}")
                n += 1
    for f in folders_all():
        spec, _ = load_spec(os.path.join(f, "mods.json"))
        rest = {k: v for k, v in spec.items() if k not in ("mods", "parked", "archived", "retired")}
        for s_ in strings(rest):
            for x in lint.text(spec["engine"], s_):
                say("FAIL", f"docs    {os.path.relpath(f, ROOT)}/mods.json (folder fields): {x}")
                n += 1
    if not n:
        say("ok", "docs    README.md, DETAILS.md, LICENSE, engines/README.md, benchmarks/ and folder fields pass the lint")


def check_layout(reg):
    print("\nlayout")
    bad = []
    for e in sorted(os.listdir(ROOT)):
        if e not in TOP_FILES:
            bad.append(f"top-level {e} is not part of the repo layout")
    for e in sorted(os.listdir(os.path.join(ROOT, "tools"))):
        if e not in ("check.py", "test_check.py", "__pycache__"):
            bad.append(f"tools/{e} is not part of the repo layout")
    benchmark_files(bad)
    top = os.path.join(ROOT, "engines")
    for e in sorted(os.listdir(top)):
        d = os.path.join(top, e)
        if not os.path.isdir(d):
            if e != "README.md":
                bad.append(f"engines/{e} is not an engine folder")
            continue
        if e not in reg["engines"]:
            bad.append(f"engines/{e} has no row in engines.json")
            continue
        vs = sorted(v for v in os.listdir(d))
        for v in vs:
            if not os.path.isfile(os.path.join(d, v, "mods.json")):
                bad.append(f"engines/{e}/{v} has no mods.json")
        if sorted(reg["engines"][e].get("versions", [])) != vs:
            bad.append(f"engines.json {e} versions {sorted(reg['engines'][e].get('versions', []))} != folders {vs}")
        for v, r in (reg["engines"][e].get("retired_versions") or {}).items():
            bad += [f"engines.json {e} retired_versions {v} is not one of its versions"] * (v not in vs)
            bad += [f"engines.json {e} retired_versions {v}: {x}" for x in retired_bad(r)]
    for e, row in reg["engines"].items():
        if row.get("versions") and not os.path.isdir(os.path.join(top, e)):
            bad.append(f"engines.json {e} lists versions but engines/{e} does not exist")
    for b in bad:
        say("FAIL", "layout  " + b)
    if not bad:
        say("ok", "layout  every folder and file is known; engines.json versions equal the folders")


# 9 exports: generated from the rows, never edited by hand

def folders_all():
    top = os.path.join(ROOT, "engines")
    return sorted(os.path.join(top, e, v) for e in os.listdir(top) if os.path.isdir(os.path.join(top, e))
                  for v in os.listdir(os.path.join(top, e)) if os.path.exists(os.path.join(top, e, v, "mods.json")))


REQUIRES_KEYS = {"boltbeam", "why"}  # engines.json requires: the oldest consumer version that reads this commit's index
VERSION_RE = re.compile(r"\d+(\.\d+){1,3}")


def check_requires(reg):
    """engines.json requires: known keys, each consumer's minimum a dotted version, and a why."""
    req, bad = reg.get("requires"), []
    if not isinstance(req, dict):
        bad.append("engines.json has no requires {boltbeam, why}")
    else:
        bad += [f"engines.json requires: unknown key {k}" for k in sorted(set(req) - REQUIRES_KEYS)]
        bad += [f"engines.json requires.{k} {req.get(k)!r} is not a dotted version (0.3.29)" for k in sorted(REQUIRES_KEYS - {"why"})
                if not isinstance(req.get(k), str) or not VERSION_RE.fullmatch(req[k])]
        bad += ["engines.json requires.why is empty"] * (not str(req.get("why") or "").strip())
    for b in bad:
        say("FAIL", "requires " + b)
    if not bad:
        say("ok", f"requires boltbeam >= {req['boltbeam']}")


def build_index(reg):
    """index.json: what a consumer reads instead of holding patches. Paths are relative to the repo root."""
    out = {"schema": "arkey_ie_mods.index.v1",
           "about": "Generated by tools/check.py --write from engines.json and every engines/<engine>/<version>/mods.json; the hook "
                    "fails when it is stale. A consumer reads it at one commit of this repo (a pin, or the newest commit whose "
                    "schema it reads and whose requires it meets) and records that commit; do not copy rows out of it. Gains, "
                    "limits, conditions and evidence live in each mods.json at the same commit. id = engine/version/name.",
           "requires": {k: v for k, v in (reg.get("requires") or {}).items() if k != "why"},
           "versions": [], "mods": [], "kernels": [], "archived": [], "tools": [], "retired": {"versions": [], "mods": []}}
    for f in folders_all():
        spec, _ = load_spec(os.path.join(f, "mods.json"))
        e, v = spec["engine"], spec["version"]
        row = reg["engines"].get(e, {})
        rel = os.path.relpath(f, ROOT)
        base = {k: spec["base"][k] for k in ("kind", "commit", "tag", "release", "package", "version", "record_sha256") if spec["base"].get(k)}
        rv = retired_version(row, v)
        # retired: named so a consumer can say why, never offered (not in versions, mods, kernels or tools)
        for state, rows in (("series", spec.get("mods", [])), ("parked", spec.get("parked", [])), ("retired", spec.get("retired", []))):
            for m in rows if rv or state == "retired" else []:
                r = rv or m.get("retired")
                if retired_bad(r):  # check_folder and check_layout fail it; never listed as current
                    continue
                out["retired"]["mods"].append({"id": f"{e}/{v}/{m['name']}", "engine": e, "version": v, "name": m["name"], "title": m.get("title"),
                                               "was": state, "by": "version" if rv else "mod", "date": r["date"], "why": r["why"],
                                               "patch": os.path.join(rel, m["patch"]), "sha256": sha256(os.path.join(f, m["patch"])), "lines": m["lines"]})
        if rv and not retired_bad(rv):
            out["retired"]["versions"].append({"engine": e, "version": v, "base": base, "date": rv["date"], "why": rv["why"]})
        if rv:
            for z in spec.get("archived", []):
                out["archived"].append({"id": f"{e}/{v}/{z['name']}", "engine": e, "version": v, "name": z["name"], "kind": z["kind"],
                                        "switch": z.get("switch"), "date": z["date"], "patch": os.path.join(rel, z["patch"]),
                                        "sha256": z["sha256"], "lines": z["lines"]})
            continue
        trees = spec.get("tested_trees") or []
        out["versions"].append({"engine": e, "version": v, "base": base, "applier": row.get("applier"),
                                "verified": {"state": "tree" if trees else "none",
                                             "why": "; ".join(f"{t['id']}: {t['contains']}" for t in trees) if trees else (spec.get("tested_trees_none") or {}).get("why"),
                                             "entries": spec.get("verified", [])}})
        for state, rows in (("series", spec.get("mods", [])), ("parked", spec.get("parked", []))):
            for i, m in enumerate(rows):
                out["mods"].append({
                    "id": f"{e}/{v}/{m['name']}", "engine": e, "version": v, "name": m["name"], "title": m.get("title"), "state": state,
                    "order": i + 1 if state == "series" else None, "applier": row.get("applier"), "base": base,
                    "switch": m.get("switch"), "switch_null_why": m.get("switch_null_why"),
                    "check": (m.get("check") or {}).get("name"), "check_needs": (m.get("check") or {}).get("needs", []), "marker": m.get("marker"),
                    "knobs": [x for x in [sw_name(m)] + m.get("reads", []) if x],
                    "output": m["output"]["label"], "patch": os.path.join(rel, m["patch"]), "sha256": sha256(os.path.join(f, m["patch"])),
                    "lines": m["lines"], "budget": m["budget"], "files": m["files"], "depends_on": m["depends_on"],
                    "rule_exceptions": [x["rule"] for x in m.get("rule_exceptions", [])], "module": m.get("module")})
        for z in spec.get("archived", []):
            out["archived"].append({"id": f"{e}/{v}/{z['name']}", "engine": e, "version": v, "name": z["name"], "kind": z["kind"],
                                    "switch": z.get("switch"), "date": z["date"], "patch": os.path.join(rel, z["patch"]),
                                    "sha256": z["sha256"], "lines": z["lines"]})
        for k in spec.get("kernels", []):
            out["kernels"].append({"id": f"{e}/{v}/kernels/{k['name']}", "engine": e, "version": v, "hook": f"{e}/{v}/{k['hook']}",
                                   "dir": os.path.join(rel, k["dir"]), "files": k["files"], "output": k["output"]["label"],
                                   "lines": k["lines"], "budget": k["budget"]})
        for t in spec.get("tools", []):
            out["tools"].append({"id": f"{e}/{v}/{os.path.basename(t['path'])}", "engine": e, "version": v, "path": os.path.join(rel, t["path"]),
                                 "sha256": t["sha256"], "needs": t.get("needs", []), "what": t["what"]})
    ids = [x["id"] for k in ("mods", "kernels", "archived", "tools") for x in out[k]] + [x["id"] for x in out["retired"]["mods"]]
    if len(set(ids)) != len(ids):
        say("FAIL", f"index   ids not unique: {sorted({i for i in ids if ids.count(i) > 1})}")
    for x in out["mods"]:
        if set(x) != INDEX_MOD_KEYS:
            say("FAIL", f"index   {x['id']}: keys {sorted(set(x) ^ INDEX_MOD_KEYS)} differ from the schema")
    return json.dumps(out, indent=1) + "\n"


def build_archive_readme(spec):
    lines = [f"# {spec['engine']} {spec['version']}: archive", "",
             "Generated by `tools/check.py --write` from mods.json `archived`. Do not edit by hand.",
             "Archived = decommissioned or rejected: kept so nothing that was patched is lost. Never in the series, parked or",
             "index.json's active list; a tested tree may apply one only to reach byte equality (mods.json tested_trees).",
             "Parked patches (may come back) live in parked/, not here.", ""]
    for z in spec["archived"]:
        sw = f"`{z['switch']}`" if z.get("switch") else "no switch"
        lines += [f"## {z['name']} ({z['kind']}, {z['date']})", "",
                  f"`{z['patch']}`, +{z['lines']['added']} -{z['lines']['removed']}, sha256 `{z['sha256'][:16]}`, {sw}.",
                  f"Written on: {z['written_on']}.", "", z["why"], "", f"Evidence: {z['evidence']}", ""]
    return "\n".join(lines)


def ledger_load():
    p = os.path.join(ROOT, "archive-ledger.json")
    return json.load(open(p)) if os.path.exists(p) else {"schema": "arkey_ie_mods.archive_ledger.v1", "entries": []}


def check_ledger(ledger, write):
    """Append-only: every archived row is in it; nothing in it is ever removed (against HEAD); every entry keeps its row."""
    print("\narchive ledger")
    rows = {}
    for f in folders_all():
        spec = json.load(open(os.path.join(f, "mods.json")))
        for z in spec.get("archived", []):
            rows[f"{spec['engine']}/{spec['version']}/{z['name']}"] = z
    have = {e["id"]: e for e in ledger["entries"]}  # the latest entry per id

    def scrubbed(z):
        x = z.get("scrubbed_from") or []
        return {e["sha256"] for e in (x if isinstance(x, list) else [x])}
    # a superseding entry is allowed only when the row records the ledger's current sha256 in scrubbed_from
    stale = [i for i in rows if i in have and rows[i]["sha256"] != have[i]["sha256"] and have[i]["sha256"] in scrubbed(rows[i])]
    missing = [i for i in rows if i not in have]
    if write and (missing or stale):
        for i in missing:
            ledger["entries"].append({"id": i, "sha256": rows[i]["sha256"], "switch": rows[i].get("switch"), "date": rows[i]["date"]})
        for i in stale:
            ledger["entries"].append({"id": i, "sha256": rows[i]["sha256"], "switch": rows[i].get("switch"), "date": rows[i]["date"],
                                      "supersedes": have[i]["sha256"], "why": "text-only rewrite recorded in the row's scrubbed_from"})
        open(os.path.join(ROOT, "archive-ledger.json"), "w").write(json.dumps(ledger, indent=1) + "\n")
        say("ok", f"wrote   archive-ledger.json (+{len(missing)} new, +{len(stale)} superseding)")
        have = {e["id"]: e for e in ledger["entries"]}
        missing = []
    for i in missing:
        say("FAIL", f"ledger  {i} is archived but not in archive-ledger.json: run python3 tools/check.py --write")
    for i, e in have.items():
        z = rows.get(i)
        if not z:
            say("FAIL", f"ledger  {i}: archived once, its archived row is gone (archived patches never come back)")
        elif z["sha256"] != e["sha256"]:
            say("FAIL", f"ledger  {i}: sha256 changed since it was archived" + (": run python3 tools/check.py --write to append a superseding entry" if e["sha256"] in scrubbed(z) else " (a rewrite must record the archived sha256 in scrubbed_from)"))
    if not missing and all(rows.get(i) and rows[i]["sha256"] == e["sha256"] for i, e in have.items()):
        say("ok", f"ledger  {len(have)} archived patches recorded; each keeps its row and sha256")
    old = head_json("archive-ledger.json")
    if old is None and GIT is None:
        say("SKIP", "ledger  no git history here: append-only not checked")
    elif old:
        cur = {(e["id"], e["sha256"]) for e in ledger["entries"]}
        gone = [e["id"] for e in old["entries"] if (e["id"], e["sha256"]) not in cur]
        say("FAIL" if gone else "ok", f"ledger  append-only against HEAD" + (f": removed or changed {gone}" if gone else f" ({len(ledger['entries'])} entries)"))


def check_exports(reg, write):
    print("\nexports")
    want = {os.path.join(ROOT, "index.json"): build_index(reg)}
    for f in folders_all():
        spec = json.load(open(os.path.join(f, "mods.json")))
        if spec.get("archived"):
            want[os.path.join(f, "archive", "README.md")] = build_archive_readme(spec)
    for p, text in want.items():
        rel = os.path.relpath(p, ROOT)
        if write:
            open(p, "w").write(text)
            say("ok", f"wrote   {rel}")
        elif not os.path.exists(p) or open(p).read() != text:
            say("FAIL", f"stale   {rel}: run python3 tools/check.py --write")
        else:
            say("ok", f"fresh   {rel}")


def main():
    global LOCAL, GIT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", nargs="*", help="engines/<engine>/<version> (default: all)")
    ap.add_argument("--write", action="store_true", help="regenerate index.json and archive/README.md; append new archived rows to the ledger")
    ap.add_argument("--strict", action="store_true", help="any SKIP fails")
    ap.add_argument("--local", help="local config (default: $ARKEY_IE_MODS_LOCAL, else local.json at the repo root)")
    ap.add_argument("--git", help="the git repo whose HEAD the history checks compare against (default: this repo when it has .git)")
    a = ap.parse_args()
    lp = a.local or os.environ.get("ARKEY_IE_MODS_LOCAL") or os.path.join(ROOT, "local.json")
    LOCAL = json.load(open(os.path.expanduser(lp))) if os.path.exists(os.path.expanduser(lp)) else {}
    GIT = a.git or (ROOT if os.path.exists(os.path.join(ROOT, ".git")) else None)
    reg = json.load(open(os.path.join(ROOT, "engines.json")))
    if not LOCAL:
        say("SKIP", f"local   no local config at {lp} (copy local.example.json to local.json)")
    folders = [os.path.abspath(f) for f in a.folder] or folders_all()
    for name, row in reg["engines"].items():
        n = sum(1 for f in folders if os.path.basename(os.path.dirname(f)) == name)
        print(f"engine {name}: {row['status']}, {n} version folder(s)")
    lint = Lint(reg)
    check_layout(reg)
    check_requires(reg)
    check_docs(reg, lint)
    ledger = ledger_load()
    for f in folders:
        engine = os.path.basename(os.path.dirname(f))
        if engine not in reg["engines"]:
            print(f"\n{os.path.relpath(f, ROOT)}")
            say("FAIL", f"engine '{engine}' has no row in engines.json")
            continue
        check_folder(f, reg["engines"][engine], reg, lint, ledger["entries"], a.strict)
    check_ledger(ledger, a.write)
    check_exports(reg, a.write)
    strict_fail = a.strict and skips
    print(f"\n{'FAILED' if fails or strict_fail else 'OK'}: {len(fails)} failure(s), {len(skips)} skip(s)" + (" (--strict: skips fail)" if strict_fail else ""))
    return 1 if fails or strict_fail else 0


if __name__ == "__main__":
    sys.exit(main())
