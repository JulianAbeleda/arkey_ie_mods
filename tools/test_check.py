#!/usr/bin/env python3
"""Tests for tools/check.py check 10 (stack) and the per-mod budget. Standard library only:
python3 -m unittest discover -s tools -p 'test_*.py'

The fixture is a three-mod series on one base file, each mod stacked on the one before. Its caps are set from the
first mod alone, so the full series crosses every stack cap; the first mod alone crosses none."""
import contextlib, io, json, os, shutil, sys, tempfile, unittest, warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check as c  # noqa: E402

BASE = "".join(f"line{i}\n" for i in range(1, 11))
# 01 adds A after line2; 02 adds B after A (A is its context: stacked, depends on 01); 03 adds C after B (depends on 02)
PATCHES = {
    "01-a.patch": "--- a/src/f.c\n+++ b/src/f.c\n@@ -2,2 +2,3 @@ int run(void)\n line2\n+A\n line3\n",
    "02-b.patch": "--- a/src/f.c\n+++ b/src/f.c\n@@ -3,2 +3,3 @@ int run(void)\n A\n+B\n line3\n",
    "03-c.patch": "--- a/src/f.c\n+++ b/src/f.c\n@@ -4,2 +4,3 @@ int run(void)\n B\n+C\n line3\n",
}
DEPS = {"a": [], "b": ["a"], "c": ["b"]}


def row(name, patch, added=1, budget=1):
    return {"name": name, "title": f"mod {name}", "patch": f"patches/{patch}", "engine": "eng", "base": "1.0",
            "switch": None, "switch_null_why": "fixture", "check": None, "marker": name.upper(), "files": ["src/f.c"],
            "lines": {"added": added, "removed": 0}, "budget": {"added": budget, "removed": 0},
            "depends_on": {"apply": DEPS[name], "run": []}, "static": {"hunks_removing_stock_lines": 0},
            "output": {"label": "untested", "evidence": {"numbers": "fixture"}},
            "rule_exceptions": [{"rule": "check", "why": "fixture"}, {"rule": "switch", "why": "fixture"}]}


def budget_for(spec_mods, folder, date="2026-11-18"):
    vals, _ = c.stack_values({"mods": spec_mods}, folder)
    n = len(spec_mods)
    return {"schema": "arkey_ie_mods.stack_budget.v1", "about": "fixture", "margin_rule": "one mod",
            "caps": {k: {"cap": c.stack_margin(v, n), "set": {"measured": v, "mods": n, "date": date}, "raised": []}
                     for k, v in vals.items()}}


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


class Fixture(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)  # check.py reads files with bare open(), as it always has
        self.tmp = tempfile.mkdtemp(prefix="stack_test_")
        self.base = os.path.join(self.tmp, "base")
        self.folder = os.path.join(self.tmp, "engines", "eng", "1.0")
        os.makedirs(os.path.join(self.base, "src"))
        os.makedirs(os.path.join(self.folder, "patches"))
        write(os.path.join(self.base, "src", "f.c"), BASE)
        for n, t in PATCHES.items():
            write(os.path.join(self.folder, "patches", n), t)
        self.mods = [row(n, p) for n, p in zip("abc", PATCHES)]
        c.fails.clear()
        c.skips.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp)
        c.fails.clear()
        c.skips.clear()

    def write_budget(self, sb):
        write(os.path.join(self.folder, "stack_budget.json"), json.dumps(sb))
    def stack(self, mods, head="committed", base=None):
        head = budget_for(self.mods[:1], self.folder) if head == "committed" else head
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            c.check_stack(self.folder, {"mods": mods}, base, head)
        return out.getvalue(), list(c.fails)


class Measure(Fixture):
    def test_values(self):
        for base in (None, self.base):
            vals, why = c.stack_values({"mods": self.mods}, self.folder, base)
            self.assertEqual(vals, {"total_lines": 3, "mods_per_file": 3, "overlap_hunks_per_file": 2, "depth": 3, "stacked_hunks": 2})
            self.assertEqual(why["chain"], ["a", "b", "c"])
            self.assertEqual(why["problems"], [])

    def test_offset_placement(self):
        # 02 says line 1, but its context (A, line3) is at line 3 once 01 applied: placed by text, like git apply
        write(os.path.join(self.folder, "patches", "02-b.patch"), 
            "--- a/src/f.c\n+++ b/src/f.c\n@@ -1,2 +1,3 @@ int run(void)\n A\n+B\n line3\n")
        vals, why = c.stack_values({"mods": self.mods}, self.folder, self.base)
        self.assertEqual(vals["stacked_hunks"], 2)
        self.assertEqual(why["problems"], [])

    def test_unplaceable_hunk_is_a_problem(self):
        write(os.path.join(self.folder, "patches", "03-c.patch"), 
            "--- a/src/f.c\n+++ b/src/f.c\n@@ -4,2 +4,3 @@\n nosuchline\n+C\n line3\n")
        _, why = c.stack_values({"mods": self.mods}, self.folder, self.base)
        self.assertTrue(why["problems"])

    def test_removed_line_starting_with_dashes_is_body(self):
        p = os.path.join(self.tmp, "x.patch")
        write(p, "--- a/g.sql\n+++ b/g.sql\n@@ -1,2 +1,1 @@\n-- old\n keep\n")
        (f, hs), = c.hunks(p)
        self.assertEqual(f, "g.sql")
        self.assertEqual(hs[0][5], [("-", "- old"), (" ", "keep")])

    def test_no_newline_marker_and_stripped_blank_context(self):
        p = os.path.join(self.tmp, "x.patch")
        write(p, "--- a/g.c\n+++ b/g.c\n@@ -1,3 +1,3 @@\n keep\n\n-old\n\\ No newline at end of file\n+new\n")
        (f, hs), = c.hunks(p)
        self.assertEqual(hs[0][5], [(" ", "keep"), (" ", ""), ("-", "old"), ("+", "new")])

    def test_deleted_file_and_several_files(self):
        p = os.path.join(self.tmp, "x.patch")
        write(p, "--- a/old.c\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-x\n-y\n"
                 "--- /dev/null\n+++ b/new.c\n@@ -0,0 +1,2 @@\n+x\n+y\n")
        self.assertEqual([f for f, _ in c.hunks(p)], ["old.c", "new.c"])

    def test_margin_formula(self):
        self.assertEqual(c.stack_margin(3513, 19), 3513 + 185)
        self.assertEqual(c.stack_margin(2, 19), 3)
        self.assertEqual(c.stack_margin(0, 6), 0)


class Caps(Fixture):
    def test_within_caps_passes_and_prints_summary(self):
        self.write_budget(budget_for(self.mods[:1], self.folder))
        out, fails = self.stack(self.mods[:1], head=None)
        self.assertEqual(fails, [])
        self.assertIn("stack health", out)
        self.assertIn("hot spot  src/f.c", out)
        self.assertIn("headroom", out)

    def test_fixture_crosses_every_cap(self):
        self.write_budget(budget_for(self.mods[:1], self.folder))
        _, fails = self.stack(self.mods)
        for name in c.STACK_METRICS:
            hit = [f for f in fails if f.startswith(f"stack   {name} ")]
            self.assertEqual(len(hit), 1, name)
            self.assertIn("> cap", hit[0])
            self.assertIn("Suggested refactor", hit[0])
            self.assertIn("Owner-approved YYYY-MM-DD", hit[0])

    def test_total_lines(self):
        self.write_budget(budget_for(self.mods[:1], self.folder))
        _, fails = self.stack(self.mods)
        self.assertTrue(any(f.startswith("stack   total_lines 3 > cap 2") for f in fails))

    def test_hot_spot_names_file_function_and_mods(self):
        self.write_budget(budget_for(self.mods[:1], self.folder))
        _, fails = self.stack(self.mods)
        f = next(f for f in fails if f.startswith("stack   mods_per_file"))
        self.assertIn("refactor: 3 mods patch src/f.c (a, b, c)", f)
        self.assertIn("3 mods patch src/f.c/int run(void)", f)
        self.assertIn("introduce an extension point", f)
        f = next(f for f in fails if f.startswith("stack   overlap_hunks_per_file"))
        self.assertIn("2 overlapping hunks in src/f.c", f)

    def test_depth(self):
        self.write_budget(budget_for(self.mods[:1], self.folder))
        _, fails = self.stack(self.mods)
        f = next(f for f in fails if f.startswith("stack   depth"))
        self.assertIn("depth 3 > cap 2", f)
        self.assertIn("c -> b -> a", f)

    def test_rebase_fragility(self):
        self.write_budget(budget_for(self.mods[:1], self.folder))
        _, fails = self.stack(self.mods)
        f = next(f for f in fails if f.startswith("stack   stacked_hunks"))
        self.assertIn("b on a at src/f.c/int run(void)", f)
        self.assertIn("c on b", f)

    def test_missing_budget_file_fails_with_derived_caps(self):
        _, fails = self.stack(self.mods, head=None)
        self.assertTrue(any("stack_budget.json missing" in f and "total_lines 3 -> 4" in f for f in fails))

    def test_hand_picked_cap_fails(self):
        sb = budget_for(self.mods, self.folder)
        sb["caps"]["total_lines"]["cap"] = 100
        self.write_budget(sb)
        _, fails = self.stack(self.mods, head=None)
        self.assertTrue(any("total_lines: cap 100 is not the derived or last raised value 4" in f for f in fails))


class Raises(Fixture):
    def raise_to(self, **kw):
        sb = budget_for(self.mods[:1], self.folder)
        r = {"from": 2, "to": 3, "measured": 3, "why": "02 and 03 reviewed, cut to one line each", "date": "2026-11-18",
             "approved": "Owner-approved 2026-11-18"}
        r.update(kw)
        sb["caps"]["total_lines"]["raised"].append({k: v for k, v in r.items() if v is not None})
        sb["caps"]["total_lines"]["cap"] = r["to"]
        self.write_budget(sb)
        return sb

    def test_owner_approved_raise_to_measured_clears_the_cap(self):
        self.raise_to()
        _, fails = self.stack(self.mods)
        self.assertFalse(any(f.startswith("stack   total_lines") or "total_lines:" in f for f in fails))

    def test_raise_without_owner_approval_fails(self):
        self.raise_to(approved="approved by me")
        _, fails = self.stack(self.mods)
        self.assertTrue(any("is not 'Owner-approved YYYY-MM-DD'" in f for f in fails))

    def test_raise_without_why_fails(self):
        self.raise_to(why=None)
        _, fails = self.stack(self.mods)
        self.assertTrue(any("missing ['why']" in f for f in fails))

    def test_raise_must_be_todays_value(self):
        self.raise_to(to=999, measured=999)
        _, fails = self.stack(self.mods)
        self.assertTrue(any("new raise measured 999 is not today's value 3" in f for f in fails))

    def test_made_up_set_fails(self):
        sb = budget_for(self.mods, self.folder)
        sb["caps"]["depth"].update({"cap": 100, "set": {"measured": 50, "mods": 1, "date": "2026-11-18"}})
        self.write_budget(sb)
        _, fails = self.stack(self.mods, head=None)
        self.assertTrue(any("depth: set {measured 50, mods 1} is new against HEAD and is not today's measurement" in f for f in fails))

    def test_raise_must_be_the_measured_value(self):
        self.raise_to(to=10, measured=3)
        _, fails = self.stack(self.mods)
        self.assertTrue(any("raise to 10 is not its measured value 3" in f for f in fails))

    def test_raise_from_must_be_the_previous_cap(self):
        self.raise_to(**{"from": 1})
        _, fails = self.stack(self.mods)
        self.assertTrue(any("raise from 1, the cap before it is 2" in f for f in fails))

    def test_cap_up_against_head_needs_a_new_raise(self):
        self.write_budget(budget_for(self.mods, self.folder))  # re-derived from today's series: a silent raise
        _, fails = self.stack(self.mods)
        self.assertTrue(any("against HEAD without a new Owner-approved raise" in f for f in fails))

    def test_bad_json_is_a_failure(self):
        write(os.path.join(self.folder, "stack_budget.json"), "{")
        _, fails = self.stack(self.mods)
        self.assertTrue(any("is not JSON" in f for f in fails))

    def test_raises_are_append_only_against_head(self):
        head = self.raise_to()
        sb = budget_for(self.mods[:1], self.folder)
        sb["caps"]["total_lines"]["cap"] = 2
        self.write_budget(sb)
        _, fails = self.stack(self.mods[:1], head=head)
        self.assertTrue(any("append-only against HEAD" in f for f in fails))


class PerMod(Fixture):
    """rule 5: the per-mod budget (check 4) still runs beside the stack caps, through check_folder."""

    def run_folder(self, mods):
        spec = {"schema": "x", "engine": "eng", "version": "1.0", "base": {"kind": "git_commit", "commit": "0" * 40},
                "about": "fixture", "budget_rule": "measured", "mods": mods, "parked": [], "archived": [],
                "tested_trees_none": {"why": "fixture"}}
        write(os.path.join(self.folder, "mods.json"), json.dumps(spec))
        reg = {"engines": {"eng": {}}, "appliers": {"git_apply": {"implemented": True}},
               "lint": {"model_name_regex": "zzzmodel", "private_regex": "zzzprivate"}}
        erow = {"applier": "git_apply", "base_identity": {"kind": "git_commit"}, "switch_kinds": {},
                "switch_literal_regex": "\"(ZZZ_[A-Z]+)\"", "env_read_regex": "getenv\\(\"([^\"]+)\"\\)", "switch_presence": {}}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            c.check_folder(self.folder, erow, reg, c.Lint(reg), [], False)
        return out.getvalue(), list(c.fails)

    def test_per_mod_budget_within(self):
        self.write_budget(budget_for(self.mods, self.folder))
        out, fails = self.run_folder(self.mods)
        self.assertEqual(fails, [], out)
        self.assertIn("stack health", out)

    def test_per_mod_over_budget_fails(self):
        self.write_budget(budget_for(self.mods, self.folder))
        self.mods[1]["lines"]["added"] = 1
        self.mods[1]["budget"]["added"] = 0
        self.mods[1]["budget_raised"] = [{"from": {"added": 1, "removed": 0}, "to": {"added": 0, "removed": 0}, "why": "x", "date": "2026-11-18"}]
        _, fails = self.run_folder(self.mods)
        self.assertTrue(any("OVER BUDGET" in f and " b " in f for f in fails))


if __name__ == "__main__":
    unittest.main()
