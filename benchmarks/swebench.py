"""
benchmarks/swebench.py

Adapter: turn SWE-bench (Lite) instances into eval.harness.TaskSpec objects so
the existing agent eval harness can drive and grade them.

SWE-bench differs fundamentally from HumanEval:
  - Each instance is a real GitHub issue on a real repo at a specific base_commit.
  - The agent is given ONLY the problem_statement (the issue text) and the repo
    checked out at base_commit. It must locate and fix the bug by editing source.
  - Grading ("resolved") is objective and mirrors the official SWE-bench
    criterion: after the agent finishes we apply the instance's *test_patch*
    (which the agent never sees) and require every FAIL_TO_PASS test to pass AND
    every PASS_TO_PASS test to keep passing.

No Docker here (by design choice for this host): instead of the official
per-instance Docker image, we build a best-effort per-(repo, version) virtualenv
on the host and run the designated pytest node ids in it. This runs real repos
and grades real tests, but cannot reproduce the exact official environment —
heavy repos (C-extension builds like numpy/matplotlib, or django's custom test
runner) may fail environment setup, which we surface as a distinct ENV_ERROR
rather than a silent FAIL. Use the runner's --mock flag to validate the whole
clone -> edit -> test_patch -> grade pipeline on real data without an LLM.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from eval.harness import TaskSpec
from eval.verifiers import Verifier

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
# Heavy, regenerable caches (clones + per-repo venvs) live under the gitignored
# eval_runs/ tree so they never get committed.
_CACHE_ROOT = Path(os.environ.get("SWEBENCH_CACHE", "eval_runs/swebench_cache")).resolve()
_REPOS = _CACHE_ROOT / "repos"
_VENVS = _CACHE_ROOT / "venvs"

# Node-id batch size when invoking pytest (keeps the argv well under ARG_MAX for
# instances whose PASS_TO_PASS set has hundreds of tests).
_PYTEST_BATCH = 80

# `uv` (if present) is used to build per-instance venvs with the exact old Python
# interpreter each SWE-bench instance targets. Falls back to the host interpreter.
_UV = shutil.which("uv")

# Per-(repo, version) install specs extracted from the official `swebench`
# package (benchmarks/data/swebench_specs.json). These provide the *pinned*
# third-party dependency versions that make an old checkout actually import and
# run — the single most important thing Docker normally handles. Without them,
# `pip install -e .` resolves modern deps that break old code (e.g. Werkzeug 3.x
# removing `url_quote` for Flask 2.0).
_SPECS_PATH = _DATA_DIR / "swebench_specs.json"
_SPECS_CACHE: dict | None = None


def _specs() -> dict:
    global _SPECS_CACHE
    if _SPECS_CACHE is None:
        try:
            _SPECS_CACHE = json.loads(_SPECS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _SPECS_CACHE = {}
    return _SPECS_CACHE


def _spec_for(repo: str, version: str) -> dict | None:
    return _specs().get(repo, {}).get(str(version))


def _runner_kind(spec: dict | None) -> str | None:
    """
    Which test runner grades this instance: "pytest", "django", or None (unsupported).

    Most Lite repos report a pytest `test_cmd`. Three do not, and each reduces to
    something we can drive locally:
      - sphinx  `tox --current-env -epy39 --`  → tox merely runs pytest in the
        current environment, and its test ids are already pytest node ids.
      - sympy   `bin/test`                     → sympy's test files are ordinary
        pytest-collectable `test_*.py` modules; only its bare test *names* need
        resolving to `file::name` node ids (see _resolve_nodeids).
      - django  `./tests/runtests.py`          → a genuinely different runner,
        driven by _run_django_tests.
    """
    tc = (spec or {}).get("test_cmd", "")
    core = re.sub(r"^([A-Za-z_][A-Za-z0-9_]*=\S+\s+)+", "", tc).strip()  # drop VAR=val prefixes
    if core.startswith("pytest"):
        return "pytest"
    if core.startswith("tox") and "--current-env" in core:
        return "pytest"          # sphinx
    if "runtests.py" in core:
        return "django"
    if core.startswith("bin/test"):
        return "pytest"          # sympy, via resolved node ids
    return None


def _test_patch_files(test_patch: str) -> list[str]:
    """Test files touched by the instance's hidden test patch."""
    return re.findall(r"^\+\+\+ b/(\S+)", test_patch or "", re.M)


def _strip_truncated_params(nodeid: str) -> str:
    """Repair a node id whose parametrisation was cut off by the dataset.

    SWE-bench built its FAIL_TO_PASS / PASS_TO_PASS lists by splitting a test log
    on whitespace, so a parametrised id containing a space is stored truncated —
    `test_escape_args_and_kwargs[x, y]` becomes the fragment
    `test_escape_args_and_kwargs[x,`. Such a fragment matches nothing and would
    fail collection, sinking the whole batch. Detect the unbalanced `[` and drop
    the parametrisation, which runs EVERY case of that test — stricter than the
    original id, so this can never turn a real failure into a pass.
    """
    if nodeid.count("[") > nodeid.count("]"):
        return nodeid[:nodeid.index("[")]
    return nodeid


def _resolve_nodeids(names: list[str], test_patch: str, repo_path: Path) -> list[str]:
    """
    Turn a repo's test identifiers into pytest node ids.

    Already-qualified ids (`tests/test_x.py::test_y`) pass through untouched.
    sympy reports *bare* function names (`test_ccode_Relational`) with no file, so
    each is resolved against the files the test patch touches, falling back to a
    repo-wide search for `def <name>(`. Unresolvable names are passed through and
    will simply fail collection, which grades as a failure rather than silently
    disappearing from the denominator.
    """
    patch_files = _test_patch_files(test_patch)
    resolved: list[str] = []
    for name in names:
        if "::" in name or name.endswith(".py"):
            resolved.append(_strip_truncated_params(name))
            continue
        target = None
        for f in patch_files:                       # cheap path first
            p = repo_path / f
            if p.exists() and re.search(rf"^\s*def {re.escape(name)}\s*\(",
                                        p.read_text(encoding="utf-8", errors="replace"), re.M):
                target = f
                break
        if target is None:                          # fall back to searching the repo
            rc, out = _run(["grep", "-rl", "--include=test_*.py",
                            f"def {name}(", str(repo_path)], timeout=120)
            if rc == 0 and out.strip():
                first = out.strip().splitlines()[0]
                try:
                    target = str(Path(first).relative_to(repo_path))
                except ValueError:
                    target = first
        resolved.append(f"{target}::{name}" if target else name)
    # Truncated parametrised ids collapse onto the same test; keep first occurrence.
    return list(dict.fromkeys(resolved))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _default_data_path(split: str, dataset: str = "SWE-bench_Lite") -> Path:
    return _DATA_DIR / f"{dataset}_{split}.jsonl.gz"


def load_records(
    split: str = "test",
    dataset: str = "SWE-bench_Lite",
    path: str | Path | None = None,
) -> list[dict]:
    """Load raw SWE-bench records; FAIL_TO_PASS / PASS_TO_PASS are decoded to lists."""
    p = Path(path) if path else _default_data_path(split, dataset)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download it first:\n"
            f"  python -m benchmarks.download_swebench --split {split}"
        )
    opener = gzip.open if p.suffix == ".gz" else open
    recs = []
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                val = rec.get(key)
                if isinstance(val, str):
                    rec[key] = json.loads(val)
                elif val is None:
                    rec[key] = []
            recs.append(rec)
    return recs


def gold_patch(rec: dict) -> str:
    """The instance's gold solution patch (used only by --mock to validate the pipeline)."""
    return rec["patch"]


def problem_prompt(rec: dict) -> str:
    """The task description shown to the agent: the issue text, nothing test-revealing."""
    return (
        f"You are working inside the `{rec['repo']}` repository, already checked out "
        f"at the exact commit for this issue. Resolve the following GitHub issue by "
        f"editing the repository's SOURCE code.\n\n"
        f"Rules:\n"
        f"- Do NOT modify or add tests — a hidden test suite will grade your fix.\n"
        f"- Make the smallest change that correctly resolves the issue.\n"
        f"- Use the file/search/git tools to locate the relevant code first.\n\n"
        f"--- ISSUE ---\n{rec['problem_statement']}\n--- END ISSUE ---\n\n"
        f"When you are confident the issue is fixed, finish. Your working-tree "
        f"changes will be verified by an independent test run."
    )


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 300,
         env: dict | None = None) -> tuple[int, str]:
    """Run a command; return (returncode, combined stdout+stderr). Never raises."""
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {' '.join(cmd[:6])}"
    except Exception as exc:  # noqa: BLE001
        return 1, f"failed to run {' '.join(cmd[:4])}: {exc}"


def _flat(repo: str) -> str:
    return repo.replace("/", "__")


# ---------------------------------------------------------------------------
# Repo checkout (setup_hook)
# ---------------------------------------------------------------------------

def _ensure_cache_clone(repo: str) -> Path:
    """Full clone of `repo` under the cache, created once and reused."""
    dest = _REPOS / _flat(repo)
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("cloning %s (first use, may take a while)...", repo)
    rc, out = _run(
        ["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(dest)],
        timeout=1800,
    )
    if rc != 0:
        raise RuntimeError(f"git clone {repo} failed: {out[-500:]}")
    return dest


def _has_commit(repo_git: Path, commit: str) -> bool:
    rc, _ = _run(["git", "-C", str(repo_git), "cat-file", "-e", f"{commit}^{{commit}}"],
                 timeout=30)
    return rc == 0


def make_setup_hook(repo: str, base_commit: str, version: str | None = None,
                    prepare_env: bool = True):
    """Return a setup_hook(repo_path) that clones `repo` and checks out base_commit.

    When `prepare_env` is set (and a version is known) the instance's virtualenv is
    also built here, i.e. BEFORE the agent starts, so the agent's own test/shell
    tools can run the project's suite via a VenvRuntime. Otherwise the env would
    only be created later by the verifier and the agent would be editing blind.
    The verifier reuses the same cached venv, so this moves the cost rather than
    adding it.
    """
    def hook(repo_path: Path) -> None:
        cache = _ensure_cache_clone(repo)
        if not _has_commit(cache, base_commit):
            # base_commit may live on a branch not represented by a local ref; the
            # object is usually already present from the full clone, but fetch to be safe.
            _run(["git", "-C", str(cache), "fetch", "--quiet", "origin", base_commit],
                 timeout=600)
        # repo_path was created empty by the harness; a local clone hardlinks the
        # object store, so every commit (incl. base_commit) is available to check out.
        # Generous timeouts: big repos (django is ~7k files) checked out onto an
        # NFS-mounted work dir can take minutes. Point --results-dir at local disk
        # to make this dramatically faster.
        rc, out = _run(["git", "clone", "--quiet", "--no-checkout", str(cache), str(repo_path)],
                       timeout=1800)
        if rc != 0:
            raise RuntimeError(f"local clone failed for {repo}: {out[-300:]}")
        rc, out = _run(["git", "-C", str(repo_path), "checkout", "--quiet", base_commit],
                       timeout=1800)
        if rc != 0:
            raise RuntimeError(f"checkout {base_commit[:8]} failed for {repo}: {out[-300:]}")

        if prepare_env and version is not None:
            # Best-effort: a failure here is reported properly by the verifier
            # later (as ENV_ERROR); it must not abort the run.
            ok, detail = _ensure_env(repo, str(version), repo_path)
            if not ok:
                logger.warning("env prebuild failed for %s: %s", repo, detail)
    return hook


# ---------------------------------------------------------------------------
# Per-repo virtualenv (best-effort environment for grading)
# ---------------------------------------------------------------------------

def _venv_dir(repo: str, version: str) -> Path:
    return _VENVS / f"{_flat(repo)}__{version}"


def _venv_python(repo: str, version: str) -> Path:
    return _venv_dir(repo, version) / "bin" / "python"


def _pip(py: Path, *args: str, cwd: Path | None = None, timeout: int = 1800) -> tuple[int, str]:
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1"}
    # Same reason as _venv_shell: constrain the setuptools pip drops into its
    # isolated build envs, not just the one installed in the venv.
    constraints = _constraints_path(py.parent.parent)
    if constraints.exists():
        env["PIP_CONSTRAINT"] = str(constraints)
    return _run([str(py), "-m", "pip", *args], cwd=cwd, timeout=timeout, env=env)


def _create_venv(pyver: str, venv_dir: Path) -> tuple[int, str]:
    """
    Create the per-instance venv with the interpreter version the instance was
    built for. SWE-bench instances target old Pythons (mostly 3.9; also 3.6-3.8)
    — running them under the host's 3.11 breaks on removed stdlib APIs
    (inspect.formatargspec) and AST changes. `uv` provisions/downloads the exact
    interpreter; if uv is unavailable we fall back to the host interpreter.
    """
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    if _UV:
        env = {**os.environ, "UV_LINK_MODE": "copy"}
        rc, out = _run([_UV, "venv", "--seed", "--python", pyver, str(venv_dir)],
                       timeout=900, env=env)
        if rc == 0:
            return rc, out
        logger.warning("uv venv --python %s failed, falling back to host python: %s",
                       pyver, out[-200:])
    rc, out = _run([sys.executable, "-m", "venv", str(venv_dir)], timeout=300)
    if rc == 0:
        _pip(_venv_python_at(venv_dir), "install", "-q", "-U", "pip", timeout=600)
    return rc, out


def _venv_python_at(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _constraints_path(venv_dir: Path) -> Path:
    """pip constraints applied to this venv's installs *and* its isolated builds."""
    return venv_dir / ".constraints.txt"


def _venv_shell(cmd: str, venv_dir: Path, cwd: Path, timeout: int = 1800) -> tuple[int, str]:
    """Run a spec-provided install command (e.g. `python -m pip install -e .`) with the
    per-repo venv on PATH, so its `python`/`pip` resolve to the venv."""
    env = {
        **os.environ,
        "VIRTUAL_ENV": str(venv_dir),
        "PATH": f"{venv_dir / 'bin'}:{os.environ.get('PATH', '')}",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
    }
    # pip builds in an *isolated* env that it seeds with the newest setuptools,
    # ignoring whatever the venv has pinned — which breaks repos whose setup.py
    # uses APIs newer setuptools dropped (astropy 5.x imports the deleted
    # setuptools.dep_util). PIP_CONSTRAINT is the one knob that reaches inside
    # build isolation.
    constraints = _constraints_path(venv_dir)
    if constraints.exists():
        env["PIP_CONSTRAINT"] = str(constraints)
    try:
        proc = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True,
                              text=True, timeout=timeout, env=env)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {cmd[:80]}"
    except Exception as exc:  # noqa: BLE001
        return 1, f"failed to run {cmd[:60]}: {exc}"


def _ensure_env(repo: str, version: str, repo_path: Path) -> tuple[bool, str]:
    """
    Ensure a per-(repo, version) venv exists with the instance's *pinned*
    dependencies, then (re)install the repo so THIS task's checkout is what gets
    imported and tested. Returns (ok, detail).

    Dependency pins come from the vendored SWE-bench install specs; installing
    them BEFORE the repo build stops `pip install -e .` from dragging in modern,
    incompatible versions. Heavy deps are gated behind a per-venv marker so only
    the first instance of a (repo, version) pays the install cost; every grading
    still re-runs the repo build to bind the current source.
    """
    spec = _spec_for(repo, version) or {}
    venv_dir = _venv_dir(repo, version)
    py = _venv_python(repo, version)
    marker = venv_dir / ".installed"

    if not py.exists():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        rc, out = _create_venv(spec.get("python") or "3.11", venv_dir)
        if rc != 0 or not py.exists():
            return False, f"ENV_ERROR: venv create failed (py={spec.get('python')}): {out[-200:]}"

    # Pins this old (astropy 0.1 wants MarkupSafe==1.0) have setup.py files that
    # import APIs modern setuptools deleted — `setuptools.Feature` went away in
    # 46, `setuptools.dep_util` in 70. The original images for these Pythons
    # shipped setuptools ~39, so match that era rather than whatever the venv
    # seed installed. Written every run: the repo build below needs it too.
    pyver = str(spec.get("python") or "")
    setuptools_pin = "setuptools<45" if pyver.startswith(("3.6", "3.7")) else "setuptools<70"
    _constraints_path(venv_dir).write_text(setuptools_pin + "\n")

    if not marker.exists():
        # Pinned third-party deps first (the crucial step).
        pins = spec.get("pip_packages") or []
        if pins:
            _pip(py, "install", "-q", setuptools_pin, timeout=600)
            rc, out = _pip(py, "install", "-q", *pins, timeout=1800)
            if rc != 0:
                # A single unbuildable pin shouldn't sink the whole environment:
                # retry one at a time and keep the ones that work. The casualties
                # are usually optional docs/lint deps the tests never import, but
                # log them so a later mystery failure is traceable.
                skipped = [pkg for pkg in pins
                           if _pip(py, "install", "-q", pkg, timeout=900)[0] != 0]
                if len(skipped) == len(pins):
                    return False, f"ENV_ERROR: pinned deps failed: {out[-300:]}"
                if skipped:
                    logger.warning("%s %s: skipped %d unbuildable pin(s): %s",
                                   repo, version, len(skipped), ", ".join(skipped))
        # A `packages` value may be an extra list of conda-style names ("numpy
        # scipy pytest"); "requirements.txt"/"pytest" are handled by the build.
        packages = spec.get("packages")
        if packages and packages not in ("requirements.txt",) and not packages.endswith(".txt"):
            _pip(py, "install", "-q", *packages.split(), timeout=1800)
        if "setuptools" not in " ".join(pins):
            _pip(py, "install", "-q", setuptools_pin, "wheel", timeout=600)
        marker.write_text("ok\n")

    # (Re)build the repo from the current checkout so the agent's edits are what
    # gets tested. Use the spec's install command when present; fall back to -e .
    install_cmd = spec.get("install") or "python -m pip install -e ."
    rc, out = _venv_shell(install_cmd, venv_dir, repo_path, timeout=1800)
    if rc != 0:
        rc, out = _venv_shell(install_cmd + " --no-build-isolation", venv_dir, repo_path, timeout=1800)
    if rc != 0:
        return False, f"ENV_ERROR: repo build failed ({install_cmd!r}): {out[-400:]}"

    # Ensure a pytest runner exists — but never clobber a repo that *is* pytest.
    if _run([str(py), "-c", "import pytest"], timeout=60)[0] != 0:
        _pip(py, "install", "-q", "pytest", timeout=600)

    return True, "env ready"


# ---------------------------------------------------------------------------
# Test execution + grading
# ---------------------------------------------------------------------------

def _apply_test_patch(repo_path: Path, test_patch: str) -> tuple[bool, str]:
    """Apply the hidden test_patch onto the (agent-edited) working tree."""
    if not test_patch.strip():
        return True, "no test_patch"
    patch_file = repo_path / ".swebench_test.patch"
    patch_file.write_text(test_patch, encoding="utf-8")
    for extra in (["--whitespace=nowarn"], ["--3way", "--whitespace=nowarn"]):
        rc, out = _run(["git", "apply", *extra, str(patch_file)], cwd=repo_path, timeout=120)
        if rc == 0:
            return True, "applied"
    # Last resort: patch(1)
    rc, out = _run(["patch", "-p1", "-i", str(patch_file)], cwd=repo_path, timeout=120)
    return (rc == 0), (out[-300:] if rc != 0 else "applied via patch")


def _run_nodeids(py: Path, repo_path: Path, nodeids: list[str],
                 timeout: int) -> tuple[bool, str]:
    """
    Run the given pytest node ids. Return (all_passed, detail).

    Grading is by pytest's exit code, batched to stay under ARG_MAX: a batch
    exits 0 only when every selected test passed; ANDing across batches means all
    node ids passed. addopts is neutralised so a repo's own pytest.ini (coverage,
    -n auto, custom plugins) can't perturb the result.
    """
    if not nodeids:
        return True, "0 tests"
    env = {**os.environ, "PYTEST_ADDOPTS": ""}
    total = len(nodeids)
    failed_batches = 0
    last_out = ""
    for i in range(0, total, _PYTEST_BATCH):
        batch = nodeids[i:i + _PYTEST_BATCH]
        cmd = [str(py), "-m", "pytest", "-p", "no:cacheprovider",
               "-o", "addopts=", "-q", "--tb=no", *batch]
        rc, out = _run(cmd, cwd=repo_path, timeout=timeout, env=env)
        last_out = out
        if rc != 0:
            failed_batches += 1
    if failed_batches == 0:
        return True, f"{total} passed"
    return False, f"{failed_batches} batch(es) with failures; tail: {last_out.strip()[-200:]}"


def _django_module_labels(names: list[str], test_patch: str) -> list[str]:
    """The django test *apps* to run, e.g. 'test_utils'.

    Derived from the class paths of well-formed ids and from the app directories
    the test patch touches. We run whole apps rather than individual tests because
    ~22% of django's SWE-bench ids are docstrings with no derivable label; running
    the app and parsing the log covers both id shapes.
    """
    labels: set[str] = set()
    for n in names:
        m = re.match(r"^\S+\s+\(([^)]+)\)", n.strip())
        if m:
            labels.add(m.group(1).split()[0].split(".")[0])
    for f in _test_patch_files(test_patch):
        parts = Path(f).parts
        if len(parts) >= 2 and parts[0] == "tests":
            labels.add(parts[1].removesuffix(".py"))
    return sorted(labels)


# `<description> ... ok` / `FAIL` / `ERROR` / `skipped '...'`, as django prints at -v2.
_DJANGO_RESULT_RE = re.compile(
    r"^(?P<desc>.+?) \.\.\. (?P<status>ok|FAIL|ERROR|skipped.*|expected failure|unexpected success)\s*$",
    re.M,
)


def _run_django_tests(py: Path, repo_path: Path, names: list[str], test_patch: str,
                      timeout: int) -> tuple[bool, str]:
    """
    Grade django tests by running its own runner and parsing the verbose log.

    django does not use pytest: `tests/runtests.py` takes dotted app labels and
    needs its settings module (`tests/test_sqlite`). Crucially, ~22% of the
    instance's test ids are docstrings rather than `test_x (module.Class)`, so
    they cannot be passed as labels — instead we run the relevant apps at
    --verbosity 2 and match each requested id against the exact description
    django prints before ` ... <status>`. A test is satisfied by `ok` (or a skip,
    which is an environment difference rather than a regression); FAIL/ERROR or a
    missing entry counts as failure.
    """
    if not names:
        return True, "0 tests"
    labels = _django_module_labels(names, test_patch)
    if not labels:
        return False, "could not derive any django test app label"

    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    env.pop("DJANGO_SETTINGS_MODULE", None)
    rc, out = _run(
        [str(py), "tests/runtests.py", "--settings=test_sqlite",
         "--parallel", "1", "--verbosity", "2", *labels],
        cwd=repo_path, timeout=timeout, env=env,
    )

    results = {m.group("desc").strip(): m.group("status")
               for m in _DJANGO_RESULT_RE.finditer(out)}
    if not results:
        return False, f"no parsable django results (rc={rc}); tail: {out.strip()[-200:]}"

    bad = []
    for n in names:
        st = results.get(n.strip())
        if st is None or not (st == "ok" or st.startswith("skipped")):
            bad.append(f"{n[:60]}={st or 'missing'}")
    if bad:
        return False, f"{len(bad)}/{len(names)} not ok: " + "; ".join(bad[:3])
    return True, f"{len(names)} passed"


class SWEbenchVerifier(Verifier):
    """
    Grade one SWE-bench instance the official way ("resolved"):
      1. Build/reuse a per-(repo, version) venv and editable-install this checkout.
      2. Apply the hidden test_patch onto the agent's working tree.
      3. Require ALL FAIL_TO_PASS tests to pass AND ALL PASS_TO_PASS tests to pass.

    Environment-setup or patch-application problems are reported with an
    ENV_ERROR / PATCH_ERROR prefix so they can be told apart from genuine
    "agent's fix was wrong" failures.
    """

    def __init__(self, rec: dict, test_timeout: int = 1200) -> None:
        self._repo = rec["repo"]
        self._version = str(rec.get("version", "0"))
        self._test_patch = rec.get("test_patch", "")
        self._f2p = list(rec.get("FAIL_TO_PASS", []))
        self._p2p = list(rec.get("PASS_TO_PASS", []))
        self._timeout = test_timeout

    def __call__(self, repo_path: str) -> tuple[bool, str]:
        rp = Path(repo_path)

        spec = _spec_for(self._repo, self._version)
        kind = _runner_kind(spec)
        if kind is None:
            runner = (spec or {}).get("test_cmd", "unknown").split()[0]
            return False, (f"UNSUPPORTED: {self._repo} grades via {runner!r}, which has no "
                           "local runner; needs the official Docker harness")

        ok, detail = _ensure_env(self._repo, self._version, rp)
        if not ok:
            return False, detail  # already ENV_ERROR-prefixed

        ok, detail = _apply_test_patch(rp, self._test_patch)
        if not ok:
            return False, f"PATCH_ERROR: test_patch did not apply: {detail}"

        py = _venv_python(self._repo, self._version)
        if kind == "django":
            run = lambda names: _run_django_tests(py, rp, names, self._test_patch, self._timeout)
        else:
            # Resolve after the test patch is applied, so newly-added test files exist.
            run = lambda names: _run_nodeids(
                py, rp, _resolve_nodeids(names, self._test_patch, rp), self._timeout)

        f2p_ok, f2p_detail = run(self._f2p)
        if not f2p_ok:
            return False, f"unresolved: FAIL_TO_PASS not green ({f2p_detail})"
        p2p_ok, p2p_detail = run(self._p2p)
        if not p2p_ok:
            return False, f"regression: PASS_TO_PASS broke ({p2p_detail})"

        return True, f"resolved: F2P {len(self._f2p)}/{len(self._f2p)} ok, P2P {p2p_detail}"


# ---------------------------------------------------------------------------
# Instance -> TaskSpec
# ---------------------------------------------------------------------------

def record_to_spec(rec: dict, max_steps: int = 40, test_timeout: int = 1200) -> TaskSpec:
    """Convert one SWE-bench record into a gradable TaskSpec."""
    return TaskSpec(
        id=rec["instance_id"],
        description=problem_prompt(rec),
        verify=SWEbenchVerifier(rec, test_timeout=test_timeout),
        setup_hook=make_setup_hook(rec["repo"], rec["base_commit"],
                                   version=str(rec.get("version", "0"))),
        max_steps=max_steps,
    )


def venv_runtime_for(rec: dict):
    """Return a VenvRuntime bound to this instance's environment, or None if it
    hasn't been built. Passed to the tool registry so the agent's `test`/`shell`
    tools run inside the project's own venv and can actually execute its suite."""
    py = _venv_python(rec["repo"], str(rec.get("version", "0")))
    if not py.exists():
        logger.warning("no venv for %s (%s); agent tools fall back to LocalRuntime",
                       rec.get("instance_id"), py)
        return None
    from tools.runtime import VenvRuntime
    return VenvRuntime(py.parent.parent)


def load_swebench(
    split: str = "test",
    dataset: str = "SWE-bench_Lite",
    limit: int | None = None,
    start: int = 0,
    instances: list[str] | None = None,
    repos: list[str] | None = None,
    max_steps: int = 40,
    test_timeout: int = 1200,
) -> tuple[list[TaskSpec], list[dict]]:
    """
    Return (specs, records) for the selected SWE-bench instances.

    Selection precedence: explicit `instances` id list wins; otherwise optionally
    filter by `repos`, then apply [start : start+limit].
    """
    recs = load_records(split=split, dataset=dataset)

    if instances:
        wanted = set(instances)
        recs = [r for r in recs if r["instance_id"] in wanted]
    else:
        if repos:
            repo_set = set(repos)
            recs = [r for r in recs if r["repo"] in repo_set]
        recs = recs[start:(start + limit) if limit is not None else None]

    specs = [record_to_spec(r, max_steps=max_steps, test_timeout=test_timeout) for r in recs]
    return specs, recs
