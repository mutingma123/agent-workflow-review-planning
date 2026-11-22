"""Parallel campaign runner for the Mac Studio.

    python kp_campaign.py plan     [--workers J] [--threads T] [--blocks X1,X7]
    python kp_campaign.py launch   [--workers J] [--threads T] [--blocks ...] [--retry-failed] [--detach]
    python kp_campaign.py preflight                # every block at tiny size, into runs/_preflight
    python kp_campaign.py status
    python kp_campaign.py stop

J worker processes each hold one Gurobi environment (one license session for
the whole campaign) and run one job at a time with T solver threads. The
default is T = 2 and J = (physical cores - 2) // T, the rule of the BiObj
Mac Studio campaigns: 11 workers x 2 threads on its 24 cores, with two cores
left for the notebook, the monitors and macOS. A job is sent to an idle
worker only when the free memory covers the per-job allowance, which is also
the solver's soft memory limit.

The campaign ends within campaign_hours (48) of its first launch. At the
first launch the time limit per solve is computed so that every job fits in
that budget (kp_blocks.budget); a relaunch keeps the first launch's deadline
and plan. Each job receives a deadline and divides its remaining time among
its solves, and a job that can no longer finish before the campaign deadline
is recorded as skipped. A relaunch after the deadline closes the campaign:
every unfinished job is recorded as skipped and CAMPAIGN_DONE.json is
written, so it never starts a second budget without --new-campaign.
`launch --detach` starts the scheduler in its own session under caffeinate
and returns, so a relaunch from a monitor or an agent outlives its caller.
`stop` writes runs/STOPPED.json, which tells the monitor not to relaunch;
the next launch removes it.
Every finished job writes runs/<block>/rep_<r>/<cell>.json atomically, so a
stopped campaign resumes where it stopped: finished jobs are skipped, a job
whose worker died is retried once, and a failed job reruns only with
--retry-failed. runs/status.json is rewritten every 10 seconds for the
monitors; runs/run.lock holds the scheduler's process id while it runs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import platform
import queue as queue_mod
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "runs"


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def machine() -> dict:
    d = dict(host=socket.gethostname(), system=platform.system(), machine=platform.machine(),
             python=platform.python_version(), logical_cpus=os.cpu_count())
    if platform.system() == "Darwin":
        d["chip"] = _sysctl("machdep.cpu.brand_string")
        d["physical_cpus"] = int(_sysctl("hw.physicalcpu") or 0) or os.cpu_count()
        d["performance_cpus"] = int(_sysctl("hw.perflevel0.physicalcpu") or 0) or None
        d["efficiency_cpus"] = int(_sysctl("hw.perflevel1.physicalcpu") or 0) or None
        d["memory_gb"] = round(int(_sysctl("hw.memsize") or 0) / 2 ** 30, 1)
    else:
        d["physical_cpus"] = os.cpu_count()
        try:
            for ln in open("/proc/cpuinfo"):
                if ln.startswith("model name"):
                    d["chip"] = ln.split(":", 1)[1].strip()
                    break
            for ln in open("/proc/meminfo"):
                if ln.startswith("MemTotal"):
                    d["memory_gb"] = round(int(ln.split()[1]) / 2 ** 20, 1)
        except OSError:
            pass
    try:
        import gurobipy as gp
        d["gurobi"] = ".".join(str(v) for v in gp.gurobi.version())
    except Exception as ex:
        d["gurobi"] = f"unavailable: {ex!r}"
    try:
        import numpy, scipy
        d["numpy"], d["scipy"] = numpy.__version__, scipy.__version__
    except Exception:
        pass
    return d


def available_gb() -> float:
    """Free plus reclaimable memory in GB."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = 16384
            first = out.splitlines()[0]
            if "page size of" in first:
                page = int(first.split("page size of")[1].split()[0])
            pages = 0
            for ln in out.splitlines()[1:]:
                key, _, val = ln.partition(":")
                if key.strip() in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                    pages += int(val.strip().rstrip("."))
            return pages * page / 2 ** 30
        except Exception:
            return float("inf")
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("inf")


def free_disk_gb(path: Path) -> float:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 2 ** 30
    except OSError:
        return float("inf")


def paths(runs: Path, job: dict):
    base = runs / job["block"] / f"rep_{job['rep']:03d}"
    return base / f"{job['cell']}.json", base / f"{job['cell']}.log", base / f"{job['cell']}.progress.json"


def settled(runs: Path, job: dict, retry_failed: bool) -> bool:
    """A job that needs no run: finished, or failed without --retry-failed.
    A job skipped for the deadline runs again in a new campaign."""
    rec = read_json(paths(runs, job)[0])
    if not rec:
        return False
    return rec.get("status") == "ok" or (rec.get("status") == "failed" and not retry_failed)


def license_info() -> dict:
    """Kind of Gurobi license file: 'wls' (Web License Service), 'file'
    (named-user or node license) or 'missing'. The file does not state how
    many sessions may run at once; probe_sessions measures that."""
    cands = [os.environ.get("GRB_LICENSE_FILE"), str(Path.home() / "gurobi.lic"),
             "/Library/gurobi/gurobi.lic", "/opt/gurobi/gurobi.lic"]
    for c in cands:
        if c and Path(c).is_file():
            try:
                text = Path(c).read_text(errors="replace").upper()
            except OSError:
                continue
            kind = "wls" if "WLSACCESSID" in text else "file"
            return dict(path=c, kind=kind)
    return dict(path=None, kind="missing")


def _probe(i, outq):
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        x = m.addVars(2600, vtype="B")
        m.setObjective(x.sum())
        m.addConstr(x.sum() >= 1)
        m.optimize()
        ok = m.Status == 2
        time.sleep(3.0)                     # hold the session while the others start
        env.dispose()
        outq.put((i, ok, None))
    except Exception as ex:
        outq.put((i, False, repr(ex)))


def probe_sessions(n: int) -> dict:
    """Start n Gurobi environments at once, each solving a 2,600-variable
    model, and count how many succeed: the number of workers the license
    supports. Each started environment is one license session."""
    ctx = mp.get_context("spawn")
    outq = ctx.Queue()
    procs = [ctx.Process(target=_probe, args=(i, outq)) for i in range(n)]
    for p in procs:
        p.start()
    res = []
    for _ in procs:
        try:
            res.append(outq.get(timeout=120))
        except queue_mod.Empty:
            break
    for p in procs:
        p.join(timeout=10)
    errors = sorted({e for _, ok, e in res if e})
    return dict(asked=n, succeeded=sum(1 for _, ok, _ in res if ok), errors=errors)


def default_workers(m: dict, threads: int) -> int:
    cores = m.get("physical_cpus") or os.cpu_count() or 1
    return max(1, (cores - 2) // threads) if cores > 4 else max(1, cores // threads)


# ------------------------------------------------------------------ worker
def _worker(wid: int, cfg: dict, runs: str, inbox, outbox, fake: bool):
    sys.path.insert(0, str(HERE))
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    runs = Path(runs)
    import kp_blocks as kb
    env = None
    if not fake:
        try:
            import kp_solve as ks
            env = ks.make_env(cfg["threads"])
        except Exception as ex:
            outbox.put(dict(kind="env_error", wid=wid, error=repr(ex)))
            return
    run = kb.fake_run_job if fake else kb.run_job
    outbox.put(dict(kind="ready", wid=wid, pid=os.getpid()))
    while True:
        job = inbox.get()
        if job is None:
            break
        res_path, log_path, prog_path = paths(runs, job)
        res_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rec = dict(job, started=now(), worker=wid, pid=os.getpid(), host=socket.gethostname())
        try:
            rec["result"] = run(job, cfg, env, log_file=str(log_path), progress=str(prog_path))
            rec["status"] = "ok"
        except Exception as ex:
            rec["status"] = "failed"
            rec["error"] = repr(ex)
            rec["trace"] = traceback.format_exc()
            rec["error_kind"] = "license" if "licen" in repr(ex).lower() else (
                "memory" if "memory" in repr(ex).lower() else "error")
        rec["wall"] = time.time() - t0
        rec["finished"] = now()
        write_json(res_path, rec)
        try:
            prog_path.unlink()
        except OSError:
            pass
        outbox.put(dict(kind="done", wid=wid, id=job["id"], status=rec["status"], wall=rec["wall"],
                        error_kind=rec.get("error_kind")))


# --------------------------------------------------------------- scheduler
class Campaign:
    def __init__(self, cfg, queue, runs: Path, workers: int, mem_gb: float, retry_failed: bool, fake: bool,
                 deadline: float = float("inf")):
        self.cfg, self.queue, self.runs = cfg, queue, runs
        self.deadline = deadline
        self.skipped = 0
        self.J, self.mem_gb, self.retry_failed, self.fake = workers, mem_gb, retry_failed, fake
        self.ctx = mp.get_context("spawn")
        self.outbox = self.ctx.Queue()
        self.workers = {}            # wid -> dict(proc, inbox, job, started, ready)
        self.attempts = read_json(runs / "_attempts.json", {}) or {}
        self.events, self.issues = [], []
        self.stop_flag = False
        self.counts = dict(ok=0, failed=0)
        self.env_errors = 0
        self.next_wid = 0

    def skip(self, job, left):
        """Record a job that cannot finish before the deadline; a failed record is kept."""
        res_path = paths(self.runs, job)[0]
        prev = read_json(res_path, {}) or {}
        if prev.get("status") != "failed":
            write_json(res_path, dict(job, status="skipped", error_kind="deadline",
                                      error=f"{max(0.0, left) / 60:.0f} min left before the campaign deadline",
                                      finished=now()))
        self.skipped += 1

    def log(self, msg):
        line = f"{now()} {msg}"
        print(line, flush=True)
        self.events = (self.events + [line])[-30:]
        with open(self.runs / "scheduler.log", "a") as f:
            f.write(line + "\n")

    def spawn(self):
        wid = self.next_wid
        self.next_wid += 1
        inbox = self.ctx.Queue()
        p = self.ctx.Process(target=_worker, args=(wid, self.cfg, str(self.runs), inbox, self.outbox, self.fake),
                             daemon=True)
        p.start()
        self.workers[wid] = dict(proc=p, inbox=inbox, job=None, started=None, ready=False)

    def status(self, pending):
        running = [dict(id=w["job"]["id"], set=w["job"]["set"], worker=wid, pid=w["proc"].pid, started=w["started"])
                   for wid, w in self.workers.items() if w["job"]]
        write_json(self.runs / "status.json", dict(
            updated=now(), scheduler_pid=os.getpid(), host=socket.gethostname(), workers=len(self.workers),
            workers_target=self.J, running=running, pending=len(pending), ok=self.counts["ok"],
            failed=self.counts["failed"], skipped=self.skipped, total=len(self.queue),
            deadline=dt.datetime.fromtimestamp(self.deadline).isoformat(timespec="seconds") if self.deadline < float("inf") else None,
            hours_left=round((self.deadline - time.time()) / 3600, 2) if self.deadline < float("inf") else None,
            time_limit=self.cfg.get("time_limit"), available_gb=round(available_gb(), 1),
            mem_gb_per_job=self.mem_gb, free_disk_gb=round(free_disk_gb(self.runs), 1),
            issues=self.issues[-20:], events=self.events[-15:], stopping=self.stop_flag))

    def run(self):
        pending = [j for j in self.queue if not settled(self.runs, j, self.retry_failed)]
        done_before = len(self.queue) - len(pending)
        self.log(f"start: {len(self.queue)} jobs, {done_before} settled, {len(pending)} to run, "
                 f"{self.J} workers x {self.cfg['threads']} threads, {self.mem_gb:.1f} GB per job")
        if pending and time.time() >= self.deadline:     # relaunch after the deadline: close the campaign
            for job in pending:
                self.skip(job, self.deadline - time.time())
            self.log(f"deadline passed: {len(pending)} unfinished jobs recorded as skipped")
            pending = []
        for _ in range(min(self.J, len(pending))):
            self.spawn()
        last_status = 0.0
        while (pending or any(w["job"] for w in self.workers.values())) and not self.stop_flag:
            # messages from workers
            try:
                while True:
                    msg = self.outbox.get(timeout=1.0)
                    w = self.workers.get(msg["wid"])
                    if msg["kind"] == "ready" and w:
                        w["ready"] = True
                    elif msg["kind"] == "env_error":
                        self.env_errors += 1
                        self.issues.append(dict(kind="license", time=now(), detail=msg["error"]))
                        self.log(f"worker {msg['wid']} could not start Gurobi: {msg['error']}")
                        if w:
                            w["proc"].join(timeout=5)
                            self.workers.pop(msg["wid"], None)
                        self.J = max(1, len(self.workers)) if self.workers else 0
                    elif msg["kind"] == "done" and w:
                        self.counts[msg["status"]] = self.counts.get(msg["status"], 0) + 1
                        self.log(f"{msg['status']} {msg['id']} in {msg['wall']:.0f}s")
                        if msg["status"] != "ok":
                            self.issues.append(dict(kind=msg.get("error_kind") or "error", time=now(), job=msg["id"]))
                        w["job"], w["started"] = None, None
            except queue_mod.Empty:
                pass
            # dead workers
            for wid, w in list(self.workers.items()):
                if w["proc"].exitcode is not None:
                    job = w["job"]
                    self.workers.pop(wid)
                    if job:
                        n = self.attempts.get(job["id"], 0) + 1
                        self.attempts[job["id"]] = n
                        write_json(self.runs / "_attempts.json", self.attempts)
                        self.log(f"worker {wid} died (exit {w['proc'].exitcode}) during {job['id']}, attempt {n}")
                        if n < 2:
                            pending.insert(0, job)
                        else:
                            res_path = paths(self.runs, job)[0]
                            write_json(res_path, dict(job, status="failed", error_kind="worker_died",
                                                      error=f"worker died {n} times (exit {w['proc'].exitcode})",
                                                      finished=now()))
                            self.counts["failed"] += 1
                            self.issues.append(dict(kind="worker_died", time=now(), job=job["id"]))
                    if pending and self.env_errors == 0:
                        self.spawn()
            if not self.workers and pending:
                self.log("no worker can start; stopping")
                break
            # dispatch
            for wid, w in self.workers.items():
                if not pending:
                    break
                if w["ready"] and w["job"] is None:
                    if available_gb() < self.mem_gb and any(x["job"] for x in self.workers.values()):
                        break                     # wait for memory while other jobs run
                    job = pending.pop(0)
                    if settled(self.runs, job, self.retry_failed):
                        continue
                    import kp_blocks as kb
                    left = self.deadline - time.time()
                    need = self.cfg["job_overhead"] + job["n_ev"] * min(30.0, self.cfg["ev_time_limit"]) + job["n_mip"] * 60.0
                    if left < need:
                        self.skip(job, left)
                        continue
                    job = dict(job, deadline=time.time() + min(left - 30.0, kb.job_seconds(job, self.cfg)))
                    w["job"], w["started"] = job, now()
                    w["inbox"].put(job)
            if time.time() - last_status >= 10:
                self.status(pending)
                last_status = time.time()
        self.status(pending)
        for w in self.workers.values():
            try:
                w["inbox"].put(None)
            except Exception:
                pass
        deadline = time.time() + (5 if not self.stop_flag else 15)
        for w in self.workers.values():
            w["proc"].join(timeout=max(0.1, deadline - time.time()))
            if w["proc"].is_alive():
                w["proc"].terminate()
        return pending


def _write_plan(runs: Path, cfg: dict, queue: list):
    plan = dict(written=now(), total=len(queue),
                sets={}, jobs=[{k: j[k] for k in ("id", "set", "block", "rep", "cell", "wc")} for j in queue])
    for j in queue:
        plan["sets"].setdefault(j["set"], []).append(j["id"])
    write_json(runs / "plan.json", plan)


def cmd_plan(args, cfg, queue, J, m, lic, budget_info):
    cores = m.get("physical_cpus") or os.cpu_count()
    by = {}
    for j in queue:
        d = by.setdefault(j["block"], dict(jobs=0, sets=set(), wc=0.0))
        d["jobs"] += 1
        d["sets"].add(j["set"])
        d["wc"] += j["wc"]
    print(f"machine: {m.get('chip')}, {cores} physical cores ({m.get('performance_cpus')} performance), "
          f"{m.get('memory_gb')} GB, Gurobi {m.get('gurobi')}")
    print(f"license: {lic['kind']} ({lic['path']})")
    print(f"{J} workers x {cfg['threads']} threads; budget {cfg['campaign_hours']:.0f} h; time limit "
          f"{cfg['time_limit']:.0f} s per solve; stop at gap {cfg['mip_gap']:.0%} (X7: {cfg['x7_mip_gap']:.0%})")
    if budget_info.get("cuts"):
        from collections import Counter
        print(f"replications cut to fit the budget: {dict(Counter(budget_info['cuts']))}; now {budget_info['reps']}")
    tot = 0.0
    for b, d in by.items():
        tot += d["wc"]
        print(f"  {b}: {d['jobs']} jobs in {len(d['sets'])} sets, worst case {d['wc'] / 3600:.1f} job-hours")
    print(f"total: {len(queue)} jobs, worst case {tot / 3600:.1f} job-hours, about "
          f"{tot / 3600 / J / cfg['efficiency']:.1f} h of wall time with {J} workers if every solve reaches its "
          f"time limit; solves that prove the gap earlier finish sooner")
    return dict(workers=J, job_hours=tot / 3600)


def main(argv=None):
    sys.path.insert(0, str(HERE))
    import kp_blocks as kb
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plan", "launch", "preflight", "status", "stop"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--blocks", default=None, help="comma-separated, for example X1,X7")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--config", default=None, help="JSON file of overrides of kp_blocks.DEFAULTS")
    ap.add_argument("--mem-gb", type=float, default=None, help="memory allowance per job")
    ap.add_argument("--fake", action="store_true", help="no solver: test the scheduler")
    ap.add_argument("--hours", type=float, default=None, help="campaign budget in hours (default 48)")
    ap.add_argument("--new-campaign", action="store_true", help="start a new budget instead of resuming the deadline")
    ap.add_argument("--time-limit", type=float, default=None, help="fixed seconds per solve instead of the budget")
    ap.add_argument("--detach", action="store_true",
                    help="launch in its own session under caffeinate, log to runs/scheduler.out, and return")
    args = ap.parse_args(argv)
    runs = Path(args.runs) if args.runs else (RUNS / "_preflight" if args.command == "preflight" else RUNS)
    if args.detach and args.command in ("launch", "preflight"):
        rest = [a for a in (sys.argv[1:] if argv is None else list(argv)) if a != "--detach"]
        cmd = [sys.executable, str(Path(__file__).resolve()), *rest]
        if shutil.which("caffeinate"):
            cmd = ["caffeinate", "-ims"] + cmd
        runs.mkdir(parents=True, exist_ok=True)
        with open(runs / "scheduler.out", "a") as out:
            p = subprocess.Popen(cmd, cwd=str(HERE), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        print(f"started the scheduler in its own session: pid {p.pid}; log {runs / 'scheduler.out'}")
        return 0
    if args.command == "status":
        print(json.dumps(read_json(runs / "status.json", {}), indent=1))
        return 0
    if args.command == "stop":
        write_json(runs / "STOPPED.json", dict(time=now(), by=os.getpid()))    # the monitor does not relaunch
        lock = read_json(runs / "run.lock", {})
        if lock.get("pid") and pid_alive(lock["pid"]):
            os.kill(int(lock["pid"]), signal.SIGTERM)
            print(f"sent SIGTERM to scheduler {lock['pid']}")
        else:
            print("no scheduler is running")
        return 0
    over = read_json(Path(args.config), {}) if args.config else {}
    cfg = kb.preflight_config() if args.command == "preflight" else kb.config(over)
    if args.threads:
        cfg["threads"] = args.threads
    if args.hours:
        cfg["campaign_hours"] = args.hours
    blocks = args.blocks.split(",") if args.blocks else None
    m = machine()
    lic = license_info()
    J = args.workers or default_workers(m, cfg["threads"])
    if args.command == "preflight":
        J = min(J, 2)
    old = read_json(runs / "campaign.json", {}) or {}
    # A relaunch keeps the first launch's deadline and plan, also after the deadline has passed: the
    # scheduler then records every unfinished job as skipped and closes the campaign.
    resume = bool(old.get("deadline")) and not args.new_campaign and args.command == "launch" \
        and old.get("blocks") == (blocks or cfg["blocks"])
    if resume:
        cfg = old["config"]
        deadline = float(old["deadline"])
        budget_info = old.get("budget", {})
    else:
        if args.time_limit:
            cfg["time_limit"] = args.time_limit
            budget_info = dict(time_limit=args.time_limit, fixed=True)
        elif args.command == "preflight" or "time_limit" in over:
            budget_info = dict(time_limit=cfg["time_limit"], fixed=True)
        else:
            budget_info = kb.budget(cfg, J, blocks)
            cfg = budget_info.pop("config")
        deadline = time.time() + cfg["campaign_hours"] * 3600.0
    queue = kb.build_queue(cfg, blocks)
    if args.command == "plan":
        cmd_plan(args, cfg, queue, J, m, lic, budget_info)
        return 0
    runs.mkdir(parents=True, exist_ok=True)
    lock_path = runs / "run.lock"
    lock = read_json(lock_path, {})
    if lock.get("pid") and pid_alive(lock["pid"]) and lock.get("host") == socket.gethostname():
        print(f"A campaign is already running (pid {lock['pid']}, started {lock.get('started')}). "
              f"Stop it with: python {Path(__file__).name} stop")
        return 3
    mem_total = m.get("memory_gb") or 16.0
    mem_gb = args.mem_gb or max(1.0, 0.8 * mem_total / J)
    if cfg.get("soft_mem_gb") is None:
        cfg["soft_mem_gb"] = round(mem_gb, 1)
    write_json(lock_path, dict(pid=os.getpid(), host=socket.gethostname(), started=now(), command=args.command))
    try:
        (runs / "STOPPED.json").unlink()
    except OSError:
        pass
    first = old.get("first_started") if resume else now()
    write_json(runs / "campaign.json", dict(
        first_started=first, started=now(), resumed=(old.get("resumed", []) if resume else []) + [now()],
        deadline=deadline, deadline_iso=dt.datetime.fromtimestamp(deadline).isoformat(timespec="seconds"),
        command=args.command, argv=sys.argv, python=sys.executable, workers=J, threads=cfg["threads"],
        mem_gb_per_job=mem_gb,
        blocks=blocks or cfg["blocks"], machine=m, license=lic, budget=budget_info, config=cfg))
    _write_plan(runs, cfg, queue)
    try:
        (runs / "CAMPAIGN_DONE.json").unlink()
    except OSError:
        pass
    camp = Campaign(cfg, queue, runs, J, mem_gb, args.retry_failed, args.fake, deadline)

    def handler(signum, frame):
        camp.stop_flag = True
        camp.log(f"signal {signum}: stopping after the running jobs are terminated")

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    try:
        left = camp.run()
        st = [(read_json(paths(runs, j)[0], {}) or {}).get("status") for j in queue]
        ok, failed, skipped = st.count("ok"), st.count("failed"), st.count("skipped")
        if not left and not camp.stop_flag:          # written before the lock goes, so no look sees a gap
            write_json(runs / "CAMPAIGN_DONE.json", dict(finished=now(), ok=ok, failed=failed, skipped=skipped,
                                                         total=len(queue), deadline=deadline))
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass
    camp.log(f"end: {ok} ok, {failed} failed, {skipped} skipped for the deadline, {len(left)} not run")
    return 0 if not left else 1


if __name__ == "__main__":
    sys.exit(main())
