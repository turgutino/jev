"""Maximum-speed job queue for a GPU that is reserved for us (standard library only).
- Runs several jobs in parallel while the sum of their declared GPU memory ("need_mib") fits MAX_TOTAL_MIB (default 22500) and at most
  MAX_PARALLEL (default 3) run at once. A job with "exclusive": true runs alone (the 4B jobs) and holds back later jobs until it can start.
- "requires": a file that must exist before the job starts (the adapter of the training job it evaluates). If the job that produces that
  file has failed for good, the dependent job is marked skipped.
- "retry_args": extra arguments added from the second try on (e.g. --grad-ckpt after a possible out-of-memory failure).
- No throttling and no pauses. It only LOGS when another user's GPU process appears (different session id); it never touches foreign processes.
Own children only: sched/STOP_MAX terminates them. Writes only inside RUN. Launch detached:
  setsid nohup python scheduler_max.py RUN QUEUE.json > sched/max.out 2>&1 < /dev/null &
"""
import json
import os
import shutil
import subprocess
import sys
import time

RUN = sys.argv[1]
QUEUE = sys.argv[2]
SDIR = os.path.join(RUN, "sched")
os.makedirs(SDIR, exist_ok=True)
TAG = os.environ.get("SCHED_TAG", "max")
LOG = os.path.join(SDIR, TAG.upper() + "_LOG.txt")
STATE = os.path.join(SDIR, f"state_{TAG}.json")
STOP = os.path.join(SDIR, "STOP_" + TAG.upper())
PY = os.path.join(RUN, "venv", "bin", "python")
MAX_TOTAL = int(os.environ.get("MAX_TOTAL_MIB", 22500))
MAX_PAR = int(os.environ.get("MAX_PARALLEL", 3))
MAX_TRIES = 3
MIN_FREE_GB = {RUN: 3, os.environ.get("HF_HOME", RUN): float(os.environ.get("MIN_FREE_HF_GB", 1.0))}
MYSID = os.getsid(0)


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def gpu_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:  # noqa: BLE001
        return None


def foreign_pids():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=15).stdout.split()
    except Exception:  # noqa: BLE001
        return []
    res = []
    for p in out:
        try:
            sid = int(subprocess.run(["ps", "-o", "sid=", "-p", p], capture_output=True, text=True, timeout=10).stdout.strip() or -1)
        except Exception:  # noqa: BLE001
            continue
        if sid not in (-1, MYSID):
            res.append(p)
    return res


def disk_ok():
    for path, need in MIN_FREE_GB.items():
        try:
            if shutil.disk_usage(path).free / 2 ** 30 < need:
                return False, f"disk free below {need} GB at {path}"
        except OSError:
            pass
    return True, ""


def save(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, STATE)


def main():
    jobs = json.load(open(QUEUE, encoding="utf-8"))
    byid = {j["id"]: j for j in jobs}
    producer = {j["marker"]: j["id"] for j in jobs}
    st = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}
    for j in jobs:
        s = st.setdefault(j["id"], {"status": "pending", "tries": 0})
        s["tries"] = 0
        if os.path.exists(os.path.join(RUN, j["marker"])):
            s["status"] = "done"
        elif s["status"] in ("failed", "skipped_dep", "running"):
            s["status"] = "pending"
    save(st)
    log(f"scheduler_max start: {len(jobs)} jobs, MAX_TOTAL {MAX_TOTAL} MiB, MAX_PARALLEL {MAX_PAR}, sid {MYSID}")
    running = {}                     # id -> dict(proc, t0, peak)
    last_foreign, last_gpu_log = None, 0
    while True:
        # reap finished children
        for jid in list(running):
            r = running[jid]
            used = gpu_used()
            if used:
                r["peak"] = max(r["peak"], used)
            rc = r["proc"].poll()
            if rc is None:
                continue
            job, s = byid[jid], st[jid]
            s["tries"] += 1
            s.update(rc=rc, peak_total_mib=r["peak"], minutes=round((time.time() - r["t0"]) / 60, 1))
            ok = rc == 0 and os.path.exists(os.path.join(RUN, job["marker"]))
            s["status"] = "done" if ok else "pending" if s["tries"] < MAX_TRIES else "failed"
            log(f"END {jid} rc={rc} {s['minutes']} min, peak total GPU {r['peak']} MiB -> {s['status']} (try {s['tries']})")
            del running[jid]
            save(st)
        if os.path.exists(STOP):
            log("STOP_MAX found: terminating OUR children")
            for jid, r in running.items():
                r["proc"].terminate()
            time.sleep(5)
            log("stopped")
            return
        # mark dependents of failed producers
        for j in jobs:
            s = st[j["id"]]
            req = j.get("requires")
            if s["status"] == "pending" and req and not os.path.exists(os.path.join(RUN, req)):
                p = producer.get(req)
                if p is None or st[p]["status"] in ("failed", "skipped_dep"):
                    s["status"] = "skipped_dep"
                    log(f"skip {j['id']}: its input {req} cannot be produced ({p or 'no job produces it'} {st[p]['status'] if p else ''})")
                    save(st)
        pending = [j for j in jobs if st[j["id"]]["status"] == "pending" and j["id"] not in running]
        if not pending and not running:
            break
        # start jobs
        ok_disk, why = disk_ok()
        used_need = sum(byid[i]["need_mib"] for i in running)
        any_excl_running = any(byid[i].get("exclusive") for i in running)
        if ok_disk and not any_excl_running:
            for j in pending:
                req = j.get("requires")
                if req and not os.path.exists(os.path.join(RUN, req)):
                    continue                       # input not ready yet
                excl = j.get("exclusive", False)
                fits = (not running) if excl else (len(running) < MAX_PAR and used_need + j["need_mib"] <= MAX_TOTAL)
                if not fits:
                    if excl:
                        break                      # let running jobs drain, start nothing behind this job
                    continue
                s = st[j["id"]]
                args = list(j["args"]) + (j.get("retry_args", []) if s["tries"] >= 1 else [])
                env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.abspath(__file__)))
                env.update({k: str(v) for k, v in j.get("env", {}).items()})
                outlog = open(os.path.join(SDIR, j["id"] + ".log"), "a", encoding="utf-8")
                log(f"START {j['id']} (try {s['tries'] + 1}, need {j['need_mib']} MiB{', exclusive' if excl else ''}): {' '.join(args)[:200]}")
                proc = subprocess.Popen([PY, os.path.join(RUN, j.get("runner", "jf_max.py"))] + args, cwd=RUN, env=env, stdout=outlog,
                                        stderr=subprocess.STDOUT)
                running[j["id"]] = {"proc": proc, "t0": time.time(), "peak": 0}
                s["status"] = "running"
                used_need += j["need_mib"]
                save(st)
                if excl:
                    break
        elif not ok_disk and (time.time() - last_gpu_log) > 300:
            log(f"not starting new jobs: {why}")
            last_gpu_log = time.time()
        # passive monitoring (no action)
        fp = foreign_pids()
        if fp != last_foreign:
            log(f"foreign GPU processes now: {fp or 'none'}; GPU used {gpu_used()} MiB (monitoring only, no action taken)")
            last_foreign = fp
        time.sleep(10)
    failed = [i for i, s in st.items() if i in byid and s["status"] in ("failed", "skipped_dep")]
    log(f"QUEUE DONE; failed/skipped: {failed or 'none'}")
    open(os.path.join(SDIR, TAG.upper() + "_DONE"), "w").write(time.strftime("%F %T"))


if __name__ == "__main__":
    main()
