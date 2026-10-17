"""Polite job queue for the shared GPU (standard library only). Runs jobs from queue.json one at a time.
Rules: never touches any process except its own child; writes only inside RUN; STOP file ends it (own child is terminated);
a job starts only when the policy allows (not PAUSE for hold_resume_s); finished jobs (marker file exists) are skipped on restart;
failed jobs are retried up to MAX_TRIES. Launch detached:  setsid nohup python scheduler.py RUN QUEUE.json > sched.out 2>&1 &
"""
import json
import os
import shutil
import subprocess
import sys
import time

import polite_gpu as pg

RUN = sys.argv[1]
QUEUE = sys.argv[2]
SDIR = os.path.join(RUN, "sched")
os.makedirs(SDIR, exist_ok=True)
LOG = os.path.join(SDIR, "SCHED_LOG.txt")
STATE = os.path.join(SDIR, "state.json")
STOP = os.path.join(SDIR, "STOP")
PY = os.path.join(RUN, "venv", "bin", "python")
MAX_TRIES = 2
MIN_FREE_DISK_GB = 30


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state():
    return json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, STATE)


def stopped():
    return os.path.exists(STOP)


def wait_admission(job):
    pol = pg.Policy(need_mib=job["need_mib"], reserve_mib=int(os.environ.get("POLITE_RESERVE_MIB", 10240)))
    ok_since = None
    last_msg = None
    while not stopped():
        free_gb = shutil.disk_usage(RUN).free / 2 ** 30
        try:
            used, total = pg.query_gpu()
            mode, why = pg.raw_decision(used, total, 0, pol)
        except Exception as e:  # noqa: BLE001
            mode, why = pg.PAUSE, f"nvidia-smi failed: {type(e).__name__}"
        if free_gb < MIN_FREE_DISK_GB:
            mode, why = pg.PAUSE, f"disk free {free_gb:.0f} GB < {MIN_FREE_DISK_GB}"
        if mode == pg.PAUSE:
            ok_since = None
            if why != last_msg:
                log(f"waiting to start {job['id']}: {why}")
                last_msg = why
        else:
            ok_since = ok_since or time.monotonic()
            if time.monotonic() - ok_since >= pol.hold_resume_s:
                log(f"admission OK for {job['id']} ({pg.NAMES[mode]}: {why})")
                return True
        time.sleep(15)
    return False


def run_job(job, st):
    env = dict(os.environ, POLITE_NEED_MIB=str(job["need_mib"]), POLITE_LOG=os.path.join(SDIR, "polite_" + job["id"] + ".log"),
               PYTHONPATH=os.path.dirname(os.path.abspath(__file__)))
    cmd = [PY, os.path.join(RUN, "jf_polite.py")] + job["args"]
    outlog = open(os.path.join(SDIR, job["id"] + ".log"), "a", encoding="utf-8")
    log(f"START {job['id']} (try {st[job['id']]['tries'] + 1}): {' '.join(cmd)[:200]}")
    t0 = time.time()
    child = subprocess.Popen(cmd, cwd=RUN, env=env, stdout=outlog, stderr=subprocess.STDOUT)
    peak = 0
    deadline = t0 + job.get("timeout_s", 72 * 3600)
    while child.poll() is None:
        try:
            peak = max(peak, pg.query_gpu()[0])
        except Exception:  # noqa: BLE001
            pass
        if stopped() or time.time() > deadline:
            log(f"terminating OUR child pid {child.pid} ({'STOP' if stopped() else 'timeout'})")
            child.terminate()
            try:
                child.wait(60)
            except subprocess.TimeoutExpired:
                child.kill()
            break
        time.sleep(20)
    rc = child.returncode
    log(f"END {job['id']} rc={rc} elapsed={(time.time() - t0) / 60:.1f} min, peak total GPU memory seen {peak} MiB")
    return rc, peak


def main():
    jobs = json.load(open(QUEUE, encoding="utf-8"))
    try:
        os.nice(19)
    except OSError:
        pass
    st = load_state()
    log(f"scheduler start; {len(jobs)} jobs; sid={os.getsid(0)}")
    for job in jobs:
        s = st.setdefault(job["id"], {"status": "pending", "tries": 0})
        marker = os.path.join(RUN, job["marker"])
        if os.path.exists(marker):
            s["status"] = "done"
            save_state(st)
            log(f"skip {job['id']}: marker exists")
            continue
        while s["tries"] < MAX_TRIES and not stopped():
            if not wait_admission(job):
                break
            rc, peak = run_job(job, st)
            s["tries"] += 1
            s.update(rc=rc, peak_mib=peak)
            if rc == 0 and os.path.exists(marker):
                s["status"] = "done"
                save_state(st)
                break
            s["status"] = "failed"
            save_state(st)
            time.sleep(30)
        if stopped():
            log("STOP file found; exiting")
            return
        if s["status"] != "done":
            log(f"giving up on {job['id']} after {s['tries']} tries; continuing with the next job")
    log("QUEUE DONE")
    open(os.path.join(SDIR, "DONE"), "w").write(time.strftime("%F %T"))


if __name__ == "__main__":
    main()
