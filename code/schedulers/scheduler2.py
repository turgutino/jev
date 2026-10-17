"""Polite job queue for the shared GPU (standard library only). Runs jobs from a JSON list one at a time.
Rules: never touches any process except its own child; writes only inside RUN; sched/STOP ends it (own child is terminated);
a job starts only when the policy allows (not PAUSE for hold_resume_s); jobs whose marker file exists are skipped on restart;
failed jobs are retried up to MAX_TRIES. Extra job fields: "idle_only" (start only after the GPU has been empty for "hold_s"),
"pause_queue_after" (scheduler exits after this job so the plan can be reviewed), "env" (extra environment variables).
Launch detached:  setsid nohup python scheduler2.py RUN QUEUE.json > sched2.out 2>&1 < /dev/null &
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
RESERVE = int(os.environ.get("POLITE_RESERVE_MIB", 10240))


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


def hold_for(job):
    return job.get("hold_s", 300) if job.get("idle_only") else pg.Policy().hold_resume_s


def admissible_now(job):
    """(ok, why): would this job be allowed to start at this very moment?"""
    pol = pg.Policy(need_mib=job["need_mib"], reserve_mib=RESERVE)
    free_gb = shutil.disk_usage(RUN).free / 2 ** 30
    if free_gb < MIN_FREE_DISK_GB:
        return False, f"disk free {free_gb:.0f} GB < {MIN_FREE_DISK_GB}"
    try:
        used, total = pg.query_gpu()
        mode, why = pg.raw_decision(used, total, 0, pol)
    except Exception as e:  # noqa: BLE001
        return False, f"nvidia-smi failed: {type(e).__name__}"
    if job.get("idle_only") and mode != pg.IDLE:
        return False, f"idle_only job: GPU not empty ({why})"
    return mode != pg.PAUSE, f"{pg.NAMES[mode]}: {why}"


def run_job(job, st):
    env = dict(os.environ, POLITE_NEED_MIB=str(job["need_mib"]), POLITE_RESERVE_MIB=str(RESERVE),
               POLITE_LOG=os.path.join(SDIR, "polite_" + job["id"] + ".log"), PYTHONPATH=os.path.dirname(os.path.abspath(__file__)))
    env.update({k: str(v) for k, v in job.get("env", {}).items()})
    cmd = [PY, os.path.join(RUN, "jf_polite2.py")] + job["args"]
    outlog = open(os.path.join(SDIR, job["id"] + ".log"), "a", encoding="utf-8")
    log(f"START {job['id']} (try {st[job['id']]['tries'] + 1}): {' '.join(cmd)[:220]}")
    t0 = time.time()
    child = subprocess.Popen(cmd, cwd=RUN, env=env, stdout=outlog, stderr=subprocess.STDOUT)
    peak = 0
    deadline = t0 + job.get("timeout_s", 96 * 3600)
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
    log(f"scheduler2 start; {len(jobs)} jobs; reserve {RESERVE} MiB; sid={os.getsid(0)}")
    for job in jobs:
        s = st.setdefault(job["id"], {"status": "pending", "tries": 0})
        s["tries"] = 0          # tries are per scheduler start
        if os.path.exists(os.path.join(RUN, job["marker"])):
            s["status"] = "done"
            log(f"skip {job['id']}: marker exists")
    save_state(st)
    ok_since, last_why = {}, {}
    while not stopped():
        todo = [j for j in jobs if st[j["id"]]["status"] != "done" and st[j["id"]]["tries"] < MAX_TRIES]
        if not todo:
            break
        chosen = None
        # priority = list order: a later job may start only while every earlier job is blocked right now;
        # an earlier job that is admissible but still inside its hold time is waited for (nothing behind it starts).
        for job in todo:
            ok, why = admissible_now(job)
            if not ok:
                ok_since.pop(job["id"], None)
                if last_why.get(job["id"]) != why:
                    log(f"blocked {job['id']}: {why}")
                    last_why[job["id"]] = why
                continue
            ok_since.setdefault(job["id"], time.monotonic())
            if time.monotonic() - ok_since[job["id"]] >= hold_for(job):
                chosen = job
            break
        if chosen is None:
            time.sleep(15)
            continue
        job, s = chosen, st[chosen["id"]]
        ok_since.pop(job["id"], None)
        log(f"admission OK for {job['id']}")
        rc, peak = run_job(job, st)
        s["tries"] += 1
        s.update(rc=rc, peak_mib=peak)
        if rc == 0 and os.path.exists(os.path.join(RUN, job["marker"])):
            s["status"] = "done"
        else:
            s["status"] = "failed"
            if s["tries"] >= MAX_TRIES:
                log(f"giving up on {job['id']} after {s['tries']} tries; continuing with the next job")
        save_state(st)
        if stopped():
            break
        if job.get("pause_queue_after"):
            log(f"pause_queue_after set on {job['id']}: scheduler exits for review (peak {s.get('peak_mib')} MiB)")
            return
        time.sleep(30)
    if stopped():
        log("STOP file found; exiting")
        return
    log("QUEUE DONE")
    open(os.path.join(SDIR, "DONE"), "w").write(time.strftime("%F %T"))


if __name__ == "__main__":
    main()
