"""Polite GPU sharing: decision logic + in-process governor. Standard library only (torch is imported lazily).

Goal (user rule): never kill or break another user's run. Slowing them is acceptable, memory pressure is not.
Modes:  IDLE  = nobody else on the GPU for a while -> run at full speed
        SLOW  = others present but plenty of free memory is left after our need -> duty-cycle throttle
        PAUSE = others present and free memory is tight (or dropping) -> stop at the next pause point, release cached memory, wait
Our own footprint is capped and never grows beyond the job's declared need.  We only ever touch our own processes.
"""
import json
import os
import subprocess
import time
from dataclasses import dataclass

PAUSE, SLOW, IDLE = 0, 1, 2
NAMES = {PAUSE: "PAUSE", SLOW: "SLOW", IDLE: "IDLE"}
CONTEXT_MIB = 500  # CUDA context, counted as ours but not visible to torch's allocator


@dataclass
class Policy:
    need_mib: int = 9000          # declared peak GPU memory of the job (incl. context)
    reserve_mib: int = 10240      # free memory that must remain for other users after our need
    idle_others_mib: int = 1500   # others using less than this = GPU idle
    hold_resume_s: float = 60     # raw decision must hold this long before PAUSE -> SLOW/IDLE
    hold_idle_s: float = 180      # ... and this long before SLOW -> IDLE
    duty: float = 0.25            # SLOW: fraction of time we compute (sleep = work * (1/duty - 1))
    poll_s: float = 5
    max_sleep_s: float = 8


def query_gpu(idx=None):
    idx = os.environ.get("POLITE_GPU_INDEX", "0") if idx is None else idx
    out = subprocess.run(["nvidia-smi", f"--id={idx}", "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    used, total = [int(x) for x in out.stdout.strip().split(",")]
    return used, total


def raw_decision(used, total, ours, p: Policy):
    others = max(0, used - ours)
    free_total = total - used
    if others > p.idle_others_mib and free_total < 0.6 * p.reserve_mib:
        return PAUSE, f"free memory low ({free_total} MiB), others {others} MiB"
    if others <= p.idle_others_mib:
        return IDLE, f"others {others} MiB"
    free_after = total - others - p.need_mib
    if free_after >= p.reserve_mib:
        return SLOW, f"others {others} MiB, free after our need {free_after} MiB"
    return PAUSE, f"others {others} MiB, free after our need only {free_after} MiB < reserve {p.reserve_mib}"


class Governor:
    def __init__(self, policy: Policy, ours_fn=lambda: 0, release_fn=lambda: None, query_fn=query_gpu,
                 sleep_fn=time.sleep, now_fn=time.monotonic, log_path=None):
        self.p, self.ours_fn, self.release_fn = policy, ours_fn, release_fn
        self.query_fn, self.sleep, self.now = query_fn, sleep_fn, now_fn
        self.log_path = log_path
        self.mode = PAUSE                      # conservative start
        self._cand_since = None
        self._last_poll = -1e9
        self._last_exit = now_fn()
        self.last_reason = "start"

    def _log(self, msg):
        line = f"{time.strftime('%F %T')} [polite] {msg}"
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def tick(self, force=False):
        t = self.now()
        if not force and t - self._last_poll < self.p.poll_s:
            return self.mode
        self._last_poll = t
        try:
            used, total = self.query_fn()
        except Exception as e:  # noqa: BLE001 - unknown state: stay/enter PAUSE (safe)
            raw, why = PAUSE, f"nvidia-smi failed: {type(e).__name__}"
            used = total = None
        else:
            raw, why = raw_decision(used, total, self.ours_fn(), self.p)
        old = self.mode
        if raw < self.mode:                    # more restrictive: immediately
            self.mode, self._cand_since = raw, None
        elif raw > self.mode:                  # more permissive: only after it held long enough
            if self._cand_since is None:
                self._cand_since = t
            hold = self.p.hold_resume_s if self.mode == PAUSE else self.p.hold_idle_s
            if t - self._cand_since >= hold:
                self.mode, self._cand_since = raw, None
        else:
            self._cand_since = None
        self.last_reason = why
        if self.mode != old:
            self._log(f"{NAMES[old]} -> {NAMES[self.mode]} (raw {NAMES[raw]}: {why}; used={used} total={total})")
        return self.mode

    def checkpoint(self):
        """Call at pause points (between examples). Blocks while PAUSE, sleeps proportionally while SLOW."""
        work = max(0.0, self.now() - self._last_exit)
        self.tick()
        if self.mode == PAUSE:
            self.release_fn()
            self._log(f"paused, cached memory released ({self.last_reason})")
            while self.mode == PAUSE:
                self.sleep(10)
                self.tick(force=True)
            self._log(f"resumed in {NAMES[self.mode]}")
        elif self.mode == SLOW and self.p.duty < 1:
            self.sleep(min(self.p.max_sleep_s, work * (1.0 / self.p.duty - 1.0)))
        self._last_exit = self.now()


def install_torch_governor(policy: Policy, log_path=None):
    """Cap this process's CUDA memory to the declared need and return a Governor wired to torch."""
    import torch
    mib = 2 ** 20
    total_mib = torch.cuda.get_device_properties(0).total_memory / mib
    cap_mib = max(1.0, policy.need_mib - CONTEXT_MIB)
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_mib / total_mib), 0)

    def ours():
        return torch.cuda.memory_reserved(0) / mib + CONTEXT_MIB

    def release():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return Governor(policy, ours_fn=ours, release_fn=release, log_path=log_path)


def policy_from_env():
    g = os.environ.get
    return Policy(need_mib=int(g("POLITE_NEED_MIB", 9000)), reserve_mib=int(g("POLITE_RESERVE_MIB", 10240)),
                  duty=float(g("POLITE_DUTY", 0.25)))
