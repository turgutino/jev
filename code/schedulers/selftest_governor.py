"""Local self-test of the decision logic with a fake clock and a fake nvidia-smi (no GPU, no server needed).
Run: python selftest_governor.py"""
import polite_gpu as pg

TOTAL = 24564
p = pg.Policy(need_mib=9000, reserve_mib=10240)


def make(r):
    clock = {"t": 0.0}
    slept = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    g = pg.Governor(p, ours_fn=lambda: r["ours"], query_fn=lambda: (r["used"], TOTAL), sleep_fn=sleep,
                    now_fn=lambda: clock["t"], release_fn=lambda: r.update(released=r.get("released", 0) + 1))
    return g, clock, slept


def advance(g, clock, seconds, step=5):
    for _ in range(int(seconds / step)):
        clock["t"] += step
        g.tick()


# 1) raw decisions (others = used - ours)
assert pg.raw_decision(1, TOTAL, 0, p)[0] == pg.IDLE
assert pg.raw_decision(5000, TOTAL, 0, p)[0] == pg.SLOW        # 24564-5000-9000 = 10564 >= 10240
assert pg.raw_decision(5600, TOTAL, 0, p)[0] == pg.PAUSE       # 24564-5600-9000 = 9964 < 10240
assert pg.raw_decision(8700, TOTAL, 0, p)[0] == pg.PAUSE       # seen neighbour 8.7 GB: with reserve 10 GB we must wait
assert pg.raw_decision(15700, TOTAL, 0, p)[0] == pg.PAUSE
p6 = pg.Policy(need_mib=9000, reserve_mib=6144)
assert pg.raw_decision(8700, TOTAL, 0, p6)[0] == pg.SLOW       # with reserve 6 GB the 8.7 GB neighbour is tolerated
assert pg.raw_decision(9400, TOTAL, 0, p6)[0] == pg.SLOW and pg.raw_decision(9500, TOTAL, 0, p6)[0] == pg.PAUSE
# our own memory must not count as "others"
assert pg.raw_decision(9000, TOTAL, 9000, p)[0] == pg.IDLE

# 2) hysteresis: starts PAUSE, needs 60 s to leave PAUSE on an idle GPU, then 180 s?  (PAUSE -> raw IDLE after hold_resume_s)
r = {"used": 1, "ours": 0}
g, c, s = make(r)
assert g.mode == pg.PAUSE
advance(g, c, 55)
assert g.mode == pg.PAUSE
advance(g, c, 10)
assert g.mode == pg.IDLE

# 3) neighbour appears (5 GB): IDLE -> SLOW immediately; huge neighbour -> PAUSE immediately
r["used"] = 5000 + 8000; r["ours"] = 8000     # our job holds 8 GB, neighbour 5 GB
g.tick(force=True)
assert g.mode == pg.SLOW, g.mode
r["used"] = 15700 + 8000
g.tick(force=True)
assert g.mode == pg.PAUSE

# 4) while paused: checkpoint blocks, releases cache, and resumes only after the neighbour is gone and the hold has passed
n_before = r.get("released", 0)
state = {"calls": 0}
orig_query = g.query_fn


def scripted():
    state["calls"] += 1
    return (r["used"], TOTAL) if state["calls"] < 4 else (1, TOTAL)       # neighbour disappears after a few polls


g.query_fn = scripted
r["ours"] = 2000
g.checkpoint()
assert r.get("released", 0) == n_before + 1
assert g.mode in (pg.SLOW, pg.IDLE)
assert sum(s) >= 60 - 1e-9, sum(s)          # waited at least the 60 s hold

# 5) SLOW throttles proportionally (duty 0.25 -> sleep = 3 x work, capped)
r = {"used": 4000 + 3000, "ours": 3000}
g, c, s = make(r)
advance(g, c, 70)                              # PAUSE -> SLOW after hold (neighbour 4 GB)
assert g.mode == pg.SLOW, g.mode
s.clear()
g._last_exit = c["t"] - 2.0                    # we worked 2 s since the last pause point
g.checkpoint()
assert abs(s[-1] - 6.0) < 1e-6, s              # 2 s work -> 6 s sleep (<= cap 8 s)

# 6) nvidia-smi failure => PAUSE (safe)
def boom():
    raise RuntimeError("no nvidia-smi")
g.query_fn = boom
g.tick(force=True)
assert g.mode == pg.PAUSE
print("all governor self-tests passed")
