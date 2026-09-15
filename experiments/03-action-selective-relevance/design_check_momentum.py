"""Stronger-momentum sweep: does a myopic true-dynamics planner fall behind?"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
import design_check as dc

dc.cem_plan.__defaults__ = (300, 30, 5, 0.7)      # N, K, iters, sd0: bigger budget
ep = dc.sample_episodes(100, np.random.default_rng(11))
print("100 episodes, T=50, CEM N=300 K=30 iters=5")
print("%-13s %-13s %3s %9s %8s %8s %9s %9s" %
      ("policy", "dynamics", "H", "ret/step", "late", "success", "chose_act", "final_px"))
for d, al in ((0.95, 0.0025), (0.97, 0.0015)):
    vt = al / (1 - d)
    print(f"-- d={d} a={al}: terminal {vt:.3f}/step, coast-to-stop {vt*d/(1-d):.2f} arena, "
          f"full-brake stop {np.log(.5)/np.log(d):.1f} steps")
    for pol, h in (("noop", 0), ("oracle", 1), ("oracle", 4), ("oracle", 8),
                   ("oracle", 15), ("colour_blind", 8)):
        t0 = time.time()
        r = dc.run(pol, ep, (d, al), T=50, H=max(h, 1))
        print("%-13s d=%.2f a=%.4f %3s %+9.3f %+8.3f %7.0f%% %8.0f%% %9.2f  (%.0fs)" %
              (pol, d, al, h or "-", r["ret"].mean(), r["late"].mean(),
               100 * r["success"].mean(), 100 * r["chose_active"].mean(),
               r["final_px"].mean(), time.time() - t0), flush=True)
