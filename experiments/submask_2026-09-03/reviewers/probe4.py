import warnings; warnings.simplefilter("ignore")
import importlib.util, sys, numpy as np, time
spec = importlib.util.spec_from_file_location("sd", "super_degenerate.py")
sd = importlib.util.module_from_spec(spec); sys.argv=["sd","19","0","0"]
try: spec.loader.exec_module(sd)
except ZeroDivisionError: pass

# 1) is the `assert e[r]==eb[r]` a tautology?  feed h values that are NOT reachable-by-construction
print("=== assert e[r]==eb[r] on arbitrary 32-byte strings ===")
for name,h in [("all-ones",b"\xff"*32),("all-zeros",b"\x00"*32),
               ("random1",bytes(np.random.default_rng(1).integers(0,256,32,dtype=np.uint8).tolist())),
               ("random2",bytes(np.random.default_rng(2).integers(0,256,32,dtype=np.uint8).tolist()))]:
    rng=np.random.default_rng(99); ctx=sd.make_ctx(rng,19)
    try:
        sd.run(h, ctx, 1<<20, 5, name, 19); print(f"  {name}: assert PASSED (no constraint)")
    except AssertionError as ex: print(f"  {name}: assert FAILED {ex}")
