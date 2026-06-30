"""Measure per-image inference latency (model already loaded)."""
import glob
import time
from predict import predict, _load

_load()  # warm: pay startup cost once, not in the loop

imgs = sorted(glob.glob("dataset/images/*.jpeg"))[:20]

# Warmup -- first call hits cold caches (FFT plans, lazy imports).
predict(imgs[0])

times = []
for p in imgs:
    t0 = time.perf_counter()
    predict(p)
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
print(f"N         = {len(times)} images")
print(f"min       = {times[0]:.1f} ms")
print(f"median    = {times[len(times)//2]:.1f} ms")
print(f"mean      = {sum(times)/len(times):.1f} ms")
print(f"p95       = {times[int(len(times)*0.95)]:.1f} ms")
print(f"max       = {times[-1]:.1f} ms")