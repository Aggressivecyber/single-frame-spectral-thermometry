import runtime
import os
for name in ("cuda", "inductor", "triton"):
    (runtime.RUNTIME/name).mkdir(parents=True, exist_ok=True)
os.environ["CUDA_CACHE_PATH"] = str(runtime.RUNTIME/"cuda")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(runtime.RUNTIME/"inductor")
os.environ["TRITON_CACHE_DIR"] = str(runtime.RUNTIME/"triton")
