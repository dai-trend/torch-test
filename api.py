import asyncio
from fastapi import FastAPI
from typing import Optional
try:
    # Preferred when running as a package (uvicorn torch_infer.api:app)
    from .model_pool import ModelPool
except Exception:
    # Fallback when running from inside the torch_infer directory as a script
    # (uvicorn api:app) where relative imports fail.
    # Use importlib to perform a best-effort import without static linters failing
    import importlib
    ModelPool = importlib.import_module('model_pool').ModelPool
import torch
import pydantic

app = FastAPI()
# Lazy-init the pool during startup so importing this module doesn't force
# heavy model downloads / GPU initialization at import time.
pool = None


@app.on_event("startup")
def init_pool():
    global pool
    # Detect CUDA devices; if none found, create CPU worker by passing empty list
    try:
        if torch.cuda.is_available():
            # ngpu = torch.cuda.device_count() // 2  # use half the GPUs
            ngpu = torch.cuda.device_count()
            gpu_list = list(range(ngpu))
        else:
            gpu_list = []
    except Exception:
        gpu_list = []
    pool = ModelPool(gpu_list=gpu_list)


@app.on_event("shutdown")
def shutdown_pool():
    # Send poison pills to worker threads so they exit cleanly.
    global pool
    if pool is None:
        return
    for _ in pool.workers:
        pool.q.put(None)


class Req(pydantic.BaseModel):
    prompt_length: int
    all_token_ids: list[int]
    r_vec: list[float]


@app.post("/score")
async def score(req: Req):
    """Endpoint expects token ids and a float vector r_vec.

    It converts r_vec to a torch tensor (CPU) and submits the job. The model
    pool threads will move tensors to the GPU as needed.
    """
    r_tensor = torch.tensor(req.r_vec)
    # submit returns a concurrent.futures.Future
    fut = pool.submit(req.prompt_length, req.all_token_ids, r_tensor)
    # convert to awaitable
    s_vals, all_logprobs = await asyncio.wrap_future(fut)
    return {
        "s_vals": s_vals,
        "all_logprobs": all_logprobs  
    }
# pm2 start "uvicorn torch_infer.api:app --host 0.0.0.0 --port 8001 --workers 1" --name torch_infer_server