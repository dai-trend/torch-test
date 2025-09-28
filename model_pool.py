# model_pool.py
import os, threading, queue, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PRIME_Q = 2_147_483_647
def dot_mod_q(hidden: torch.Tensor, r_vec: torch.Tensor) -> int:
    # Ensure both tensors are on the same device
    device = hidden.device
    r_vec = r_vec.to(device)

    # Scale and convert to float for computation (avoid int64 issues on CUDA)
    scaled = torch.round(hidden * 1024)
    prod = torch.dot(scaled, r_vec.float())

    # Convert to int and apply modulo
    return int(prod.item()) % PRIME_Q

class GPUWorker:
    def __init__(self, gpu_id: int | None):
        """Create a worker pinned to a CUDA device or CPU when gpu_id is None.

        gpu_id: integer index of CUDA device, or None to use CPU.
        """
        self.gpu_id = gpu_id
        if gpu_id is None:
            # CPU fallback
            self.device = torch.device("cpu")
            # load model to CPU (use float32 for stability on CPU)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
        else:
            self.device = torch.device(f"cuda:{gpu_id}")
            # one independent copy per GPU
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            self.model =  AutoModelForCausalLM.from_pretrained(MODEL_ID, use_safetensors=True)
            self.model = self.model.to(self.device).eval()
        # set eval mode for inference
        try:
            self.model.eval()
        except Exception:
            pass
        self.lock = threading.Lock()       # 1 request at a time per worker
        
    
    @torch.inference_mode()
    def compute_s_vals_and_logprobs(self, prompt_length: int, all_token_ids: list[int], r_vec: torch.Tensor) -> Tuple[List[int], List[float]]:
        """Compute s values for a list of token ids using this GPU's model.

        all_token_ids: list of token ids (integers). We assume these are already
        token ids (not raw text).
        """
        with self.lock:                    # exclusive access to this GPU
            s_vals: List[int] = []
            # Build an input_ids tensor directly since caller provides token ids
            token_tensor = torch.tensor([all_token_ids], dtype=torch.long, device=self.device)
            model_outputs = self.model(input_ids=token_tensor, output_hidden_states=True)
            h_layer = model_outputs.hidden_states[-1][0]
            for pos in range(len(all_token_ids)):
                if pos < h_layer.size(0):
                    s_val = dot_mod_q(h_layer[pos], r_vec)
                    s_vals.append(s_val)

            logits = model_outputs.logits[0]  # [T, V]
            comp_logprobs: list[float] = []
            for t_idx in range(prompt_length, len(all_token_ids)):
                prev_t = t_idx - 1
                if prev_t < 0 or prev_t >= logits.size(0):
                    comp_logprobs.append(0.0)
                    continue
                step_logits = logits[prev_t]
                # log softmax for numerical stability
                log_probs = torch.log_softmax(step_logits, dim=-1)
                tok_id = int(all_token_ids[t_idx])
                comp_logprobs.append(float(log_probs[tok_id].item()))
            all_logprobs = [0.0] * prompt_length + comp_logprobs[: len(all_token_ids) - prompt_length]
            return s_vals, all_logprobs
class ModelPool:
    def __init__(self, gpu_list: List[int]):
        # if gpu_list is empty, create a single CPU worker for safety
        if not gpu_list:
            self.workers = [GPUWorker(None)]
        else:
            self.workers = [GPUWorker(g) for g in gpu_list]
        self.q = queue.Queue()
        # Create one thread per worker and bind the worker index to the thread so
        # each thread consistently services a single GPUWorker.
        self.threads = [threading.Thread(target=self._serve, args=(i,), daemon=True)
                        for i in range(len(self.workers))]
        for t in self.threads:
            t.start()

    def _serve(self, worker_index: int):
        """Thread loop: pull job, run, put result. Each thread is bound to a
        specific worker_index chosen at thread creation time.
        """
        worker = self.workers[worker_index]
        while True:
            job = self.q.get()
            if job is None:                # poison pill
                break
            prompt_length, all_token_ids, r_vec, future = job
            try:
                future.set_result(worker.compute_s_vals_and_logprobs(prompt_length, all_token_ids, r_vec))
            except Exception as e:
                future.set_exception(e)

    def submit(self, prompt_length, all_token_ids: list[int], r_vec: torch.Tensor):
        """Non-blocking submit. Returns a concurrent.futures.Future.

        all_token_ids: list of token ids (integers).
        r_vec: 1D tensor of floats.
        """
        import concurrent.futures as fut
        future: fut.Future = fut.Future()
        self.q.put((prompt_length, all_token_ids, r_vec, future))
        return future