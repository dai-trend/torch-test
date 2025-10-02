# model_pool.py
import os, threading, queue, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import math

PROOF_NUM_BUCKETS = 16 
LAYER_INDEX = -1
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PRIME_Q = 2_147_483_647
# Top-K activation selection (focus on stable, important features)
PROOF_TOPK = 256

# Logarithmic bucketing parameters
PROOF_NUM_BUCKETS = 16  # Buckets per sign

# Small bounded coefficients for sketch robustness
PROOF_COEFF_RANGE = 127  # r ∈ [-127, 127]


def log_magnitude_bucket(value: float, num_buckets: int = PROOF_NUM_BUCKETS) -> int:
    """Map activation to logarithmic magnitude bucket with sign preservation.

    Logarithmic bucketing provides natural robustness:
    - Small values: coarse bins (where drift happens)
    - Large values: finer bins (where we have precision)
    - Matches floating-point representation behavior

    Args:
        value: Activation value to bucket
        num_buckets: Number of buckets per sign (default: 16)

    Returns:
        Signed bucket index in [-num_buckets+1, 0, num_buckets-1]
    """
    abs_val = abs(value)

    # Deadzone for near-zero values
    if abs_val < 1e-6:
        return 0

    # Logarithmic scale: map log2(|x|+1) to bucket range
    # Typical hidden state range: [-3, 3] → log2 range ~ [0, 2]
    # Scale factor maps this to [0, num_buckets)
    log_val = math.log2(abs_val + 1.0)
    # TODO: come up with a more robust approach for measuring max log value
    scale_factor = num_buckets / 10.0  # Assuming max log value ~ 10
    bucket = int(log_val * scale_factor)
    bucket = max(0, min(num_buckets - 1, bucket))

    # Preserve sign
    return bucket if value >= 0 else -bucket

def create_commitment(
        hidden_state: torch.Tensor, r_vec: torch.Tensor, position: int
    ) -> dict:
        """Create commitment for a single token position.

        Args:
            hidden_state: Hidden vector at position [hidden_dim]
            r_vec: Coefficient vector [topk]
            position: Token position (for metadata)

        Returns:
            Commitment dict with sketch, indices, ranks, histogram
        """
        # Step 1: Select top-k activations by absolute magnitude
        abs_hidden = torch.abs(hidden_state)
        topk_result = torch.topk(abs_hidden, k=PROOF_TOPK)
        indices = topk_result.indices  # [topk]
        values = hidden_state[indices]  # [topk] with signs preserved

        # Step 2: Logarithmic bucketing
        buckets = torch.tensor(
            [log_magnitude_bucket(val.item(), PROOF_NUM_BUCKETS) for val in values],
            dtype=torch.int8,
        )

        # Step 3: Compute sketch via dot product with small coefficients
        sketch = torch.dot(buckets.to(torch.int32), r_vec.to(torch.int32))
        sketch_val = int(sketch.item()) % PRIME_Q

        # Step 4: Rank ordering (top-5 for verification)
        sorted_indices = torch.argsort(values, descending=True)
        top_5_ranks = sorted_indices[:5].tolist()

        # Step 5: Bucket histogram (statistical fingerprint)
        # Shift buckets to positive range for bincount
        shifted_buckets = (buckets + PROOF_NUM_BUCKETS).to(torch.long)
        histogram = torch.bincount(shifted_buckets, minlength=2 * PROOF_NUM_BUCKETS + 1)

        return {
            "sketch": sketch_val,
            "indices": indices.tolist(),
            "top_5_ranks": top_5_ranks,
            "histogram": histogram.tolist(),
            "position": position,
        }
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
    def compute_commitments_and_logprobs(self, prompt_length: int, all_token_ids: list[int], r_vec: torch.Tensor) -> Tuple[List[int], List[float]]:
        """Compute s values for a list of token ids using this GPU's model.

        all_token_ids: list of token ids (integers). We assume these are already
        token ids (not raw text).
        """
        with self.lock:                    # exclusive access to this GPU
            commitments = []
            # Build an input_ids tensor directly since caller provides token ids
            token_tensor = torch.tensor([all_token_ids], dtype=torch.long, device=self.device)
            model_outputs = self.model(input_ids=token_tensor, output_hidden_states=True)
            h_layer = model_outputs.hidden_states[LAYER_INDEX][0]
            for pos in range(len(all_token_ids)):
                if pos < h_layer.size(0):
                    commitment = create_commitment(h_layer[pos], r_vec, pos)
                    commitments.append(commitment)
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
            return commitments, all_logprobs
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
                future.set_result(worker.compute_commitments_and_logprobs(prompt_length, all_token_ids, r_vec))
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