"""
gemma3_utils.py - Minimal from-scratch decoder-only Gemma-3-style Transformer
using CuPy only. No autograd: forward AND backward passes are hand-derived.

Differences from a GPT-2 style model implemented here to match Gemma 3:
  * RMSNorm instead of LayerNorm (pre-norm AND post-norm around each sub-block,
    i.e. Gemma3 does: x + post_attn_norm(attn(pre_attn_norm(x))), same for MLP)
  * Rotary position embeddings (RoPE) instead of learned position embeddings
  * Grouped-Query Attention (GQA): num_kv_heads <= num_attention_heads
  * QK-Norm: RMSNorm applied per-head to q and k before the dot product
  * Alternating local (sliding-window) / global attention layers, pattern of
    5 local : 1 global, as in Gemma 3
  * GeGLU MLP (gate_proj * up_proj -> down_proj) with GELU on the gate branch
  * Attention logit soft-capping is OFF in Gemma 3 (kept as a no-op knob)
  * Embedding scaling by sqrt(hidden_size) applied to token embeddings
  * Tied input/output embeddings

Tokenizer: Gemma normally ships a SentencePiece unigram tokenizer. Reproducing
unigram LM training from scratch is out of scope for a minimal trainer, so we
keep a byte-level BPE tokenizer (same algorithm as the GPT-2 file) but expose
it as `GemmaTokenizer` with Gemma's special-token conventions
(<bos>, <eos>, <pad>, <unk>, <start_of_turn>, <end_of_turn>) so the rest of
the pipeline (dataset packing, generation, GGUF export) lines up with how
llama.cpp expects a "gemma"-family GGUF to be laid out. If you have a real
Gemma sentencepiece.model, load it with `GemmaTokenizer.from_sentencepiece`
instead (requires `pip install sentencepiece`) to get byte-identical tokenization.
"""

import os
import json
import random
import time
import math
import subprocess
import glob
import cupy as xp
import numpy as _np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# --------------------------------------------------------------------------------------
# small helpers (unchanged from the GPT-2 utils)
# --------------------------------------------------------------------------------------
def get_gpu_stats():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        output = subprocess.check_output(cmd, universal_newlines=True, timeout=2)
        parts = [x.strip() for x in output.strip().split('\n')[0].split(',')]
        if len(parts) < 6:
            return None

        def pf(s):
            try:
                return None if s in ('N/A', '[N/A]', 'None', '') else float(s)
            except ValueError:
                return None

        power_draw = pf(parts[3])
        if power_draw is not None and power_draw >= 500.0:
            power_draw = None
        return {
            "gpu_util": pf(parts[0]), "mem_used": pf(parts[1]), "mem_total": pf(parts[2]),
            "power_draw": power_draw, "power_limit": pf(parts[4]), "temp": pf(parts[5]),
        }
    except Exception as e:
        print(e)
        return None


def format_time(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# --------------------------------------------------------------------------------------
# Config (Gemma-3 style knobs)
# --------------------------------------------------------------------------------------
class Config:
    def __init__(self, **kw):
        self.vocab_size = kw.get("vocab_size", 8000)
        self.unk_token = kw.get("unk_token", "<unk>")
        self.bos_token = kw.get("bos_token", "<bos>")
        self.eos_token = kw.get("eos_token", "<eos>")
        self.pad_token = kw.get("pad_token", "<pad>")
        self.start_of_turn_token = kw.get("start_of_turn_token", "<start_of_turn>")
        self.end_of_turn_token = kw.get("end_of_turn_token", "<end_of_turn>")

        # architecture
        self.hidden_size = kw.get("hidden_size", 256)
        self.intermediate_size = kw.get("intermediate_size", 1024)
        self.num_hidden_layers = kw.get("num_hidden_layers", 6)
        self.num_attention_heads = kw.get("num_attention_heads", 8)
        self.num_key_value_heads = kw.get("num_key_value_heads", 2)   # GQA
        self.head_dim = kw.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.max_position_embeddings = kw.get("max_position_embeddings", 8192)

        # Gemma-3 specific
        self.rms_norm_eps = kw.get("rms_norm_eps", 1e-6)
        self.rope_theta_global = kw.get("rope_theta_global", 1_000_000.0)
        self.rope_theta_local = kw.get("rope_theta_local", 10_000.0)
        self.sliding_window = kw.get("sliding_window", 512)
        # every Nth layer (1-indexed) is global attention; others are local/sliding
        self.global_attn_every_n = kw.get("global_attn_every_n", 6)
        self.query_pre_attn_scalar = kw.get("query_pre_attn_scalar", None)  # default: head_dim**-0.5
        self.final_logit_softcap = kw.get("final_logit_softcap", 0.0)  # 0.0 = disabled (Gemma3 default)

        # data / training shapes
        self.max_sequence_length = kw.get("max_sequence_length", 512)
        self.ignore_index = kw.get("ignore_index", -100)
        self.batch_size = kw.get("batch_size", 4)
        self.accumulation_steps = kw.get("accumulation_steps", 1)
        self.val_split = kw.get("val_split", 0.0)
        self.attention_dropout = kw.get("attention_dropout", 0.0)
        self.weight_decay = kw.get("weight_decay", 0.01)

        self.learning_rate = kw.get("learning_rate", 1.2e-3)
        self.steps_per_run = kw.get("steps_per_run", 10000)
        self.log_interval = kw.get("log_interval", 10)
        self.save_interval = kw.get("save_interval", 10)
        self.total_training_steps = kw.get("total_training_steps", 10000)
        self.warmup_steps = kw.get("warmup_steps", 1000)

        self.seed = kw.get("seed", 1234)
        self.checkpoint_dir = kw.get("checkpoint_dir", "checkpoints")
        self.keep_checkpoint_max = kw.get("keep_checkpoint_max", None)

        # tokenizer-vocab-build sampling knobs
        self.vocab_sample_size = kw.get("vocab_sample_size", 100000)
        self.vocab_source_path = kw.get("vocab_source_path", None)

        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            "num_attention_heads must be divisible by num_key_value_heads for GQA"

    def to_dict(self):
        return dict(self.__dict__)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp_path, path)  # atomic on POSIX and Windows

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------
class GemmaTokenizer:
    """
    Byte-level BPE utilizing the HuggingFace tokenizers library.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.special_tokens = [
            self.cfg.pad_token,           # 0
            self.cfg.eos_token,           # 1
            self.cfg.bos_token,           # 2
            self.cfg.unk_token,           # 3
            self.cfg.start_of_turn_token, # 4
            self.cfg.end_of_turn_token    # 5
        ]

        self.tokenizer = Tokenizer(models.BPE(unk_token=self.cfg.unk_token))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

        self.tokenizer.add_special_tokens(self.special_tokens)
        self._update_special_ids()

    def _update_special_ids(self):
        self.pad_id = self.tokenizer.token_to_id(self.cfg.pad_token)
        self.eos_id = self.tokenizer.token_to_id(self.cfg.eos_token)
        self.bos_id = self.tokenizer.token_to_id(self.cfg.bos_token)
        self.unk_id = self.tokenizer.token_to_id(self.cfg.unk_token)
        self.start_of_turn_id = self.tokenizer.token_to_id(self.cfg.start_of_turn_token)
        self.end_of_turn_id = self.tokenizer.token_to_id(self.cfg.end_of_turn_token)

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def build_vocab(self, texts, max_vocab_samples=None):
        if max_vocab_samples is not None and len(texts) > max_vocab_samples:
            texts = random.sample(texts, max_vocab_samples)

        trainer = trainers.BpeTrainer(
            vocab_size=self.cfg.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        self.tokenizer.train_from_iterator(texts, trainer)
        self._update_special_ids()

    def build_vocab_from_jsonl(self, path, sample_size=100000, seed=1234):
        """
        Build the BPE vocab from a *representative* sample of an arbitrarily
        large JSONL file, without ever loading the whole file into memory.

        Uses reservoir sampling while streaming line-by-line, so the vocab
        reflects the statistics of the *entire* file (e.g. all 4M rows) even
        though only `sample_size` rows are ever held in RAM at once. This is
        what you want to call BEFORE starting an incremental training loop
        that grows train.jsonl a little at a time (e.g. 1k -> 2k -> ... -> 4M
        rows): build the tokenizer once against the full corpus (or the
        biggest file you already have), save it, and then reuse it
        (build_vocab=False) for every subsequent incremental training run so
        the vocab never has to change and old checkpoints stay loadable.
        """
        rng = random.Random(seed)
        reservoir = []
        seen = 0
        print(f"[{time.strftime('%X')}] streaming {path} to reservoir-sample "
              f"up to {sample_size} rows for vocab building...", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen += 1
                item = (obj["input"], obj["output"])
                if len(reservoir) < sample_size:
                    reservoir.append(item)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < sample_size:
                        reservoir[j] = item

        corpus = [x[0] for x in reservoir] + [x[1] for x in reservoir]
        print(f"[{time.strftime('%X')}] building vocab from {len(reservoir)} "
              f"reservoir-sampled rows (saw {seen} total rows in file)", flush=True)
        # corpus is already a random sample of the desired size, so don't
        # re-sample inside build_vocab.
        self.build_vocab(corpus, max_vocab_samples=None)

    def encode(self, text, add_bos=False, add_eos=False):
        seq = self.tokenizer.encode(text).ids
        if add_bos:
            seq = [self.bos_id] + seq
        if add_eos:
            seq = seq + [self.eos_id]
        return seq

    def encode_batch(self, texts):
        # Uses fully parallelized Rust threads to encode millions of strings instantly
        return [enc.ids for enc in self.tokenizer.encode_batch(texts)]

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
        self._update_special_ids()


# --------------------------------------------------------------------------------------
# Dataset (Gemma chat-style packing: <bos> <start_of_turn>user ... <end_of_turn> ...)
# --------------------------------------------------------------------------------------
class MemmapDataset:
    """Pre-tokenized dataset backed by on-disk memmaps. Replaces on-the-fly
    tokenization in JsonlDataset.__getitem__ for large datasets."""
    def __init__(self, in_path, lb_path, n_rows, seq_len):
        self.n = n_rows
        self.seq_len = seq_len
        self.inp = _np.memmap(in_path, dtype=_np.int32, mode="r", shape=(n_rows, seq_len))
        self.lbl = _np.memmap(lb_path, dtype=_np.int32, mode="r", shape=(n_rows, seq_len))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # copy out of the memmap so downstream xp.asarray/np.stack doesn't
        # hold a view into the mmap across batch boundaries
        return _np.array(self.inp[idx]), _np.array(self.lbl[idx])

    def split(self, val_frac, seed=1234):
        idx = _np.arange(self.n)
        _np.random.RandomState(seed).shuffle(idx)
        n_val = int(self.n * val_frac)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        class LazySubset:
            def __init__(self, dataset, indices):
                self.dataset, self.indices = dataset, indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                return self.dataset[self.indices[i]]

        train_subset = LazySubset(self, train_idx)
        val_subset = LazySubset(self, val_idx) if n_val > 0 else []
        return train_subset, val_subset


def pretokenize_to_memmap(jsonl_dataset, out_prefix, cfg, chunk_size=10000):
    """Batch-tokenize a JsonlDataset once into int32 memmaps on disk.
    Skips work if the memmaps already exist and match the row count."""
    N = len(jsonl_dataset.inputs)
    seq_len = cfg.max_sequence_length - 1
    in_path = out_prefix + "_in.dat"
    lb_path = out_prefix + "_lb.dat"
    meta_path = out_prefix + "_meta.json"

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("n_rows") == N and meta.get("seq_len") == seq_len:
            print(f"[{time.strftime('%X')}] memmap cache found ({N} rows), skipping pretokenize", flush=True)
            return in_path, lb_path, N, seq_len

    print(f"[{time.strftime('%X')}] pretokenizing {N} rows -> memmap...", flush=True)
    inp_mm = _np.memmap(in_path, dtype=_np.int32, mode="w+", shape=(N, seq_len))
    lbl_mm = _np.memmap(lb_path, dtype=_np.int32, mode="w+", shape=(N, seq_len))

    t0 = time.time()
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        inp_ids_batch = jsonl_dataset.tok.encode_batch(jsonl_dataset.inputs[start:end])
        out_ids_batch = jsonl_dataset.tok.encode_batch(jsonl_dataset.outputs[start:end])
        for i, (a, b) in enumerate(zip(inp_ids_batch, out_ids_batch)):
            seq_in, seq_lb = jsonl_dataset._pack_sequence(a, b)
            inp_mm[start + i] = seq_in
            lbl_mm[start + i] = seq_lb
        if start % (chunk_size * 10) == 0:
            elapsed = time.time() - t0
            rate = (start + chunk_size) / max(elapsed, 1e-9)
            eta = (N - start) / max(rate, 1e-9)
            print(f"  {start}/{N} rows ({rate:.0f} rows/s, eta {format_time(eta)})", flush=True)

    inp_mm.flush()
    lbl_mm.flush()
    with open(meta_path, "w") as f:
        json.dump({"n_rows": N, "seq_len": seq_len}, f)
    print(f"[{time.strftime('%X')}] pretokenize done in {format_time(time.time() - t0)}", flush=True)
    return in_path, lb_path, N, seq_len

class JsonlDataset:
    def __init__(self, path, tokenizer: GemmaTokenizer, cfg: Config, build_vocab=True):
        self.cfg = cfg
        self.tok = tokenizer

        # 1. Read only raw text to save massive amounts of RAM
        self.inputs = []
        self.outputs = []

        print(f"[{time.strftime('%X')}] Reading JSONL file...", flush=True)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        print("This line error:", line)
                        raise e
                    self.inputs.append(obj["input"])
                    self.outputs.append(obj["output"])

        if build_vocab and self.inputs:
            print(f"[{time.strftime('%X')}] Building vocabulary...", flush=True)
            n = len(self.inputs)
            sample_n = min(n, 25000)
            # Randomly sample across the WHOLE file instead of just taking the
            # first 25k rows, so the vocab is representative of the full
            # dataset even if it's small or ordered non-randomly (e.g. sorted
            # by topic/length/date). For files with millions of rows that
            # can't be held in memory at all, use
            # tokenizer.build_vocab_from_jsonl(path, ...) instead, which
            # streams the file and never loads it fully into RAM.
            idx = random.sample(range(n), sample_n) if n > sample_n else list(range(n))
            corpus = [self.inputs[i] for i in idx] + [self.outputs[i] for i in idx]
            self.tok.build_vocab(corpus, max_vocab_samples=50000)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # 2. Tokenize and pack ONLY when an item is requested during training
        inp_text = self.inputs[idx]
        out_text = self.outputs[idx]

        inp_ids = self.tok.encode(inp_text)
        out_ids = self.tok.encode(out_text)

        return self._pack_sequence(inp_ids, out_ids)

    def _pack_sequence(self, inp_ids, out_ids):
        cfg, tok = self.cfg, self.tok
        max_len = cfg.max_sequence_length

        prompt_ids = [tok.bos_id, tok.start_of_turn_id] + inp_ids + [tok.end_of_turn_id, tok.start_of_turn_id]
        answer_ids = out_ids + [tok.end_of_turn_id, tok.eos_id]

        seq = (prompt_ids + answer_ids)[:max_len]
        actual_len = len(seq)
        pad_len = max_len - actual_len

        seq.extend([tok.pad_id] * pad_len)

        input_ids = _np.array(seq[:-1], dtype=_np.int32)
        labels = _np.array(seq[1:], dtype=_np.int32)

        prompt_len = len(prompt_ids)
        mask_prompt_end = min(prompt_len - 1, max_len - 1)
        if mask_prompt_end > 0:
            labels[:mask_prompt_end] = cfg.ignore_index

        if pad_len > 0:
            mask_pad_start = actual_len - 1
            if mask_pad_start < len(labels):
                labels[mask_pad_start:] = cfg.ignore_index

        return input_ids, labels

    def split(self, val_frac, seed=1234):
        # 3. Memory-safe split!
        # Instead of copying data, we just shuffle the indices.
        n = len(self.inputs)
        idx = _np.arange(n)
        _np.random.RandomState(seed).shuffle(idx)

        n_val = int(n * val_frac)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        # A lightweight wrapper that acts like a list but processes data on the fly
        class LazySubset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, i):
                # When asked for item 'i', look up its real index and process it
                real_idx = self.indices[i]
                return self.dataset[real_idx]

        train_subset = LazySubset(self, train_idx)
        val_subset = LazySubset(self, val_idx) if n_val > 0 else []

        return train_subset, val_subset

def iterate_batches(examples, batch_size, seed=0, shuffle=True):
    n = len(examples)
    order = _np.arange(n)
    if shuffle:
        _np.random.RandomState(seed).shuffle(order)
    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        if len(idxs) == 0:
            continue
        batch_in = _np.stack([examples[i][0] for i in idxs])
        batch_lb = _np.stack([examples[i][1] for i in idxs])
        yield xp.asarray(batch_in), xp.asarray(batch_lb)


# --------------------------------------------------------------------------------------
# Low-level ops with manual forward/backward
# --------------------------------------------------------------------------------------
def init_linear_nobias(rng, in_dim, out_dim, scale=0.02):
    return xp.asarray(rng.normal(0, scale, size=(in_dim, out_dim)), dtype=xp.float32)


def linear_forward(x, W):
    y = x @ W
    return y, (x, W)


def linear_backward(dy, cache):
    x, W = cache
    flat_x = x.reshape(-1, x.shape[-1])
    flat_dy = dy.reshape(-1, dy.shape[-1])
    dW = flat_x.T @ flat_dy
    dx = dy @ W.T
    return dx, dW


def rmsnorm_forward(x, weight, eps):
    """Gemma-style RMSNorm: normalize, then scale by (1 + weight)."""
    ms = xp.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / xp.sqrt(ms + eps)
    xhat = x * inv_rms
    y = xhat * (1.0 + weight)
    return y, (x, xhat, inv_rms, weight, eps)


def rmsnorm_backward(dy, cache):
    x, xhat, inv_rms, weight, eps = cache
    N = x.shape[-1]
    dw = (dy * xhat).reshape(-1, N).sum(axis=0)

    dxhat = dy * (1.0 + weight)
    # d/dx of x * inv_rms(x), inv_rms = (mean(x^2)+eps)^-1/2
    dot = xp.sum(dxhat * x, axis=-1, keepdims=True)
    dx = inv_rms * dxhat - (inv_rms ** 3 / N) * x * dot
    return dx, dw


def gelu_forward(x):
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x ** 3)
    t = xp.tanh(inner)
    y = 0.5 * x * (1.0 + t)
    return y, (x, t, c)


def gelu_backward(dy, cache):
    x, t, c = cache
    sech2 = 1.0 - t * t
    dinner_dx = c * (1.0 + 3 * 0.044715 * x ** 2)
    dy_dx = 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner_dx
    return dy * dy_dx


def silu_like_geglu_forward(gate, up):
    """GeGLU: gelu(gate) * up"""
    act, act_cache = gelu_forward(gate)
    y = act * up
    return y, (act, up, act_cache)


def silu_like_geglu_backward(dy, cache):
    act, up, act_cache = cache
    dact = dy * up
    dup = dy * act
    dgate = gelu_backward(dact, act_cache)
    return dgate, dup


def softmax_forward(x, axis=-1):
    x_max = xp.max(x, axis=axis, keepdims=True)
    e = xp.exp(x - x_max)
    return e / xp.sum(e, axis=axis, keepdims=True)


def dropout_forward(x, p, rng, train):
    if not train or p <= 0.0:
        return x, None
    mask = (xp.asarray(rng.random_sample(x.shape)) >= p).astype(x.dtype) / (1.0 - p)
    return x * mask, mask


def dropout_backward(dy, mask):
    return dy if mask is None else dy * mask


def build_mask(seq_len, pad_mask=None, sliding_window=None):
    """Additive mask (1,1,T,T): 0 allowed, -1e9 disallowed. Causal, optionally
    combined with a sliding-window (local attention) restriction and padding."""
    i = xp.arange(seq_len)[:, None]
    j = xp.arange(seq_len)[None, :]
    causal = (j > i)
    m = xp.where(causal, xp.float32(-1e9), xp.float32(0.0))
    if sliding_window is not None:
        too_far = (i - j) >= sliding_window
        m = xp.where(too_far, xp.float32(-1e9), m)
    m = m[None, None, :, :]
    if pad_mask is not None:
        extra = (1.0 - pad_mask[:, None, None, :].astype(xp.float32)) * -1e9
        m = m + extra
    return m


# ---------------- RoPE ----------------
def rope_freqs(head_dim, theta, seq_len):
    inv_freq = 1.0 / (theta ** (xp.arange(0, head_dim, 2, dtype=xp.float32) / head_dim))
    t = xp.arange(seq_len, dtype=xp.float32)
    freqs = xp.outer(t, inv_freq)          # (T, head_dim/2)
    emb = xp.concatenate([freqs, freqs], axis=-1)  # (T, head_dim)
    return xp.cos(emb), xp.sin(emb)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return xp.concatenate([-x2, x1], axis=-1)


def apply_rope(x, cos, sin):
    # x: (B, nh, T, hd); cos/sin: (T, hd)
    cos_ = cos[None, None, :, :]
    sin_ = sin[None, None, :, :]
    return x * cos_ + rotate_half(x) * sin_


def apply_rope_backward(dy, cos, sin):
    """RoPE is an orthogonal (rotation) transform per pair of dims, so the
    backward pass is the inverse rotation, i.e. apply RoPE with -sin."""
    return apply_rope(dy, cos, -sin)


# --------------------------------------------------------------------------------------
# Grouped-Query Attention with QK-Norm + RoPE (manual backward)
# --------------------------------------------------------------------------------------
def attention_forward(x, p, cfg, cos, sin, attn_mask, rng, train):
    B, T, H = x.shape
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = nh // nkv
    scale = cfg.query_pre_attn_scalar if cfg.query_pre_attn_scalar else hd ** -0.5

    q, cache_q = linear_forward(x, p["wq"])   # (B,T,nh*hd)
    k, cache_k = linear_forward(x, p["wk"])   # (B,T,nkv*hd)
    v, cache_v = linear_forward(x, p["wv"])   # (B,T,nkv*hd)

    qh = q.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
    kh = k.reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)
    vh = v.reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)

    # QK-Norm: per-head RMSNorm over head_dim, no learned bias (Gemma3 style)
    qh_n, qkn_cache = rmsnorm_forward(qh, p["q_norm_w"], cfg.rms_norm_eps)
    kh_n, kkn_cache = rmsnorm_forward(kh, p["k_norm_w"], cfg.rms_norm_eps)

    qh_r = apply_rope(qh_n, cos, sin)
    kh_r = apply_rope(kh_n, cos, sin)

    # repeat kv heads to match query heads (GQA)
    kh_rep = xp.repeat(kh_r, n_rep, axis=1)
    vh_rep = xp.repeat(vh, n_rep, axis=1)

    scores = (qh_r @ kh_rep.transpose(0, 1, 3, 2)) * scale
    scores = scores + attn_mask
    attn = softmax_forward(scores, axis=-1)
    attn_drop, drop_mask = dropout_forward(attn, cfg.attention_dropout, rng, train)

    out_h = attn_drop @ vh_rep                       # (B, nh, T, hd)
    out = out_h.transpose(0, 2, 1, 3).reshape(B, T, nh * hd)

    y, cache_o = linear_forward(out, p["wo"])

    cache = dict(
        cache_q=cache_q, cache_k=cache_k, cache_v=cache_v, cache_o=cache_o,
        qh=qh, kh=kh, vh=vh, qh_n=qh_n, kh_n=kh_n, qkn_cache=qkn_cache, kkn_cache=kkn_cache,
        qh_r=qh_r, kh_r=kh_r, kh_rep=kh_rep, vh_rep=vh_rep,
        attn=attn, attn_drop=attn_drop, drop_mask=drop_mask,
        cos=cos, sin=sin, scale=scale, n_rep=n_rep,
        B=B, T=T, H=H, nh=nh, nkv=nkv, hd=hd,
    )
    return y, cache


def attention_backward(dy, cache):
    B, T, H = cache["B"], cache["T"], cache["H"]
    nh, nkv, hd, n_rep = cache["nh"], cache["nkv"], cache["hd"], cache["n_rep"]
    cos, sin, scale = cache["cos"], cache["sin"], cache["scale"]

    dout, dwo = linear_backward(dy, cache["cache_o"])
    dout_h = dout.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)

    attn, attn_drop, drop_mask = cache["attn"], cache["attn_drop"], cache["drop_mask"]
    qh_r, kh_rep, vh_rep = cache["qh_r"], cache["kh_rep"], cache["vh_rep"]

    dattn_drop = dout_h @ vh_rep.transpose(0, 1, 3, 2)
    dvh_rep = attn_drop.transpose(0, 1, 3, 2) @ dout_h

    dattn = dropout_backward(dattn_drop, drop_mask)
    dscores = attn * (dattn - xp.sum(dattn * attn, axis=-1, keepdims=True))
    dscores = dscores * scale

    dqh_r = dscores @ kh_rep
    dkh_rep = dscores.transpose(0, 1, 3, 2) @ qh_r

    # sum grouped-kv gradients back down to nkv heads
    def sum_groups(t):
        return t.reshape(B, nkv, n_rep, T, hd).sum(axis=2)

    dkh_r = sum_groups(dkh_rep)
    dvh = sum_groups(dvh_rep)

    dqh_n = apply_rope_backward(dqh_r, cos, sin)
    dkh_n = apply_rope_backward(dkh_r, cos, sin)

    dqh, dqnorm_w = rmsnorm_backward(dqh_n, cache["qkn_cache"])
    dkh, dknorm_w = rmsnorm_backward(dkh_n, cache["kkn_cache"])

    dq = dqh.transpose(0, 2, 1, 3).reshape(B, T, nh * hd)
    dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, nkv * hd)
    dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, nkv * hd)

    dx_q, dwq = linear_backward(dq, cache["cache_q"])
    dx_k, dwk = linear_backward(dk, cache["cache_k"])
    dx_v, dwv = linear_backward(dv, cache["cache_v"])

    dx = dx_q + dx_k + dx_v
    grads = dict(
        wq=dwq, wk=dwk, wv=dwv, wo=dwo,
        q_norm_w=dqnorm_w, k_norm_w=dknorm_w,
    )
    return dx, grads


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class Gemma3:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        rng = _np.random.RandomState(cfg.seed)
        H, I, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
        nh, nkv, hd, V = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.vocab_size

        self.params = {}
        self.params["embed_tokens"] = xp.asarray(rng.normal(0, 0.02, size=(V, H)), dtype=xp.float32)

        for i in range(L):
            pre = f"h{i}."
            self.params[pre + "input_norm.w"] = xp.zeros((H,), dtype=xp.float32)
            self.params[pre + "post_attn_norm.w"] = xp.zeros((H,), dtype=xp.float32)
            self.params[pre + "pre_ffn_norm.w"] = xp.zeros((H,), dtype=xp.float32)
            self.params[pre + "post_ffn_norm.w"] = xp.zeros((H,), dtype=xp.float32)

            self.params[pre + "attn.wq"] = init_linear_nobias(rng, H, nh * hd)
            self.params[pre + "attn.wk"] = init_linear_nobias(rng, H, nkv * hd)
            self.params[pre + "attn.wv"] = init_linear_nobias(rng, H, nkv * hd)
            self.params[pre + "attn.wo"] = init_linear_nobias(rng, nh * hd, H)
            self.params[pre + "attn.q_norm_w"] = xp.zeros((hd,), dtype=xp.float32)
            self.params[pre + "attn.k_norm_w"] = xp.zeros((hd,), dtype=xp.float32)

            self.params[pre + "mlp.gate"] = init_linear_nobias(rng, H, I)
            self.params[pre + "mlp.up"] = init_linear_nobias(rng, H, I)
            self.params[pre + "mlp.down"] = init_linear_nobias(rng, I, H)

        self.params["final_norm.w"] = xp.zeros((H,), dtype=xp.float32)

        # precompute per-layer RoPE tables lazily in forward() based on seq len
        self._rope_cache = {}
        self.rng = _np.random.RandomState(cfg.seed + 1)

    def is_global_layer(self, layer_idx):
        return (layer_idx + 1) % self.cfg.global_attn_every_n == 0

    def _get_rope(self, is_global, T):
        key = (is_global, T)
        if key not in self._rope_cache:
            theta = self.cfg.rope_theta_global if is_global else self.cfg.rope_theta_local
            self._rope_cache[key] = rope_freqs(self.cfg.head_dim, theta, T)
        return self._rope_cache[key]

    # ---------------- forward ----------------
    def forward(self, input_ids, pad_mask=None, train=True):
        cfg, p = self.cfg, self.params
        B, T = input_ids.shape
        H = cfg.hidden_size

        x = p["embed_tokens"][input_ids] * xp.float32(math.sqrt(H))  # embedding scaling

        global_mask = build_mask(T, pad_mask, sliding_window=None)
        local_mask = build_mask(T, pad_mask, sliding_window=cfg.sliding_window)

        caches = []
        for i in range(cfg.num_hidden_layers):
            pre = f"h{i}."
            is_global = self.is_global_layer(i)
            mask = global_mask if is_global else local_mask
            cos, sin = self._get_rope(is_global, T)

            in_norm_out, in_norm_cache = rmsnorm_forward(x, p[pre + "input_norm.w"], cfg.rms_norm_eps)

            attn_params = {
                "wq": p[pre + "attn.wq"], "wk": p[pre + "attn.wk"], "wv": p[pre + "attn.wv"],
                "wo": p[pre + "attn.wo"], "q_norm_w": p[pre + "attn.q_norm_w"], "k_norm_w": p[pre + "attn.k_norm_w"],
            }
            attn_out, attn_cache = attention_forward(in_norm_out, attn_params, cfg, cos, sin, mask, self.rng, train)

            post_attn_out, post_attn_cache = rmsnorm_forward(attn_out, p[pre + "post_attn_norm.w"], cfg.rms_norm_eps)
            x = x + post_attn_out

            pre_ffn_out, pre_ffn_cache = rmsnorm_forward(x, p[pre + "pre_ffn_norm.w"], cfg.rms_norm_eps)

            gate, gate_cache = linear_forward(pre_ffn_out, p[pre + "mlp.gate"])
            up, up_cache = linear_forward(pre_ffn_out, p[pre + "mlp.up"])
            act, geglu_cache = silu_like_geglu_forward(gate, up)
            down, down_cache = linear_forward(act, p[pre + "mlp.down"])

            post_ffn_out, post_ffn_cache = rmsnorm_forward(down, p[pre + "post_ffn_norm.w"], cfg.rms_norm_eps)
            x = x + post_ffn_out

            caches.append(dict(
                in_norm_cache=in_norm_cache, attn_cache=attn_cache, post_attn_cache=post_attn_cache,
                pre_ffn_cache=pre_ffn_cache, gate_cache=gate_cache, up_cache=up_cache,
                geglu_cache=geglu_cache, down_cache=down_cache, post_ffn_cache=post_ffn_cache,
            ))

        final_norm_out, final_norm_cache = rmsnorm_forward(x, p["final_norm.w"], cfg.rms_norm_eps)
        logits = final_norm_out @ p["embed_tokens"].T  # tied embeddings

        if cfg.final_logit_softcap and cfg.final_logit_softcap > 0:
            cap = cfg.final_logit_softcap
            logits = cap * xp.tanh(logits / cap)

        fwd_cache = dict(
            tok_ids=input_ids, final_norm_out=final_norm_out, final_norm_cache=final_norm_cache,
            layer_caches=caches, T=T, softcap_pre_tanh_logits=(logits if False else None),
        )
        return logits, fwd_cache

    # ---------------- loss ----------------
    def loss(self, logits, labels, ignore_index):
        B, T, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        flat_labels = labels.reshape(-1)
        mask = (flat_labels != ignore_index)
        safe_labels = xp.where(mask, flat_labels, 0)

        probs = softmax_forward(flat_logits, axis=-1)
        n = int(mask.sum().item())
        if n == 0:
            return xp.float32(0.0), probs, mask, safe_labels, n

        picked = xp.clip(probs[xp.arange(flat_labels.shape[0]), safe_labels], 1e-9, 1.0)
        loss = (-xp.log(picked) * mask).sum() / n
        return loss, probs, mask, safe_labels, n

    def loss_backward(self, probs, mask, safe_labels, n, logits_shape):
        B, T, V = logits_shape
        if n == 0:
            return xp.zeros(logits_shape, dtype=xp.float32)
        dlogits = probs.copy()
        dlogits[xp.arange(safe_labels.shape[0]), safe_labels] -= 1.0
        dlogits = dlogits * mask[:, None] / n
        return dlogits.reshape(B, T, V)

    # ---------------- backward ----------------
    def backward(self, dlogits, fwd_cache):
        cfg, p = self.cfg, self.params
        grads = {k: xp.zeros_like(v) for k, v in p.items()}

        final_norm_out = fwd_cache["final_norm_out"]
        flat_fn = final_norm_out.reshape(-1, final_norm_out.shape[-1])
        flat_dl = dlogits.reshape(-1, dlogits.shape[-1])
        grads["embed_tokens"] += flat_dl.T @ flat_fn
        dfinal_norm_out = dlogits @ p["embed_tokens"]

        dx, dfn_w = rmsnorm_backward(dfinal_norm_out, fwd_cache["final_norm_cache"])
        grads["final_norm.w"] += dfn_w

        for i in reversed(range(cfg.num_hidden_layers)):
            pre = f"h{i}."
            c = fwd_cache["layer_caches"][i]

            dpost_ffn_out = dx
            ddown, dpfn_w = rmsnorm_backward(dpost_ffn_out, c["post_ffn_cache"])
            grads[pre + "post_ffn_norm.w"] += dpfn_w

            dact, dw_down = linear_backward(ddown, c["down_cache"])
            grads[pre + "mlp.down"] += dw_down

            dgate, dup = silu_like_geglu_backward(dact, c["geglu_cache"])
            dpre_ffn_out1, dw_gate = linear_backward(dgate, c["gate_cache"])
            dpre_ffn_out2, dw_up = linear_backward(dup, c["up_cache"])
            grads[pre + "mlp.gate"] += dw_gate
            grads[pre + "mlp.up"] += dw_up
            dpre_ffn_out = dpre_ffn_out1 + dpre_ffn_out2

            dx_from_preffn, dprn_w = rmsnorm_backward(dpre_ffn_out, c["pre_ffn_cache"])
            grads[pre + "pre_ffn_norm.w"] += dprn_w
            dx = dx + dx_from_preffn

            dpost_attn_out = dx
            dattn_out, dpan_w = rmsnorm_backward(dpost_attn_out, c["post_attn_cache"])
            grads[pre + "post_attn_norm.w"] += dpan_w

            din_norm_out, attn_grads = attention_backward(dattn_out, c["attn_cache"])
            for gk, gv in attn_grads.items():
                grads[pre + "attn." + gk] += gv

            dx_from_innorm, din_w = rmsnorm_backward(din_norm_out, c["in_norm_cache"])
            grads[pre + "input_norm.w"] += din_w
            dx = dx + dx_from_innorm

        tok_ids = fwd_cache["tok_ids"]
        H = cfg.hidden_size
        flat_ids = tok_ids.reshape(-1)
        flat_dx = (dx * xp.float32(math.sqrt(H))).reshape(-1, H)
        xp.add.at(grads["embed_tokens"], flat_ids, flat_dx)

        return grads

    def num_params(self):
        return int(sum(v.size for v in self.params.values()))

    def save_checkpoint(self, path, step, optimizer_state=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arrays = {k: xp.asnumpy(v) for k, v in self.params.items()}
        if optimizer_state is not None:
            arrays['optim_step'] = _np.int64(optimizer_state['t'])
            for k in self.params.keys():
                arrays['optim_m_' + k] = xp.asnumpy(optimizer_state['m'][k])
                arrays['optim_v_' + k] = xp.asnumpy(optimizer_state['v'][k])
        _np.savez(path, step=step, **arrays)

    def load_checkpoint(self, path):
        data = _np.load(path)
        step = int(data['step'])
        for k in self.params.keys():
            self.params[k] = xp.asarray(data[k])
        optimizer_state = None
        if 'optim_step' in data:
            optimizer_state = {
                't': int(data['optim_step']),
                'm': {k: xp.asarray(data['optim_m_' + k]) for k in self.params.keys()},
                'v': {k: xp.asarray(data['optim_v_' + k]) for k in self.params.keys()},
            }
        return step, optimizer_state

# --------------------------------------------------------------------------------------
# Adam optimizer (manual) - identical to GPT-2 utils
# --------------------------------------------------------------------------------------
class AdamOptimizer:
    def __init__(self, params, lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.wd = lr, weight_decay
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = {k: xp.zeros_like(v) for k, v in params.items()}
        self.v = {k: xp.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for k in params.keys():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            update = lr * mhat / (xp.sqrt(vhat) + self.eps)
            if self.wd > 0 and params[k].ndim >= 2:
                params[k] -= lr * self.wd * params[k]
            params[k] -= update

    def state_dict(self):
        return {'m': {k: xp.asnumpy(v) for k, v in self.m.items()},
                'v': {k: xp.asnumpy(v) for k, v in self.v.items()}, 't': self.t}

    def load_state_dict(self, state):
        self.m = {k: xp.asarray(v) for k, v in state['m'].items()}
        self.v = {k: xp.asarray(v) for k, v in state['v'].items()}
        self.t = int(state['t'])


def lr_schedule(step, cfg: Config):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, (cfg.total_training_steps - cfg.warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    min_lr = cfg.learning_rate * 0.1
    return min_lr + 0.5 * (cfg.learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------------------
# Generation (greedy/sampled)
# --------------------------------------------------------------------------------------
def generate(model, tokenizer, prompt, max_new_tokens=20, temperature=0.8, do_sample=True):
    cfg = model.cfg
    ids = [tokenizer.bos_id, tokenizer.start_of_turn_id] + tokenizer.encode(prompt) + \
          [tokenizer.end_of_turn_id, tokenizer.start_of_turn_id]
    ids = ids[:cfg.max_sequence_length]
    start_idx = len(ids)

    for _ in range(max_new_tokens):
        cur = ids[-cfg.max_sequence_length:]
        arr = xp.asarray(_np.array([cur], dtype=_np.int32))
        logits, _ = model.forward(arr, pad_mask=None, train=False)

        if do_sample:
            logits_last = logits[0, -1] / temperature
            probs = softmax_forward(logits_last, axis=-1)
            probs = xp.asnumpy(probs.ravel())
            probs = probs / probs.sum()
            next_id = int(_np.random.choice(len(probs), p=probs))
        else:
            next_id = int(xp.argmax(logits[0, -1]).item())

        ids.append(next_id)
        if next_id in (tokenizer.eos_id, tokenizer.end_of_turn_id):
            break

    return tokenizer.decode(ids[start_idx:])


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg, model, tokenizer, train_examples, val_examples=None, start_step=0):
        self.cfg = cfg
        self.model = model
        self.tok = tokenizer
        self.train_examples = train_examples
        self.val_examples = val_examples or []
        self.opt = AdamOptimizer(model.params, cfg.learning_rate, cfg.weight_decay)
        self.global_step = start_step
        self.best_metric = None
        self.best_model_step = None
        self.log_history = []
        self.config_path = os.path.join(cfg.checkpoint_dir, "config.json")
        self._load_train_state()
        # Persist the config immediately (not just at the very end of the
        # script). This means if training is interrupted after a checkpoint
        # is saved, resuming later always loads config.json instead of
        # silently falling back to whatever hardcoded defaults happen to be
        # in the launcher script at that time (which could drift and desync
        # the LR schedule / architecture from the saved weights).
        self.cfg.save(self.config_path)

    def _pad_mask(self, input_ids):
        return (input_ids != self.tok.pad_id).astype(xp.float32)

    def _run_batch(self, input_ids, labels, train):
        pad_mask = self._pad_mask(input_ids)
        logits, fwd_cache = self.model.forward(input_ids, pad_mask=pad_mask, train=train)
        loss, probs, mask, safe_labels, n = self.model.loss(logits, labels, self.cfg.ignore_index)
        if train:
            dlogits = self.model.loss_backward(probs, mask, safe_labels, n, logits.shape)
            grads = self.model.backward(dlogits, fwd_cache)
            return loss, grads
        return loss, None

    def _compute_steps_per_epoch(self):
        if len(self.train_examples) == 0:
            return 1
        return max(1, int(math.ceil(len(self.train_examples) / self.cfg.batch_size)))

    def _load_train_state(self):
        state_path = os.path.join(self.cfg.checkpoint_dir, "train_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            self.best_metric = state.get("best_metric", None)
            self.best_model_step = state.get("best_model_step", None)
            self.log_history = state.get("log_history", [])

    def save_train_state(self, lr):
        state_path = os.path.join(self.cfg.checkpoint_dir, "train_state.json")
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        steps_per_epoch = self._compute_steps_per_epoch()
        state = {
            "epoch": self.global_step // steps_per_epoch,
            "global_step": self.global_step,
            "learning_rate": lr,
            "best_model_step": self.best_model_step,
            "best_metric": self.best_metric,
            "train_batch_size": self.cfg.batch_size,
            "num_train_epochs": int(math.ceil(self.cfg.total_training_steps / steps_per_epoch)),
            "model_total_params": self.model.num_params(),
            "config": self.cfg.to_dict(),
            "log_history": self.log_history,
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        # keep config.json in lockstep with train_state.json so a resume at
        # any point in time (not just script exit) sees the exact config
        # that produced the latest checkpoint.
        self.cfg.save(self.config_path)

    def save_checkpoint(self, path):
        self.model.save_checkpoint(path, self.global_step, self.opt.state_dict())
        self._prune_checkpoints()

    def _prune_checkpoints(self):
        """Keep only the N most recent step_*.npz checkpoints (N =
        cfg.keep_checkpoint_max). Older ones are deleted to bound disk usage
        during long/incremental training runs. Set keep_checkpoint_max to
        None or <= 0 to disable pruning and keep every checkpoint."""
        keep = self.cfg.keep_checkpoint_max
        if not keep or keep <= 0:
            return

        ckpts = glob.glob(os.path.join(self.cfg.checkpoint_dir, "step_*.npz"))

        def step_num(fp):
            try:
                return int(os.path.basename(fp).split('_')[-1].split('.')[0])
            except ValueError:
                return -1

        ckpts.sort(key=step_num)
        if len(ckpts) <= keep:
            return

        for old_path in ckpts[:-keep]:
            try:
                os.remove(old_path)
                print(f"pruned old checkpoint -> {old_path}", flush=True)
            except OSError as e:
                print(f"warning: could not remove old checkpoint {old_path}: {e}", flush=True)

    def evaluate(self, max_batches=50):
        if not self.val_examples:
            return None
        losses = []
        for i, (input_ids, labels) in enumerate(iterate_batches(self.val_examples, self.cfg.batch_size, shuffle=True)):
            if i >= max_batches: break
            loss, _ = self._run_batch(input_ids, labels, train=False)
            losses.append(float(loss.item()))
        return sum(losses) / len(losses) if losses else None

    def train(self, sample_prompt="hello world"):
        cfg = self.cfg
        steps_target = min(cfg.steps_per_run, cfg.total_training_steps - self.global_step)
        step_in_run = 0
        batch_iter = None
        last_log_time = time.time()
        last_log_global_step = self.global_step

        while step_in_run < steps_target:
            if batch_iter is None:
                batch_iter = iterate_batches(self.train_examples, cfg.batch_size, seed=self.global_step, shuffle=True)

            accum_grads = {k: xp.zeros_like(v) for k, v in self.model.params.items()}
            accum_loss, micro_steps = 0.0, 0

            for _ in range(cfg.accumulation_steps):
                try:
                    input_ids, labels = next(batch_iter)
                except StopIteration:
                    batch_iter = iterate_batches(self.train_examples, cfg.batch_size, seed=self.global_step + 1, shuffle=True)
                    input_ids, labels = next(batch_iter)

                loss, grads = self._run_batch(input_ids, labels, train=True)
                for k in accum_grads:
                    accum_grads[k] += grads[k]
                accum_loss += float(loss.item())
                micro_steps += 1

            for k in accum_grads:
                accum_grads[k] /= micro_steps

            lr = lr_schedule(self.global_step, cfg)
            self.opt.step(self.model.params, accum_grads, lr=lr)

            self.global_step += 1
            step_in_run += 1
            train_loss = accum_loss / micro_steps

            if self.global_step % cfg.log_interval == 0 or step_in_run == steps_target:
                val_loss = self.evaluate()
                sample = generate(self.model, self.tok, sample_prompt, max_new_tokens=100, do_sample=False)

                now = time.time()
                elapsed = now - last_log_time
                steps_since_log = max(1, self.global_step - last_log_global_step)
                avg_step_time = elapsed / steps_since_log
                eta_seconds = (cfg.total_training_steps - self.global_step) * avg_step_time
                last_log_time, last_log_global_step = now, self.global_step

                msg = (f"[step {self.global_step}/{cfg.total_training_steps}] lr={lr:.6f} "
                       f"train_loss={train_loss:.4f}")
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.4f}"
                msg += f" | sample('{sample_prompt}') -> '{sample}' | avg_step={avg_step_time:.3f}s eta={format_time(eta_seconds)}"
                print(msg, flush=True)

                metric = val_loss if val_loss is not None else train_loss
                if self.best_metric is None or metric < self.best_metric:
                    self.best_metric, self.best_model_step = float(metric), self.global_step

                log_entry = {
                    "step": self.global_step, "learning_rate": lr, "train_loss": train_loss,
                    "avg_step_second": avg_step_time, "eta_second": eta_seconds,
                }
                if val_loss is not None:
                    log_entry["val_loss"] = val_loss
                gpu_stats = get_gpu_stats()
                if gpu_stats is not None:
                    log_entry["gpu_util_percent"] = gpu_stats["gpu_util"]
                    log_entry["mem_used_mb"] = gpu_stats["mem_used"]
                    log_entry["power_draw_watt"] = gpu_stats["power_draw"]
                    log_entry["temp_c"] = gpu_stats["temp"]
                self.log_history.append(log_entry)
                self.save_train_state(lr)

            if self.global_step % cfg.save_interval == 0 or step_in_run == steps_target:
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"step_{self.global_step}.npz")
                self.save_checkpoint(ckpt_path)
                print(f"saved checkpoint -> {ckpt_path}", flush=True)
                self.save_train_state(lr)

        return self.global_step

    def load_checkpoint(self, path):
        step, opt_state = self.model.load_checkpoint(path)
        if opt_state is not None:
            self.opt.load_state_dict(opt_state)
        self.global_step = step
        return step