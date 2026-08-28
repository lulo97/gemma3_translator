What changed, concretely:

**1. Real vocab coverage for the full 4M-row corpus**
- New `GemmaTokenizer.build_vocab_from_jsonl(path, sample_size, seed)` streams a file line-by-line and reservoir-samples up to `sample_size` rows — never loads the whole file into RAM, but the sample is statistically representative of the *entire* file, not just the first N lines.
- `gemma3_app.py` now builds the tokenizer once (first run only) from `VOCAB_SOURCE_PATH` (defaults to `train.jsonl`, but you should point it at your biggest available file, e.g. `VOCAB_SOURCE_PATH=full_corpus.jsonl python gemma3_app.py`). After that, `tokenizer.json` exists and every later run reuses it (`build_vocab=False`) regardless of how many rows are currently in `train.jsonl`.
- `JsonlDataset`'s own fallback vocab-building path (used if you skip the streaming helper) now random-samples across the whole loaded file instead of just `[:25000]`.

**2. Config no longer silently drifts**
- `Trainer.__init__` writes `config.json` immediately, and `save_train_state` rewrites it every log/save interval — not just at script exit. So if you kill training mid-run (e.g. right when loss hits ~1), resuming always reloads the *actual* config that produced the checkpoint, never the hardcoded dict in the script.
- `Config.save` now writes atomically (tmp file + `os.replace`) so a crash mid-write can't corrupt it.

**3. Checkpoint pruning (N=3)**
- `cfg.keep_checkpoint_max=3` is set in the launcher.
- `Trainer._prune_checkpoints()` runs after every `save_checkpoint` call, sorts `step_*.npz` by step number, and deletes all but the newest 3.

**Your incremental workflow now looks like:**
```bash
VOCAB_SOURCE_PATH=full_4m_corpus.jsonl python gemma3_app.py   # builds tokenizer.json once
# train.jsonl has 1000 rows, run until loss ~1, Ctrl+C
# append 1000 more rows to train.jsonl
python gemma3_app.py   # resumes from latest checkpoint, reuses same tokenizer, retokenizes only because row count changed
# repeat, growing train.jsonl up to 4M rows
```

**`total_training_steps`:** it only feeds the cosine LR schedule (`lr_schedule`) and the ETA/ log math — it's not tied to your data size or checkpoint compatibility at all. For open-ended/"infinite" training you have two reasonable options:

1. **Set it very large** (e.g. `50_000_000`). Since `progress = (step - warmup) / (total - warmup)`, with a huge denominator the schedule barely moves off peak LR — you effectively get constant LR after warmup, forever. This is the simplest fix and matches "I don't know when I'll stop."
2. **Bump it upward periodically** (e.g. double it whenever you get close), which lets the cosine decay eventually kick in once you're actually approaching the end. More effort, no real benefit unless you know roughly when you'll stop for good.

For a true "keep training forever on growing data" setup, (1) is simpler and fine. Just don't leave it *too* close to `global_step` — if `total_training_steps <= global_step`, `steps_per_run = min(steps_per_run, total - global_step)` goes ≤0 and the training loop won't run at all next time you resume.

**Config values you can safely change between runs** (all get picked up automatically since `config.json` is now loaded and reused, not the hardcoded script defaults):

Freely changeable, no checkpoint compatibility issues:
- `batch_size`, `accumulation_steps`
- `steps_per_run`, `log_interval`, `save_interval`, `keep_checkpoint_max`
- `learning_rate` (peak LR — takes effect immediately, no shape dependency)
- `warmup_steps` (only matters if `global_step` hasn't passed it yet)
- `total_training_steps` (as above — just keep it ahead of `global_step`)
- `weight_decay`, `attention_dropout`
- `val_split` (just reshuffles the train/val split next run)
- `max_sequence_length` — this isn't baked into any weight tensor, only into the runtime shapes (RoPE tables, memmap cache). Changing it just invalidates the `tokenized_cache_*` files, which `pretokenize_to_memmap` detects and regenerates automatically.

Change with caution (won't crash, but silently changes model behavior mid-training):
- `sliding_window`, `global_attn_every_n`, `rope_theta_global`, `rope_theta_local` — these aren't stored in the checkpoint weights, they're just used at forward time. So the code will happily load an old checkpoint and apply a *different* attention pattern/RoPE base than what it was trained under. Nothing errors, but the model's learned attention behavior was shaped by the old values, so changing these mid-stream degrades quality rather than helping. Best to pick these once and leave them.

**Do NOT change, ever, for an existing `checkpoint_dir`:**
- `vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`

These define the actual parameter tensor shapes. `load_checkpoint` does a blind `self.params[k] = xp.asarray(data[k])` with no shape check — it won't error on load, but the forward pass will then break (matmul shape mismatches) or silently produce garbage. If you ever want to change these, start a fresh `checkpoint_dir` (and a fresh tokenizer only if `vocab_size` changed).
