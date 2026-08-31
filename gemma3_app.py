"""
gemma3_app.py - launcher for incremental training with a FIXED vocab.

Workflow this script is built for:
  1. You have (or will grow towards) a big corpus, e.g. 4,000,000 rows.
  2. You build the tokenizer ONCE against a large, representative sample of
     that full corpus (VOCAB_SOURCE_PATH), so it already covers essentially
     all tokens you'll ever see, even the ones in rows you haven't added to
     train.jsonl yet.
  3. You then train incrementally: start train.jsonl with e.g. 1,000 rows,
     train until loss drops to ~1, stop, append another 1,000 rows, resume
     training (same tokenizer, same model, same checkpoint dir), repeat...
     all the way up to the full 4,000,000 rows.

Because the tokenizer is built once and reused (`build_vocab=False`) for
every subsequent run, growing train.jsonl never changes vocab_size, so
existing checkpoints stay perfectly compatible across every stage of this
loop.

Only the newest N=3 checkpoint .npz files are kept on disk automatically
(see Config.keep_checkpoint_max / Trainer._prune_checkpoints in
gemma3_utils.py) so this doesn't fill your disk over a long incremental run.
"""

import os
import json
import glob
import time
from gemma3_utils import (
    Config, GemmaTokenizer, JsonlDataset, Gemma3, Trainer,
    pretokenize_to_memmap, MemmapDataset,
)

# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------
DATA_PATH = "train.jsonl"          # the (possibly small, growing) file you actually train on
CKPT_DIR = "checkpoints"

# The file to build the tokenizer's vocab from. Point this at your FULL
# corpus (e.g. the eventual 4M-row file, or any large file you already have
# lying around that's representative of it) so the vocab has real coverage
# before you ever start the small-to-large incremental training loop.
# If you don't have the full corpus yet, just point this at DATA_PATH and
# rebuild the tokenizer later against a bigger file before it starts to
# matter (note: changing the tokenizer after training has started will make
# old checkpoints' vocab_size mismatch, so only do that intentionally, e.g.
# by starting a fresh checkpoint_dir).
VOCAB_SOURCE_PATH = "full_4m_corpus.jsonl"
VOCAB_SAMPLE_SIZE = int(os.environ.get("VOCAB_SAMPLE_SIZE", "100000"))


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    config_path = os.path.join(CKPT_DIR, "config.json")
    if os.path.exists(config_path):
        cfg = Config.load(config_path)
        print(f"[{time.strftime('%X')}] loaded existing config from {config_path}", flush=True)
    else:
        cfg = Config(
            vocab_size=6000,
            hidden_size=256,
            intermediate_size=1536,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
            max_position_embeddings=512,

            rms_norm_eps=1e-6,
            rope_theta_global=1_000_000.0,
            rope_theta_local=10_000.0,
            sliding_window=128,
            global_attn_every_n=6,
            final_logit_softcap=0.0,

            max_sequence_length=256,
            batch_size=64,
            accumulation_steps=2,
            val_split=0.01,
            weight_decay=0.01,
            learning_rate=4e-5,
            steps_per_run=2000,
            log_interval=50,
            save_interval=200,
            total_training_steps=50_000_000, #Set to big number
            warmup_steps=500,
            checkpoint_dir=CKPT_DIR,

            # only keep the 3 newest step_*.npz checkpoints on disk
            keep_checkpoint_max=3,

            vocab_source_path=VOCAB_SOURCE_PATH,
            vocab_sample_size=VOCAB_SAMPLE_SIZE,
        )
        # persist immediately: even if this run gets interrupted before the
        # first checkpoint is saved, the *actual* config used is on disk,
        # not just the hardcoded defaults above (which you might edit later).
        cfg.save(config_path)
        print(f"[{time.strftime('%X')}] wrote new config -> {config_path}", flush=True)

    tokenizer = GemmaTokenizer(cfg)

    vocab_path = os.path.join(CKPT_DIR, "tokenizer.json")
    t0 = time.time()

    if os.path.exists(vocab_path):
        # Reuse the existing fixed vocab. This is the normal path for every
        # incremental run after the very first one: train.jsonl can have
        # grown from 1,000 to 2,000 to ... to 4,000,000 rows and the vocab
        # (and therefore vocab_size / all model shapes / old checkpoints)
        # never changes.
        tokenizer.load(vocab_path)
        print(f"[{time.strftime('%X')}] vocab loaded from {vocab_path} "
              f"({len(tokenizer)} tokens); reading dataset...", flush=True)
        dataset = JsonlDataset(DATA_PATH, tokenizer, cfg, build_vocab=False)
    else:
        # First-ever run for this checkpoint_dir: build the vocab once from
        # a representative, reservoir-sampled slice of VOCAB_SOURCE_PATH
        # (streamed, so this is safe even if that file has millions of
        # rows and doesn't fit in RAM). Do this BEFORE reading DATA_PATH so
        # the initial small train.jsonl doesn't accidentally define the
        # vocab on its own.
        print(f"[{time.strftime('%X')}] building vocab from {VOCAB_SOURCE_PATH} "
              f"(sample_size={VOCAB_SAMPLE_SIZE})...", flush=True)
        tokenizer.build_vocab_from_jsonl(
            VOCAB_SOURCE_PATH, sample_size=VOCAB_SAMPLE_SIZE, seed=cfg.seed
        )
        tokenizer.save(vocab_path)
        print(f"[{time.strftime('%X')}] vocab built and saved -> {vocab_path} "
              f"({len(tokenizer)} tokens)", flush=True)

        dataset = JsonlDataset(DATA_PATH, tokenizer, cfg, build_vocab=False)

    print(f"[{time.strftime('%X')}] dataset ready in {time.time() - t0:.1f}s "
          f"({len(dataset)} rows in {DATA_PATH})", flush=True)

    # vocab_size is fixed once the tokenizer exists; don't let a freshly-read
    # dataset silently resize it.
    actual_vocab_len = len(tokenizer)
    if cfg.vocab_size != actual_vocab_len:
        print(f"[{time.strftime('%X')}] note: cfg.vocab_size ({cfg.vocab_size}) != "
              f"tokenizer length ({actual_vocab_len}); using tokenizer length.", flush=True)
    cfg.vocab_size = actual_vocab_len

    # pretokenize (or reuse cached memmaps) for whatever train.jsonl currently
    # contains. If the row count changed since last run (you appended more
    # rows), this automatically detects the mismatch and retokenizes.
    memmap_prefix = os.path.join(CKPT_DIR, "tokenized_cache")
    in_path, lb_path, n_rows, seq_len = pretokenize_to_memmap(dataset, memmap_prefix, cfg)

    # free the raw text lists - no longer needed, and they're large in RAM at
    # millions of rows
    dataset.inputs = None
    dataset.outputs = None

    mm_dataset = MemmapDataset(in_path, lb_path, n_rows, seq_len)
    train_examples, val_examples = mm_dataset.split(cfg.val_split, seed=cfg.seed)

    print(f"train examples: {len(train_examples)} | val examples: {len(val_examples)}")
    print(f"vocab size (fixed): {len(tokenizer)}")

    #raise

    model = Gemma3(cfg)
    trainer = Trainer(cfg, model, tokenizer, train_examples, val_examples, start_step=0)
    trainer.train_file_size = os.path.getsize(DATA_PATH)

    print(f"model params: {model.num_params():,}")

    checkpoint_files = glob.glob(os.path.join(CKPT_DIR, "step_*.npz"))
    if checkpoint_files:
        latest = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        print(f"Loading checkpoint: {latest}")
        step = trainer.load_checkpoint(latest)
        print(f"Resumed from step {step}")
    else:
        print("No checkpoint found. Starting from scratch.")

    trainer.train(sample_prompt="i have a cat.")

    # config.json / tokenizer.json are already kept in sync during training
    # (see Trainer.__init__ and Trainer.save_train_state), but saving again
    # here is harmless and covers the "steps_per_run reached without hitting
    # a save_interval boundary" edge case.
    cfg.save(config_path)
    tokenizer.save(vocab_path)
    print("done.")


if __name__ == "__main__":
    main()