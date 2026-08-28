"""
export_gguf.py - standalone GGUF export function for the custom Gemma3 model.

Usage:
    from export_gguf import export_gguf
    export_gguf(model, tokenizer, "checkpoints/model.gguf")

`model` needs:
    - model.cfg            (the Config object: hidden_size, intermediate_size,
                             num_hidden_layers, num_attention_heads,
                             num_key_value_heads, head_dim, rms_norm_eps,
                             rope_theta_global, rope_theta_local,
                             sliding_window, global_attn_every_n,
                             final_logit_softcap, max_position_embeddings)
    - model.params          (dict[str, array] - cupy or numpy arrays both fine)

`tokenizer` needs:
    - len(tokenizer)
    - tokenizer.tokenizer.get_vocab() / .to_str()
    - tokenizer.pad_id / .eos_id / .bos_id / .unk_id
    - tokenizer.start_of_turn_id / .end_of_turn_id
"""

import json
import numpy as np


def _to_numpy(arr):
    """Works whether `arr` is a numpy array or a cupy array, without importing cupy."""
    if hasattr(arr, "get"):          # cupy arrays expose .get()
        return arr.get()
    return np.asarray(arr)


def export_gguf(model, tokenizer, path):
    """
    Export `model` to a GGUF file llama.cpp can load with the 'gemma3' architecture.
    Requires: pip install gguf

    Tensor / metadata names follow llama.cpp's gemma3 loader
    (see llama.cpp: llm_load_tensors / LLM_ARCH_GEMMA3 mappings).
    """
    import gguf

    cfg = model.cfg
    params = model.params

    H, I, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

    writer = gguf.GGUFWriter(path, "gemma3")

    writer.add_name("CustomGemma3")
    writer.add_context_length(cfg.max_position_embeddings)
    writer.add_embedding_length(H)
    writer.add_block_count(L)
    writer.add_feed_forward_length(I)
    writer.add_head_count(nh)
    writer.add_head_count_kv(nkv)
    writer.add_key_length(hd)
    writer.add_value_length(hd)
    writer.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_rope_freq_base(cfg.rope_theta_global)

    # llama.cpp gemma3 also stores the local rope base + sliding window + pattern
    try:
        writer.add_rope_freq_base_swa(cfg.rope_theta_local)
        writer.add_sliding_window(cfg.sliding_window)
        writer.add_attn_sliding_window_pattern(cfg.global_attn_every_n)
    except AttributeError:
        # older `gguf` python package versions may not expose these setters;
        # the model will still load but sliding-window/rope-local metadata
        # will be missing (llama.cpp will fall back to global rope everywhere).

        #AttributeError: 'GGUFWriter' object has no attribute 'add_attn_sliding_window_pattern'. Did you mean: 'add_sliding_window_pattern'?

        #raise


        pass

    if cfg.final_logit_softcap and cfg.final_logit_softcap > 0:
        try:
            writer.add_final_logit_softcapping(cfg.final_logit_softcap)
        except AttributeError:
            raise
            pass

    # ---------------- tokenizer ----------------
    n_tokens = len(tokenizer)
    vocab = tokenizer.tokenizer.get_vocab()  # {token_str: id}
    tokens = [""] * n_tokens
    toktypes = [gguf.TokenType.NORMAL] * n_tokens
    for tok_str, tid in vocab.items():
        if 0 <= tid < n_tokens:
            tokens[tid] = tok_str

    special_ids = {
        tokenizer.pad_id: gguf.TokenType.CONTROL,
        tokenizer.eos_id: gguf.TokenType.CONTROL,
        tokenizer.bos_id: gguf.TokenType.CONTROL,
        tokenizer.unk_id: gguf.TokenType.UNKNOWN,
        tokenizer.start_of_turn_id: gguf.TokenType.CONTROL,
        tokenizer.end_of_turn_id: gguf.TokenType.CONTROL,
    }
    for tid, ttype in special_ids.items():
        toktypes[tid] = ttype

    # pull merges out of the tokenizer's own serialized state
    tok_json = json.loads(tokenizer.tokenizer.to_str())
    raw_merges = tok_json.get("model", {}).get("merges", [])
    merges = []
    for m in raw_merges:
        if isinstance(m, list):
            merges.append(f"{m[0]} {m[1]}")
        else:
            merges.append(m)

    writer.add_tokenizer_model("gpt2")  # byte-BPE; use "llama"/unigram if using real sentencepiece
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(tokenizer.bos_id)
    writer.add_eos_token_id(tokenizer.eos_id)
    writer.add_unk_token_id(tokenizer.unk_id)
    writer.add_pad_token_id(tokenizer.pad_id)

    # ---------------- chat formatting metadata ----------------
    # Training always packs: <bos><start_of_turn>{input}<end_of_turn><start_of_turn>{output}<end_of_turn><eos>
    # Without these, llama.cpp/llama-server will NOT prepend <bos> and will NOT
    # know how to wrap a user turn, so raw inference sees out-of-distribution
    # input and the model degenerates into gibberish (this was the actual bug).
    try:
        writer.add_add_bos_token(True)
        writer.add_add_eos_token(False)
    except AttributeError:
        raise
        pass

    sot = cfg.start_of_turn_token
    eot = cfg.end_of_turn_token
    chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        f"{sot}{{{{ message['content'] }}}}{eot}"
        "{% endfor %}"
        f"{sot}"
    )
    try:
        writer.add_chat_template(chat_template)
    except AttributeError:
        raise
        pass

    # ---------------- tensors ----------------
    def W(name):
        # our Linear stores (in, out); ggml wants (out, in)
        return _to_numpy(params[name]).T.astype(np.float32).copy()

    def Bnorm(name):
        return (_to_numpy(params[name]) + 1.0).astype(np.float32).copy()

    writer.add_tensor("token_embd.weight", _to_numpy(params["embed_tokens"]).astype(np.float32))

    for i in range(L):
        pre = f"h{i}."
        writer.add_tensor(f"blk.{i}.attn_q.weight", W(pre + "attn.wq"))
        writer.add_tensor(f"blk.{i}.attn_k.weight", W(pre + "attn.wk"))
        writer.add_tensor(f"blk.{i}.attn_v.weight", W(pre + "attn.wv"))
        writer.add_tensor(f"blk.{i}.attn_output.weight", W(pre + "attn.wo"))
        writer.add_tensor(f"blk.{i}.attn_q_norm.weight", Bnorm(pre + "attn.q_norm_w"))
        writer.add_tensor(f"blk.{i}.attn_k_norm.weight", Bnorm(pre + "attn.k_norm_w"))

        writer.add_tensor(f"blk.{i}.attn_norm.weight", Bnorm(pre + "input_norm.w"))
        writer.add_tensor(f"blk.{i}.post_attention_norm.weight", Bnorm(pre + "post_attn_norm.w"))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", Bnorm(pre + "pre_ffn_norm.w"))
        writer.add_tensor(f"blk.{i}.post_ffw_norm.weight", Bnorm(pre + "post_ffn_norm.w"))

        writer.add_tensor(f"blk.{i}.ffn_gate.weight", W(pre + "mlp.gate"))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", W(pre + "mlp.up"))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", W(pre + "mlp.down"))

    writer.add_tensor("output_norm.weight", Bnorm("final_norm.w"))
    # output.weight omitted on purpose -> llama.cpp reuses token_embd.weight (tied embeddings)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Exported GGUF model to {path}")

# Run block to test export functionality if run directly
if __name__ == "__main__":
    import os
    import glob

    from gemma3_utils import Config, GemmaTokenizer, Gemma3

    CKPT_DIR = "checkpoints"

    cfg = Config.load(os.path.join(CKPT_DIR, "config.json"))

    tokenizer = GemmaTokenizer(cfg)
    tokenizer.load(os.path.join(CKPT_DIR, "tokenizer.json"))

    model = Gemma3(cfg)

    checkpoint_files = glob.glob(os.path.join(CKPT_DIR, "step_*.npz"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No step_*.npz checkpoints found in {CKPT_DIR}")

    latest = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    print(f"Loading checkpoint: {latest}")
    step, _ = model.load_checkpoint(latest)
    print(f"Loaded step {step}")

    out_path = os.path.join(CKPT_DIR, "model.gguf")
    export_gguf(model, tokenizer, out_path)