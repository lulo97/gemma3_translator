# import gguf

# reader = gguf.GGUFReader('checkpoints/model.gguf')

# print("=== ALL METADATA ===")
# for field in reader.fields.values():
#     val = field.parts[field.data[0]] if field.data else None
#     print(f"{field.name} = {val}")

# print("\n=== TENSOR COUNT ===")
# print(len(reader.tensors))

# print("\n=== TENSOR NAMES (first 30) ===")
# for t in reader.tensors[:30]:
#     print(t.name, t.shape, t.tensor_type)

# print("\n=== Does 'output.weight' exist? (should NOT, since tied embeddings) ===")
# print(any(t.name == "output.weight" for t in reader.tensors))

# print("\n=== attn_sliding_window_pattern specifically ===")
# print([f.name for f in reader.fields.values() if "pattern" in f.name.lower()])

from gemma3_utils import Config, GemmaTokenizer, Gemma3, generate

cfg = Config.load("checkpoints/config.json")
tokenizer = GemmaTokenizer(cfg)
tokenizer.load("checkpoints/tokenizer.json")
model = Gemma3(cfg)
step, _ = model.load_checkpoint("checkpoints/step_1200.npz")  # your latest checkpoint

out = generate(model, tokenizer, "i have a cat.", max_new_tokens=50, do_sample=False)
print(out)