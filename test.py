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

import glob
import re

from gemma3_utils import Config, GemmaTokenizer, Gemma3, generate

cfg = Config.load("checkpoints/config.json")
tokenizer = GemmaTokenizer(cfg)
tokenizer.load("checkpoints/tokenizer.json")
model = Gemma3(cfg)

checkpoint_files = glob.glob("checkpoints/step_*.npz")

if not checkpoint_files:
    raise FileNotFoundError("No checkpoints found in 'checkpoints/'")

latest_checkpoint = max(
    checkpoint_files,
    key=lambda path: int(re.search(r"step_(\d+)\.npz", path).group(1)),
)

# Load the latest checkpoint
step, _ = model.load_checkpoint(latest_checkpoint)

out = generate(model, tokenizer, "I have money for it.", max_new_tokens=50, do_sample=False)
print(out)

test_prompts = [
    "I have money for it.",
    "I don't have enough money.",
    "She has a lot of work today.",
    "He went to the store yesterday.",
    "We need to finish this before Friday.",
    "They are waiting for the bus.",
    "I don't know what you mean.",
    "Can you help me with this?",
    "What time does the meeting start?",
    "Where did you put my phone?",
    "I think this is a good idea.",
    "I don't think that will work.",
    "She told me about the problem.",
    "He asked me to wait outside.",
    "We should talk about this later.",
    "I was tired after work.",
    "They were happy to see us.",
    "I have never been there before.",
    "Have you seen this movie?",
    "Do you want to come with me?",
    "I would like a cup of coffee.",
    "Could you open the window?",
    "Please give me a few minutes.",
    "I need some time to think.",
    "She is learning English at home.",
    "He is working on a new project.",
    "We are trying to find a solution.",
    "They are planning a trip next month.",
    "I will call you tomorrow.",
    "She will probably arrive late.",
    "He is going to buy a new car.",
    "We need to leave early tomorrow.",
    "I can do it myself.",
    "Can you send me the file?",
    "You should check your email.",
    "You don't have to worry about it.",
    "I want to know the truth.",
    "She wants to become a doctor.",
    "He likes listening to music.",
    "We enjoy spending time together.",
    "They don't like living in the city.",
    "I used to live near here.",
    "She used to work at a bank.",
    "He has been working all day.",
    "We have been waiting for an hour.",
    "I have already finished my work.",
    "She hasn't called me yet.",
    "Have they arrived yet?",
    "I was watching TV when you called.",
    "She was cooking dinner when I got home.",
    "He didn't understand the question.",
    "We didn't know about the change.",
    "Why didn't you tell me earlier?",
    "What are you doing right now?",
    "Why are they waiting outside?",
    "How much does this cost?",
    "How long will it take?",
    "How many people are coming?",
    "Which one do you prefer?",
    "I prefer this one because it is cheaper.",
    "This is more difficult than I expected.",
    "That was the best decision I could make.",
    "The weather is getting colder.",
    "It looks like it is going to rain.",
    "I forgot to bring my umbrella.",
    "She remembered where she left the keys.",
    "He lost his wallet on the train.",
    "We found a nice restaurant nearby.",
    "They bought a house last year.",
    "I need to buy some food.",
    "She wants to go shopping this weekend.",
    "He doesn't eat meat.",
    "We usually have dinner at seven.",
    "They often visit their parents.",
    "I rarely watch television these days.",
    "She always arrives on time.",
    "He never talks about his personal life.",
    "We sometimes go for a walk after dinner.",
    "I really like this song.",
    "She doesn't really understand the situation.",
    "He seems to be very busy.",
    "It sounds like a good plan.",
    "That sounds interesting.",
    "I am not sure about that.",
    "Maybe we should wait a little longer.",
    "I hope everything goes well.",
    "I am afraid we are too late.",
    "She was surprised by the news.",
    "He was disappointed with the result.",
    "We were worried about the weather.",
    "They were excited about the trip.",
    "I am looking for a new job.",
    "She is looking for her glasses.",
    "He is waiting for his friend.",
    "We are talking about the future.",
    "They are thinking about moving abroad.",
    "I have enough money to pay for it.",
    "I don't have enough time to finish it.",
    "She knows how to solve the problem.",
    "He doesn't know where to go.",
    "We need to decide what to do next.",
    "They asked me if I could help them.",
]

for prompt in test_prompts:
    out = generate(model, tokenizer, prompt, max_new_tokens=50, do_sample=False)
    print(f"\nINPUT:  {prompt}")
    print(f"OUTPUT: {out}")