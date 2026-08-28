CLOUD GPU:
- python
- uv
- venv
- install packages
- clone code
- copy train.jsonl file
- python gemma3_app.py
- done then copy back
    + checkpoints\train_state.json 
    + checkpoints\step_5000.npz

(gemma3_translator) C:\Users\ADMIN\Desktop\gemma3_translator>uv run python --version && uv pip list
Python 3.14.5
Package            Version
------------------ ---------
anyio              4.14.2
certifi            2026.7.22
charset-normalizer 3.5.1
click              8.5.0
colorama           0.4.6
cuda-pathfinder    1.7.0
cupy-cuda12x       14.2.0
filelock           3.32.4
fsspec             2026.7.0
gguf               0.19.0
h11                0.16.0
hf-xet             1.6.0
httpcore           1.0.9
httpx              0.28.1
huggingface-hub    1.29.0
idna               3.19
numpy              2.5.2
packaging          26.3
pyyaml             6.0.3
regex              2026.7.19
requests           2.34.2
tiktoken           0.14.0
tokenizers         0.23.1
tqdm               4.70.0
typing-extensions  4.16.0
urllib3            2.7.0