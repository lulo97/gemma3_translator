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

https://github.com/lulo97/gemma3_translator

uv python install 3.14.5

uv venv --python 3.14.5

uv pip install "cupy-cuda12x==14.2.0" "numpy==2.5.2" "gguf==0.19.0" "tokenizers==0.23.1" "requests==2.34.2"
