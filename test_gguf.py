import json
import os
import subprocess
import sys
import time

import requests

llama_cli_executable = "llama-cli"
llama_server_executable = "llama-server"  # must be on PATH, or use full path
model_path = r"checkpoints\model.gguf"
prompt_text = "the bird ask chicken for dinner."

PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"


def run_cmd(cmd):
    """Utility to run shell commands and print output."""
    print(f"\n[EXEC] {cmd}")
    # ADD encoding="utf-8" HERE
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8")
    print(f"[STDOUT]\n{res.stdout}")
    if res.stderr:
        print(f"[STDERR]\n{res.stderr}")
    return res.returncode == 0


def quick_cli_sanity_check():
    """Quick non-interactive llama-cli smoke test with deterministic sampling."""
    cmd = [
        llama_cli_executable,
        "-m", model_path,
        "-p", prompt_text,
        "--temp", "0",
        "-n", "64",       # limit tokens so it doesn't run forever
        "--single-turn",  # avoid dropping into interactive REPL, if supported by your build
    ]
    proc = subprocess.Popen(cmd)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("Process hung or entered interactive mode. Forcing termination...")
        proc.kill()
        proc.wait()


def main():
    if not os.path.exists(model_path):
        print(f"Error: Model not found at path '{model_path}'")
        sys.exit(1)

    quick_cli_sanity_check()

    # 1. Start llama-server on port 8080
    print(f"Starting server with model: {model_path} on port {PORT}...")
    server_cmd = [
        llama_server_executable,
        "--model", model_path,
        "--port", str(PORT),
        "--host", "127.0.0.1",
    ]

    server_process = subprocess.Popen(server_cmd)

    try:
        # 2. Poll /health endpoint until it responds OK
        health_url = f"{BASE_URL}/health"
        print(f"Waiting for server to become healthy at {health_url}...")

        max_retries = 30
        healthy = False
        for i in range(max_retries):
            if server_process.poll() is not None:
                print("Error: server process exited before becoming healthy.")
                break
            try:
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    print(f"Health check passed! Response: {resp.text.strip()}")
                    healthy = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        if not healthy:
            print("Error: Server failed to reach healthy status within time limit.")
            return

        # 3. Test CURL /v1/completions (deterministic: temperature=0)
        v1_curl = (
            f'curl -X POST "{BASE_URL}/v1/completions" '
            f'-H "Content-Type: application/json" '
            f'-d "{{\\"prompt\\": \\"{prompt_text}\\", \\"temperature\\": 0}}"'
        )
        run_cmd(v1_curl)

        # 4. Test CURL /completions (legacy/alias route)
        legacy_curl = (
            f'curl -X POST "{BASE_URL}/completions" '
            f'-H "Content-Type: application/json" '
            f'-d "{{\\"prompt\\": \\"{prompt_text}\\", \\"temperature\\": 0}}"'
        )
        run_cmd(legacy_curl)

        payload = {
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0
        }

        # Convert dict to JSON string and escape double quotes for Windows CMD shell execution
        payload_json = json.dumps(payload).replace('"', '\\"')

        chat_curl = (
            f'curl -X POST "{BASE_URL}/v1/chat/completions" '
            f'-H "Content-Type: application/json" '
            f'-d "{payload_json}"'
        )

        run_cmd(chat_curl)

    finally:
        # 5. Shut down server and release port 8080
        print("\nShutting down server on port 8080...")
        if server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
        print("Server stopped and port 8080 cleared.")


if __name__ == "__main__":
    main()