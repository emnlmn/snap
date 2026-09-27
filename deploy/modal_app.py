"""snap demo on Modal — one CPU container running `snap serve`.

Dev:     modal serve deploy/modal_app.py    (ephemeral URL, live reload)
Deploy:  modal deploy deploy/modal_app.py   (persistent https://<ws>--snap-demo-serve.modal.run)

`/` redirects to /playground, so the URL is the demo console itself.
The GGUF downloads into the `snap-models` volume once, then every cold
start loads from there — seconds, not minutes.
"""

import os
import subprocess

import modal

MODEL = "minicpm5-2b"  # tested set in src/models.rs
PORT = 8018            # DEFAULT_PORT — web_server must see the same port

app = modal.App("snap-demo")

image = modal.Image.from_dockerfile("deploy/Dockerfile", context_dir=".")
models = modal.Volume.from_name("snap-models", create_if_missing=True)


@app.function(
    image=image,
    cpu=4,
    memory=4096,
    volumes={"/models": models},
    timeout=3600,
    # min_containers=1,  # warm pool: no cold start, billed while idle
)
@modal.concurrent(max_inputs=8)
@modal.web_server(PORT, startup_timeout=300)
def serve():
    env = {**os.environ, "HF_HOME": "/models", "HF_HUB_CACHE": "/models"}
    subprocess.Popen(
        ["snap", "serve", "--host", "0.0.0.0", "--port", str(PORT), "--model", MODEL],
        env=env,
    )
