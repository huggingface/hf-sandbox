<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.png">
  <img alt="hf-sandbox" src="assets/banner-light.png">
</picture>

Modal-style sandbox API on top of Hugging Face Jobs.

> [!IMPORTANT]
> **`hf-sandbox` is moving into `huggingface_hub`.**
>
> This prototype is being upstreamed as a first-class `Sandbox` API (plus an `hf sandbox` CLI). It ships in `huggingface_hub` 1.22.0 (`pip install "huggingface_hub>=1.22.0"`). Use:
>
> ```python
> from huggingface_hub import Sandbox
>
> with Sandbox.create(image="python:3.12") as sbx:
>     sbx.files.write("/app/main.py", "print(40 + 2)")
>     print(sbx.run("python /app/main.py").stdout)    # 42
> ```
>
> Or from the CLI:
>
> ```bash
> id=$(hf sandbox create)
> hf sandbox exec $id -- python -c "print(40 + 2)"   # 42
> hf sandbox cp data.csv $id:/data/data.csv
> hf sandbox kill $id
> ```
>
> This repo will be archived in favor of that.

```python
from hf_sandbox import Sandbox

sb = Sandbox.create(image="python:3.12")
proc = sb.exec("python", "-c", "print(1+1)")  # → CompletedProcess(stdout='2\n', returncode=0, ...)
sb.write_file("/tmp/foo.txt", "hello")
print(sb.read_file("/tmp/foo.txt"))           # → 'hello'
sb.terminate()
```

## How it works

`Sandbox.create()` launches an HF Job that:

1. `pip install`s a tiny FastAPI RPC server (FastAPI + uvicorn)
2. starts the server on `localhost:8000`
3. declares port 8000 as exposed on the Job (`expose=[8000]`), so the HF Jobs proxy registers `https://<job_id>--8000.hf.jobs`

The client computes the URL from the Job id and talks to the sandbox over plain HTTPS through the Jobs proxy.

`exec`, `write_file`, `read_file` are simple authenticated POSTs.
`terminate()` cancels the job.

## Install

```bash
pip install hf-sandbox
```

Requires `hf auth login` (the same token authenticates against the Jobs proxy and, opt-in, is forwarded to the sandbox so it can access HF Hub).

## Limits

- Image must have Python + `pip` (used to install the RPC server).

## Security

The sandbox runs **untrusted code by design**. A few things to be aware of:

- **HF token forwarding is opt-in.** By default, your HF token is *not* exposed to the sandbox. Pass `forward_hf_token=True` to `Sandbox.create()` if your workload needs it. With it enabled, anything running inside the sandbox can read the token from `/proc/self/environ` and use it to act as you on the Hub.
- **Traffic stays on Hugging Face infrastructure.** Requests reach the sandbox through the HF Jobs proxy and require an HF token with read access to the job's namespace; the sandbox URL alone (`https://<job_id>--8000.hf.jobs`) is unauthenticated and returns 401.
- **Sandbox token** is a 256-bit random URL-safe string per sandbox, sent as `X-Sandbox-Token` on every authenticated endpoint and validated by the in-pod RPC server. Defends against namespace-mates being able to reach your sandbox once they pass the proxy.

## Telemetry

`hf-sandbox` reports anonymous usage data to help us understand how the library is used. Two events are sent per sandbox:

- `hf-sandbox/create` — flavor, timeout, whether `forward_hf_token` was set, and a random per-sandbox session id
- `hf-sandbox/terminate` — same session id, duration in seconds, and termination reason

We never send: the image name, commands, file paths, file contents, the tunnel URL, the auth token, your HF token, your username, or anything from inside the sandbox.

Disable by setting `HF_HUB_DISABLE_TELEMETRY=1` (or `DO_NOT_TRACK=1`).
