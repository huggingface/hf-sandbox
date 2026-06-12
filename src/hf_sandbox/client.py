"""Sandbox client. Use from the master process."""

import atexit
import base64
import functools
import secrets
import subprocess
import time
import uuid
from pathlib import Path

import httpx
from huggingface_hub import cancel_job, get_token, inspect_job, run_job
from huggingface_hub.utils import send_telemetry

# Must match `PORT` in server.py (the server runs in a separate process inside
# the job and cannot import from this module).
_PORT = 8000

_active: set["Sandbox"] = set()


@atexit.register
def _terminate_all_active():
    for sb in list(_active):
        try:
            sb.terminate(_reason="atexit")
        except Exception:
            pass


def _telemetry(topic: str, data: dict) -> None:
    from hf_sandbox import __version__
    try:
        send_telemetry(
            topic=f"hf-sandbox/{topic}",
            library_name="hf-sandbox",
            library_version=__version__,
            user_agent=data,
        )
    except Exception:
        pass


_FASTAPI_VERSION = "0.115.0"
_UVICORN_VERSION = "0.30.6"


_ENSURE_PYTHON = """\
if ! command -v python3 > /dev/null 2>&1 && ! command -v python > /dev/null 2>&1; then
    if command -v apt-get > /dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -q python3 python3-pip
    elif command -v apk > /dev/null 2>&1; then
        apk add --no-cache python3 py3-pip
    elif command -v yum > /dev/null 2>&1; then
        yum install -y -q python3 pip3
    elif command -v dnf > /dev/null 2>&1; then
        dnf install -y -q python3 pip3
    else
        echo "hf-sandbox: cannot install Python — no supported package manager found" >&2
        exit 1
    fi
fi
PYTHON=$(command -v python3 || command -v python)
"""


@functools.cache
def _bootstrap() -> str:
    server_src = (Path(__file__).parent / "server.py").read_text()
    return f"""set -e
{_ENSURE_PYTHON}
$PYTHON -m pip install -q --break-system-packages fastapi=={_FASTAPI_VERSION} uvicorn=={_UVICORN_VERSION}
cat > /tmp/server.py << 'PYEOF'
{server_src}
PYEOF
exec $PYTHON -u /tmp/server.py
"""


class Sandbox:
    def __init__(self, job_id: str, url: str, sandbox_token: str, hf_token: str):
        self.job_id = job_id
        self.url = url
        # `Authorization` is consumed by the jobs proxy (HF token + namespace
        # gate). `X-Sandbox-Token` is forwarded to the in-pod RPC server and
        # gates the actual exec/read/write endpoints.
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {hf_token}",
                "X-Sandbox-Token": sandbox_token,
            },
        )
        self._session_id = uuid.uuid4().hex
        self._started_at = time.time()
        self._terminated = False

    @classmethod
    def create(cls, image: str, flavor: str = "cpu-basic", timeout: str = "1h",
               forward_hf_token: bool = False):
        hf_token = get_token()
        if hf_token is None:
            raise RuntimeError("No HF token found. Run `hf auth login` first.")
        sandbox_token = secrets.token_urlsafe(32)
        job_secrets = {"HF_SANDBOX_TOKEN": sandbox_token}
        if forward_hf_token:
            job_secrets["HF_TOKEN"] = hf_token
        job = run_job(
            image=image,
            command=["bash", "-c", _bootstrap()],
            secrets=job_secrets,
            flavor=flavor,
            timeout=timeout,
            expose=[_PORT],
        )
        # TODO: read from `job.status.expose_urls` once huggingface_hub
        # surfaces it on `JobInfo` (the hub already returns the field).
        url = f"https://{job.id}--{_PORT}.hf.jobs"
        sb = cls(job.id, url, sandbox_token, hf_token)
        sb._wait_healthy()
        _active.add(sb)
        _telemetry("create", {
            "session_id": sb._session_id,
            "flavor": flavor,
            "timeout": timeout,
            "forward_hf_token": forward_hf_token,
        })
        return sb

    _TERMINAL_STAGES = {"ERROR", "CANCELED", "DELETED", "COMPLETED"}

    def _wait_healthy(self, timeout: float = 300):
        # Job has to schedule a pod, run `pip install`, then start uvicorn
        # before the proxy can route — typical cold start is 30-90s, so
        # idle for the first 15s before probing.
        deadline = time.time() + timeout
        time.sleep(min(15, timeout))
        while time.time() < deadline:
            job = inspect_job(job_id=self.job_id)
            stage = str(getattr(job.status.stage, "value", job.status.stage))
            if stage in self._TERMINAL_STAGES:
                msg = getattr(job.status, "message", None) or stage
                raise RuntimeError(
                    f"Sandbox job {self.job_id} failed before becoming healthy "
                    f"(stage={stage}): {msg}"
                )
            try:
                if self._http.get(f"{self.url}/health", timeout=3).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3)
        raise TimeoutError(f"sandbox at {self.url} never became healthy")

    def exec(self, *cmd: str, workdir: str | None = None, stdin: str | None = None,
             timeout: int = 600) -> subprocess.CompletedProcess:
        r = self._http.post(
            f"{self.url}/exec",
            json={"cmd": list(cmd), "workdir": workdir, "stdin": stdin, "timeout": timeout},
            timeout=timeout + 10,
        )
        r.raise_for_status()
        body = r.json()
        return subprocess.CompletedProcess(
            args=list(cmd), returncode=body["rc"], stdout=body["stdout"], stderr=body["stderr"],
        )

    def write_file(self, path: str, content: str | bytes):
        if isinstance(content, str):
            content = content.encode()
        r = self._http.post(
            f"{self.url}/write",
            json={"path": path, "content_b64": base64.b64encode(content).decode()},
        )
        r.raise_for_status()

    def read_file(self, path: str, text: bool = True) -> str | bytes:
        r = self._http.post(f"{self.url}/read", json={"path": path})
        if r.status_code == 404:
            raise FileNotFoundError(r.json().get("detail", path))
        r.raise_for_status()
        data = base64.b64decode(r.json()["content_b64"])
        return data.decode("utf-8") if text else data

    def terminate(self, _reason: str = "user"):
        if self._terminated:
            return
        self._terminated = True
        _telemetry("terminate", {
            "session_id": self._session_id,
            "duration_s": int(time.time() - self._started_at),
            "reason": _reason,
        })
        self._http.close()
        cancel_job(job_id=self.job_id)
        _active.discard(self)
