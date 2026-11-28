"""Cross-platform launcher for CNN Tutor."""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import venv
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
URL = "http://localhost:8501"
HEARTBEAT_TIMEOUT_SECONDS = 20
FIRST_CONNECTION_TIMEOUT_SECONDS = 90


def virtual_python() -> Path:
    relative_path = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return VENV_DIR.joinpath(*relative_path)


def requirements_are_installed(python: Path) -> bool:
    check = subprocess.run(
        [str(python), "-c", "import streamlit, matplotlib, numpy"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return check.returncode == 0


def prepare_environment() -> Path:
    python = virtual_python()
    if not python.exists():
        print("Creating the virtual environment...")
        venv.create(VENV_DIR, with_pip=True)
        python = virtual_python()
    if not requirements_are_installed(python):
        print("Installing CNN Tutor requirements...")
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(PROJECT_DIR / "requirements.txt")],
            check=True,
        )
    return python


def server_is_ready() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(.25)
        return connection.connect_ex(("127.0.0.1", 8501)) == 0


def stop_process(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


class HeartbeatServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), HeartbeatHandler)
        self.first_heartbeat: float | None = None
        self.last_heartbeat: float | None = None


class HeartbeatHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/heartbeat":
            self.send_error(404)
            return
        now = time.monotonic()
        if self.server.first_heartbeat is None:
            self.server.first_heartbeat = now
        self.server.last_heartbeat = now
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def log_message(self, format: str, *arguments: object) -> None:
        return


def start_heartbeat_server() -> HeartbeatServer:
    server = HeartbeatServer()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def wait_for_streamlit(process: subprocess.Popen[bytes], log_path: Path) -> bool:
    for _ in range(30):
        if server_is_ready():
            return True
        if process.poll() is not None:
            print(f"CNN Tutor could not start. Check {log_path}", file=sys.stderr)
            return False
        time.sleep(1)
    print(f"CNN Tutor did not start in time. Check {log_path}", file=sys.stderr)
    return False


def run_dashboard(python: Path) -> int:
    if server_is_ready():
        webbrowser.open(URL)
        print(f"CNN Tutor is already running at {URL}")
        return 0

    heartbeat_server = start_heartbeat_server()
    heartbeat_port = heartbeat_server.server_address[1]
    log_descriptor, log_name = tempfile.mkstemp(prefix="cnn-tutor-", suffix=".log")
    os.close(log_descriptor)
    log_path = Path(log_name)
    environment = os.environ | {"CNN_TUTOR_HEARTBEAT_PORT": str(heartbeat_port)}

    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            [str(python), "-m", "streamlit", "run", "app/cnn_teaching_lab.py", "--server.headless",
             "true", "--server.port", "8501", "--logger.level", "error"],
            cwd=PROJECT_DIR,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        if not wait_for_streamlit(process, log_path):
            heartbeat_server.shutdown()
            return 1
        webbrowser.open(URL)
        print(f"CNN Tutor is running at {URL}")
        print("Closing the dashboard tab stops the local server.")
        launched_at = time.monotonic()
        try:
            while process.poll() is None:
                now = time.monotonic()
                if heartbeat_server.last_heartbeat is not None:
                    if now - heartbeat_server.last_heartbeat > HEARTBEAT_TIMEOUT_SECONDS:
                        stop_process(process)
                        print("Dashboard tab closed. CNN Tutor stopped.")
                        return 0
                elif now - launched_at > FIRST_CONNECTION_TIMEOUT_SECONDS:
                    stop_process(process)
                    print("No browser connection was detected. CNN Tutor stopped.")
                    return 0
                time.sleep(1)
            return process.returncode
        except KeyboardInterrupt:
            stop_process(process)
            print("CNN Tutor stopped.")
            return 0
        finally:
            heartbeat_server.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-only", action="store_true")
    arguments = parser.parse_args()
    try:
        python = prepare_environment()
    except subprocess.CalledProcessError as error:
        print(f"Installation failed with exit code {error.returncode}.", file=sys.stderr)
        return error.returncode
    if arguments.install_only:
        print("CNN Tutor is installed. Run the platform launcher to open it.")
        return 0
    return run_dashboard(python)


if __name__ == "__main__":
    raise SystemExit(main())
