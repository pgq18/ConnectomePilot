"""Loopback-only browser playground. Simulation advances in its own thread."""
from __future__ import annotations

import argparse
import errno
import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener

from .brain import DATA, FlyBrain, data_status
from .controllers import Controller
from .world import NavigationWorld
from .ppo_controller import PPOController, policy_status

ROOT = Path(__file__).resolve().parents[1]


class Lab:
    def __init__(self):
        self.lock = threading.RLock()
        self.world = NavigationWorld()
        self.controller = Controller("reflex")
        self.brain = None
        self.paused = True
        self.telemetry = {"base": [0, 0], "residual": [0, 0], "final": [0, 0], "neural": {}}
        self.error = ""
        self.elapsed_ms = 0

    def snapshot(self):
        with self.lock:
            return {"world": self.world.state(), "mode": self.controller.mode,
                    "gain": self.controller.gain, "paused": self.paused,
                    "telemetry": self.telemetry, "brain": data_status(), "error": self.error,
                    "policies": policy_status(),
                    "step_ms": self.elapsed_ms}

    def command(self, body):
        with self.lock:
            cmd = body.get("command")
            if cmd == "pause":
                self.paused = True
            elif cmd == "play":
                if self.world.status != "running":
                    raise ValueError("请先重置已结束的场景")
                self.paused = False
            elif cmd == "reset":
                seed = int(body.get("seed", self.world.seed))
                scenario = body.get("scenario", self.world.scenario)
                mode = body.get("mode", self.controller.mode)
                gain = float(body.get("gain", self.controller.gain))
                if not math.isfinite(gain) or not 0 <= gain <= 2:
                    raise ValueError("Residual gain must be between 0 and 2")
                if mode == "fly" and self.brain is None:
                    self.brain = FlyBrain()
                if mode in ("ppo-rays", "ppo-fly"):
                    controller = PPOController(mode, self.brain, gain)
                    if controller.brain is not None:
                        self.brain = controller.brain
                else:
                    controller = Controller(mode, self.brain, gain)
                world = NavigationWorld(seed, scenario)
                if (seed, scenario) == (self.world.seed, self.world.scenario):
                    world.set_goal(*self.world.goal)
                controller.reset(seed)
                self.world, self.controller = world, controller
                self.telemetry = {"base": [0, 0], "residual": [0, 0], "final": [0, 0], "neural": {}}
                self.paused, self.error = True, ""
                self.elapsed_ms = 0
            elif cmd == "goal":
                self.world.set_goal(body["x"], body["y"])
                self.controller.reset(self.world.seed)
                self.telemetry = {"base": [0, 0], "residual": [0, 0], "final": [0, 0], "neural": {}}
                self.paused = self.paused or self.world.status != "running"
                self.error, self.elapsed_ms = "", 0
            elif cmd == "gain":
                gain = float(body["value"])
                if not math.isfinite(gain) or not 0 <= gain <= 2:
                    raise ValueError("Invalid gain")
                self.controller.gain = gain
            else:
                raise ValueError("Unknown command")
        return self.snapshot()

    def run(self):
        while True:
            start = time.perf_counter()
            with self.lock:
                if not self.paused and self.world.status == "running":
                    try:
                        action, self.telemetry = self.controller.act(self.world.observe())
                        self.world.step(action)
                        self.elapsed_ms = round((time.perf_counter()-start)*1000, 1)
                        if self.world.status != "running":
                            self.paused = True
                    except Exception as exc:
                        self.error, self.paused = str(exc), True
            time.sleep(max(0.005, 0.1-(time.perf_counter()-start)))


class Server(ThreadingHTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the local playground in the default browser")
    args = parser.parse_args()
    lab = Lab()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, payload, mime="application/json"):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/api/state":
                return self.send(200, lab.snapshot())
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if route not in files:
                return self.send(404, {"error": "Not found"})
            name, mime = files[route]
            self.send(200, (ROOT/"web"/name).read_bytes(), mime)

        def do_POST(self):
            if self.path != "/api/command":
                return self.send(404, {"error": "Not found"})
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"):
                return self.send(403, {"error": "Cross-origin commands are not allowed"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 4096:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                self.send(200, lab.command(body))
            except (ValueError, KeyError, FileNotFoundError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception as exc:
                self.send(500, {"error": str(exc)})

    url = f"http://127.0.0.1:{args.port}"
    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        # A second double-click should reopen this project's running playground.
        # Check its project data path before treating an occupied port as ours.
        try:
            with build_opener(ProxyHandler({})).open(url+"/api/state", timeout=3) as response:
                existing = json.load(response)
            ours = existing.get("brain", {}).get("path") == str(DATA)
        except Exception:
            ours = False
        if not ours:
            raise SystemExit(f"端口 {args.port} 已被其他程序占用。请使用 --port 指定其他端口。") from exc
        print(f"小游戏已在运行：{url}", flush=True)
        if args.open:
            webbrowser.open(url)
        return
    threading.Thread(target=lab.run, daemon=True).start()
    print(f"Fruit Fly Lab: {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
