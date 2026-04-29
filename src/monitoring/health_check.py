"""Health check HTTP server."""

from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from threading import Thread
from typing import Dict


@dataclass
class HealthStatus:
    status: str = "ok"
    details: Dict[str, str] = field(default_factory=dict)


class _HealthHandler(BaseHTTPRequestHandler):
    health_status = HealthStatus()

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/health"}:
            self.send_response(404)
            self.end_headers()
            return

        payload = {
            "status": self.health_status.status,
            "details": self.health_status.details,
        }

        body = (json.dumps(payload) + "\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def start_health_server(port: int = 8080, health_status: HealthStatus | None = None) -> Thread:
    """Start a background health check server."""
    if health_status is not None:
        _HealthHandler.health_status = health_status

    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread
