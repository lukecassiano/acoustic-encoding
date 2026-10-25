"""Serve the CAMP interface locally, where it may reach Apple's iTunes Search API.

Pages on claude.ai cannot contact other sites, so song search only works in a
local copy. This serves web/ at / and the exported bundle (data/web/) at /data/,
the same layout the published artifact uses, and wraps index.html in the
document skeleton the artifact host would add.

Usage:
  python -m src.serve_web            # http://localhost:8000
  python -m src.serve_web --port 8080
"""

import argparse
import http.server
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB, DATA = ROOT / "web", ROOT / "data" / "web"
SKELETON = ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            "<style>:root{color-scheme:light}body{margin:0;font:14px system-ui,sans-serif}"
            "img{max-width:100%}[hidden]{display:none!important}</style></head><body>")


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
                      ".txt": "text/plain; charset=utf-8", ".html": "text/html; charset=utf-8"}

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        if path.startswith("/data/"):
            return str(DATA / path[len("/data/"):])
        return str(WEB / (path.lstrip("/") or "index.html"))

    def do_GET(self):
        if self.path.split("?", 1)[0] in ("/", "/index.html"):
            body = (SKELETON + (WEB / "index.html").read_text(encoding="utf-8") + "</body></html>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    port = ap.parse_args().port
    print(f"CAMP at http://localhost:{port}  (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Handler)).serve_forever()


if __name__ == "__main__":
    main()
