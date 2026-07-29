#!/usr/bin/env python3
"""
Agricultural Safety AI - Web GIF Viewer
Serves the HTML interface for viewing compiled demo GIFs
"""

import http.server
import socketserver
import os
import json
import tempfile
from pathlib import Path
import argparse
import webbrowser
import time

class DemoDataHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/get_demo_data.php':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            # Find latest demo data
            demo_data = find_demo_data()
            self.wfile.write(json.dumps(demo_data).encode())
        else:
            # Serve static files (HTML, etc.)
            super().do_GET()

def find_demo_data():
    """Find the latest demo data and return paths"""
    temp_base = Path(tempfile.gettempdir())
    demo_dirs = []

    for item in temp_base.iterdir():
        if item.is_dir() and (item / "demo_frames").exists():
            demo_dirs.append(item)

    if not demo_dirs:
        return {"error": "No demo outputs found"}

    # Sort by modification time, most recent first
    demo_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    demo_dir = demo_dirs[0]

    result = {
        "demo_dir": str(demo_dir),
        "gif_path": None,
        "stats": None
    }

    # Check for GIF
    gif_path = demo_dir / "compiled_demo.gif"
    if gif_path.exists():
        result["gif_path"] = f"/temp_demo/{gif_path.name}"

    # Load stats
    stats_path = demo_dir / "demo_stats.json"
    if stats_path.exists():
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                result["stats"] = json.load(f)
        except Exception as e:
            result["stats_error"] = str(e)

    return result

def main():
    parser = argparse.ArgumentParser(description="Serve web interface for viewing demo GIFs")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")

    args = parser.parse_args()

    # Change to the directory containing the HTML file
    os.chdir(Path(__file__).parent)

    # Create symbolic link to temp demo directory for serving
    create_temp_link()

    handler = DemoDataHandler

    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        url = f"http://{args.host}:{args.port}/gif_display.html"
        print("🚜 Agricultural Safety AI - GIF Viewer")
        print(f"🌐 Server running at: {url}")
        print("📊 Serving demo data from temp directory")
        print("🔄 Press Ctrl+C to stop")

        if not args.no_browser:
            print("🌐 Opening browser...")
            time.sleep(1)  # Give server time to start
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

def create_temp_link():
    """Create a symbolic link to the latest demo directory for web serving"""
    temp_base = Path(tempfile.gettempdir())
    demo_dirs = []

    for item in temp_base.iterdir():
        if item.is_dir() and (item / "demo_frames").exists():
            demo_dirs.append(item)

    if demo_dirs:
        demo_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        demo_dir = demo_dirs[0]

        link_path = Path("temp_demo")
        if link_path.exists():
            link_path.unlink()

        try:
            link_path.symlink_to(demo_dir, target_is_directory=True)
            print(f"🔗 Linked demo directory: {demo_dir}")
        except Exception as e:
            print(f"⚠️  Could not create symlink: {e}")

if __name__ == "__main__":
    main()