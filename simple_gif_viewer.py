#!/usr/bin/env python3
"""
Agricultural Safety AI - Simple GIF Viewer
Direct file serving for demo GIFs without symlinks
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
import shutil

class SimpleGIFHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/demo/'):
            # Serve demo files directly
            demo_data = find_demo_data()
            if demo_data.get('demo_dir'):
                demo_dir = Path(demo_data['demo_dir'])
                file_path = self.path.replace('/demo/', '')

                if file_path == 'compiled_demo.gif':
                    gif_path = demo_dir / 'compiled_demo.gif'
                    if gif_path.exists():
                        self.send_response(200)
                        self.send_header('Content-type', 'image/gif')
                        self.send_header('Content-length', os.path.getsize(gif_path))
                        self.end_headers()
                        with open(gif_path, 'rb') as f:
                            self.wfile.write(f.read())
                        return

                elif file_path == 'demo_stats.json':
                    stats_path = demo_dir / 'demo_stats.json'
                    if stats_path.exists():
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        with open(stats_path, 'r', encoding='utf-8') as f:
                            self.wfile.write(f.read().encode())
                        return

            self.send_error(404, "Demo file not found")
            return

        elif self.path == '/get_demo_data.php':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            demo_data = find_demo_data()
            # Modify paths for web serving
            if demo_data.get('gif_path'):
                demo_data['gif_path'] = '/demo/compiled_demo.gif'

            self.wfile.write(json.dumps(demo_data).encode())
            return

        else:
            # Serve static files from current directory
            super().do_GET()

def find_demo_data():
    """Find the latest demo data"""
    temp_base = Path(tempfile.gettempdir())
    demo_dirs = []

    for item in temp_base.iterdir():
        if item.is_dir() and (item / "demo_frames").exists():
            demo_dirs.append(item)

    if not demo_dirs:
        return {"error": "No demo outputs found", "message": "Run a demo first"}

    # Sort by modification time, most recent first
    demo_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    demo_dir = demo_dirs[0]

    result = {
        "demo_dir": str(demo_dir),
        "gif_exists": (demo_dir / "compiled_demo.gif").exists(),
        "stats_exists": (demo_dir / "demo_stats.json").exists(),
        "frames_count": len(list((demo_dir / "demo_frames").glob("frame_*.jpg"))) if (demo_dir / "demo_frames").exists() else 0
    }

    # Load stats if available
    stats_path = demo_dir / "demo_stats.json"
    if stats_path.exists():
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                result["stats"] = json.load(f)
        except Exception as e:
            result["stats_error"] = str(e)

    return result

def main():
    parser = argparse.ArgumentParser(description="Simple web server for viewing demo GIFs")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")

    args = parser.parse_args()

    # Change to the directory containing the HTML file
    os.chdir(Path(__file__).parent)

    handler = SimpleGIFHandler

    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        url = f"http://{args.host}:{args.port}/gif_display.html"
        print("🚜 Agricultural Safety AI - Simple GIF Viewer")
        print(f"🌐 Server running at: {url}")
        print("🎬 Serving demo GIFs and statistics")
        print("🔄 Press Ctrl+C to stop")

        if not args.no_browser:
            print("🌐 Opening browser in 2 seconds...")
            time.sleep(2)
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main()