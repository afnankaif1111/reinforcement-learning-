"""
Local HTTP Server Runner for Swarm Robotics Interactive Web Visualizer.
Runs a local server and opens web_viewer.html in your default web browser.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080

def start_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = http.server.SimpleHTTPRequestHandler
    
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            url = f"http://localhost:{PORT}/web_viewer.html"
            print("=" * 65)
            print(f" Swarm Robotics Interactive Simulator launched!")
            print(f" Open your browser at: {url}")
            print(" Press Ctrl+C in terminal to stop server.")
            print("=" * 65)
            try:
                webbrowser.open(url)
            except Exception:
                pass
            httpd.serve_forever()
    except OSError as e:
        print(f"Port {PORT} in use, trying port 8081...")
        with socketserver.TCPServer(("", 8081), handler) as httpd:
            url = f"http://localhost:8081/web_viewer.html"
            print(f"Server started at: {url}")
            httpd.serve_forever()

if __name__ == "__main__":
    start_server()
