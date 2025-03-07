from http.server import SimpleHTTPRequestHandler
from http.server import HTTPServer as BaseHTTPServer
from socketserver import ThreadingMixIn
import threading
import argparse
import os


class HTTPHandler(SimpleHTTPRequestHandler):
    '''
    Handler that uses server.base_path instead of os.getcwd()
    '''
    def translate_path(self, path):
        path = SimpleHTTPRequestHandler.translate_path(self, path)
        relpath = os.path.relpath(path, os.getcwd())
        fullpath = os.path.join(self.server.base_path, relpath)
        return fullpath.lower()
    


class HTTPServer(ThreadingMixIn, BaseHTTPServer):
    '''
    Main server that servers requests from the given base path
    Creates a new thread for each incoming request to handle asynchronously

    Parameters
    ----------
    base_path : str
        Base path to serve from
    server_address : tuple(str, int)
        Server address, e.g. ("localhost", 8000)
    RequestHandlerClass : class = HTTPHandler
        Class to use for handling requests
    '''
    def __init__(self, base_path, server_address, RequestHandlerClass=HTTPHandler):
        self.base_path = base_path
        BaseHTTPServer.__init__(self, server_address, RequestHandlerClass)


def serve(serve_dir : str, serve_port):
    '''
    Serve the specified directory to the given port

    Parameters
    ----------
    serve_dir : str
        Directory to serve from
    serve_port : int
        Port to serve on
    '''

    # Check that the serve_dir exists
    if not os.path.isdir(serve_dir):
        raise Exception(f"Directory {serve_dir} not accessible from {os.getcwd()}")

    httpd = HTTPServer(serve_dir, ("", int(serve_port)))
    httpd.serve_forever()


def parse_args():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve_dir",  default='/home/vesh/live/', help="Path to serve files.")
    parser.add_argument("--serve_port",  default=8000, help="Port to serve files.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    serve(args.serve_dir, args.serve_port)