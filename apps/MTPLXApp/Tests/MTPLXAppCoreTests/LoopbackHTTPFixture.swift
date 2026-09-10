/// Shared Python HTTP fixture for listeners bound to loopback. No test needs
/// HTTPServer.server_bind's reverse DNS lookup, which can stall a clean runner.
enum LoopbackHTTPFixture {
    static let serverSource = """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as PythonThreadingHTTPServer
    from socketserver import TCPServer

    class ThreadingHTTPServer(PythonThreadingHTTPServer):
        def server_bind(self):
            TCPServer.server_bind(self)
            self.server_name = "localhost"
            self.server_port = self.server_address[1]
    """
}
