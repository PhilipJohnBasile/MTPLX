import Foundation
import SwiftData
import XCTest

@testable import MTPLXAppCore

// MARK: - ChatFakeDaemon
//
// A scripted stand-in for the MTPLX daemon's `/v1/chat/completions`
// SSE endpoint, for view-model tests that need to control exactly what
// bytes the client sees (a failure frame, a socket that closes without
// a terminal chunk, a tool-call round). Same Python-per-test pattern as
// the existing chat tests; the caller supplies only the body of the
// chat handler. Inside it, `self` is the request handler, `body` the
// raw request bytes, `count` the 1-based request number, `sse(payload)`
// encodes one SSE frame, and returning without writing `[DONE]` closes
// the socket. Any hit on the cancel endpoint drops a marker file.

struct ChatFakeDaemon {
    let process: Process
    let baseURL: URL
    let cancelMarkerURL: URL

    var cancelWasCalled: Bool {
        FileManager.default.fileExists(atPath: cancelMarkerURL.path)
    }

    func terminate() {
        process.terminate()
    }

    /// Starts the daemon and waits until it answers on its port.
    static func start(chatHandler: String) async throws -> ChatFakeDaemon {
        let port = try freeTCPPort()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-chat-fake-daemon-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let cancelMarkerURL = directory.appendingPathComponent("cancel-was-called")
        let script = directory.appendingPathComponent("fake-chat-daemon")

        let handlerBody = chatHandler
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { "        " + $0 }
            .joined(separator: "\n")

        let python = """
        import json
        import os
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        PORT = \(port)
        CANCEL_MARKER = r'''\(cancelMarkerURL.path)'''

        def sse(payload):
            return ("data: " + json.dumps(payload) + "\\n\\n").encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            count = 0

            def log_message(self, *_args):
                return

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(length) if length else b""
                if self.path.startswith("/v1/mtplx/cancel/"):
                    with open(CANCEL_MARKER, "wb") as f:
                        f.write(b"cancelled")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b"{\\"ok\\": true}")
                    return
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return
                Handler.count += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    self.chat(body, Handler.count)
                    self.wfile.flush()
                except BrokenPipeError:
                    return

            def chat(self, body, count):
        \(handlerBody)

        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
        """

        try ("#!/bin/sh\nexec python3 -u - <<'PY'\n" + python + "\nPY\n")
            .data(using: .utf8)!
            .write(to: script)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: script.path
        )

        let process = Process()
        process.executableURL = script
        try process.run()

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let startupDeadline = Date().addingTimeInterval(5)
        while Date() < startupDeadline {
            if (try? await URLSession.shared.data(from: baseURL)) != nil {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }
        return ChatFakeDaemon(process: process, baseURL: baseURL, cancelMarkerURL: cancelMarkerURL)
    }

    private static func freeTCPPort() throws -> Int {
        let socketFD = socket(AF_INET, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        defer { Darwin.close(socketFD) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(0).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

        var bindAddress = address
        let bindResult = withUnsafePointer(to: &bindAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(socketFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }

        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        var boundAddress = sockaddr_in()
        let nameResult = withUnsafeMutablePointer(to: &boundAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(socketFD, $0, &length)
            }
        }
        guard nameResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        return Int(UInt16(bigEndian: boundAddress.sin_port))
    }
}

// MARK: - View-model test helpers

extension XCTestCase {
    /// A view model over a fresh in-memory store, talking to `daemon`.
    @MainActor
    func makeChatViewModel(
        daemon: ChatFakeDaemon,
        toolFactory: MTPLXChatToolFactory = MTPLXChatToolFactory(),
        maxToolRounds: Int = 1
    ) throws -> (ChatViewModel, ModelContainer) {
        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: daemon.baseURL))
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            toolFactory: toolFactory,
            modelName: { "mtplx-test-model" },
            maxToolRounds: maxToolRounds
        )
        return (viewModel, container)
    }

    /// Polls a MainActor condition until it holds or the deadline
    /// passes, then asserts it.
    @MainActor
    func pollUntil(
        _ label: String,
        timeout: TimeInterval = 10,
        _ condition: @MainActor () throws -> Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if try condition() { return }
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertTrue(try condition(), "timed out waiting for: \(label)", file: file, line: line)
    }

    @MainActor
    func persistedAssistantMessages(
        in container: ModelContainer,
        conversationID: UUID
    ) throws -> [ChatMessage] {
        let descriptor = FetchDescriptor<ChatMessage>(
            predicate: #Predicate<ChatMessage> { message in
                message.conversationID == conversationID
            },
            sortBy: [SortDescriptor(\.createdAt)]
        )
        return try container.mainContext.fetch(descriptor)
            .filter { $0.role == .assistant }
    }
}
