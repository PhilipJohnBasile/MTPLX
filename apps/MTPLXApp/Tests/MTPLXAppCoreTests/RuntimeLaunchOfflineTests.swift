import Darwin
import XCTest

@testable import MTPLXAppCore

/// Pressing Play, restarting from Settings and the first app launch must not
/// wait on mtplx.com. The runtime decision is local, and the advisory
/// manifest fetch gives up within a few seconds when the host is silent.
final class RuntimeLaunchOfflineTests: XCTestCase {
    /// A loopback listener that completes TCP handshakes (the kernel does
    /// that from the backlog) but never accepts or answers. A client that
    /// connects to it sits waiting for bytes until its own timeout fires,
    /// which is exactly the shape of a firewalled or captive-portal Mac.
    private final class SilentTCPServer {
        let port: Int
        private let socketFD: Int32

        init() throws {
            let descriptor = socket(AF_INET, SOCK_STREAM, 0)
            guard descriptor >= 0 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
            }
            var address = sockaddr_in()
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            address.sin_family = sa_family_t(AF_INET)
            address.sin_port = in_port_t(0).bigEndian
            address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
            let bindResult = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard bindResult == 0, Darwin.listen(descriptor, 8) == 0 else {
                let failure = errno
                Darwin.close(descriptor)
                throw POSIXError(POSIXErrorCode(rawValue: failure) ?? .ENOTSUP)
            }
            var length = socklen_t(MemoryLayout<sockaddr_in>.size)
            var bound = sockaddr_in()
            let nameResult = withUnsafeMutablePointer(to: &bound) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.getsockname(descriptor, $0, &length)
                }
            }
            guard nameResult == 0 else {
                let failure = errno
                Darwin.close(descriptor)
                throw POSIXError(POSIXErrorCode(rawValue: failure) ?? .ENOTSUP)
            }
            socketFD = descriptor
            port = Int(UInt16(bigEndian: bound.sin_port))
        }

        var manifestURL: URL {
            URL(string: "http://127.0.0.1:\(port)/releases/latest.json")!
        }

        deinit {
            Darwin.close(socketFD)
        }
    }

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-runtime-offline-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeExecutable(named name: String) throws -> URL {
        let directory = temporaryDirectory()
        let url = directory.appendingPathComponent(name)
        try "#!/bin/sh\necho 'mtplx 1.0.0 (1.0.0)'\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    /// The whole launch-path runtime decision, with the manifest host
    /// silent, returns the installed runtime in well under a second. Before
    /// the fix this awaited the manifest on the shared session and sat for
    /// its 60 s default timeout on every Play.
    func testStartPathRuntimeDecisionNeverWaitsOnTheManifestHost() async throws {
        let server = try SilentTCPServer()
        let installed = try makeExecutable(named: "mtplx")
        let service = MTPLXRuntimeUpdateService(
            manifestURL: server.manifestURL,
            environment: [
                "PATH": installed.deletingLastPathComponent().path,
                "HOME": temporaryDirectory().path,
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
            ]
        )

        let started = ContinuousClock.now
        let runtime = try await service.prepareRuntimeForLaunch()
        let elapsed = ContinuousClock.now - started

        XCTAssertEqual(runtime.path, installed.path)
        XCTAssertLessThan(elapsed, .seconds(1), "start-path runtime decision took \(elapsed)")
    }

    /// The manifest fetch itself is bounded: a connected-but-silent host
    /// produces a timeout error after the short request timeout, not after
    /// the minute the shared session would have allowed.
    func testManifestFetchGivesUpWithinTheShortTimeout() async throws {
        let server = try SilentTCPServer()
        let service = MTPLXRuntimeUpdateService(
            manifestURL: server.manifestURL,
            environment: ["HOME": temporaryDirectory().path]
        )

        let started = ContinuousClock.now
        var thrown: Error?
        do {
            _ = try await service.fetchManifest()
            XCTFail("a silent manifest host must not yield a manifest")
        } catch {
            thrown = error
        }
        let elapsed = ContinuousClock.now - started

        let urlError = try XCTUnwrap(thrown as? URLError, "expected a URLError, got \(String(describing: thrown))")
        XCTAssertEqual(urlError.code, .timedOut)
        let timeout = MTPLXRuntimeUpdateService.manifestRequestTimeout
        XCTAssertGreaterThanOrEqual(elapsed, .seconds(timeout - 1), "gave up before the configured timeout: \(elapsed)")
        XCTAssertLessThan(elapsed, .seconds(timeout + 3), "fetch outlived the configured timeout: \(elapsed)")
    }

    /// A refresh that lands after the launch path decided still reports the
    /// installed runtime and marks the manifest as unavailable instead of
    /// failing or waiting.
    func testRefreshSnapshotWithSilentHostReportsInstalledRuntimeWithoutManifest() async throws {
        let server = try SilentTCPServer()
        let installed = try makeExecutable(named: "mtplx")
        let service = MTPLXRuntimeUpdateService(
            manifestURL: server.manifestURL,
            environment: [
                "PATH": installed.deletingLastPathComponent().path,
                "HOME": temporaryDirectory().path,
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
            ]
        )

        let started = ContinuousClock.now
        let snapshot = await service.refreshSnapshot()
        let elapsed = ContinuousClock.now - started

        XCTAssertEqual(snapshot.cliPath, installed.path)
        XCTAssertEqual(snapshot.cliVersion, "1.0.0")
        XCTAssertNil(snapshot.latestAppVersion)
        XCTAssertEqual(snapshot.action, .useExisting)
        XCTAssertLessThan(elapsed, .seconds(MTPLXRuntimeUpdateService.manifestRequestTimeout + 5), "\(elapsed)")
    }
}
