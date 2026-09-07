from pathlib import Path
import sys
p=Path(sys.argv[1])/'apps/MTPLXApp/Tests/MTPLXAppCoreTests/MTPLXAppCoreTests.swift'
s=p.read_text()
for name in ('testBackendHeadlineDecodeUsesRawCompletionTPSBeforeDisplayTPS','testBackendHeadlineDecodeIgnoresCumulativeAndStaleSnapshotMaxDuringLiveRequest'):
    a=s.index('    func '+name+'(');b=s.index('\n    }',a)+6
    block=s[a:b]
    old='        defer { process.terminate() }\n'
    new=old+'        try await waitForFixtureListener(port: port, process: process)\n'
    assert block.count(old)==1
    s=s[:a]+block.replace(old,new)+s[b:]
anchor='    private func freeTCPPort() throws -> Int {'
assert s.count(anchor)==1
helper='''    @MainActor
    private func waitForFixtureListener(port: Int, process: Process) async throws {
        // These are decoder/headline tests, not process-startup timing tests.
        // Keep their assertion deadline separate from Python/socket startup.
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            guard process.isRunning else { throw DaemonSupervisorError.healthTimeout }
            let fd = socket(AF_INET, SOCK_STREAM, 0)
            guard fd >= 0 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
            }
            var address = sockaddr_in()
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            address.sin_family = sa_family_t(AF_INET)
            address.sin_port = in_port_t(port).bigEndian
            address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
            let result = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            Darwin.close(fd)
            if result == 0 { return }
            try await Task.sleep(for: .milliseconds(25))
        }
        throw DaemonSupervisorError.healthTimeout
    }

'''
p.write_text(s.replace(anchor,helper+anchor))
