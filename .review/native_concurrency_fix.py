from pathlib import Path
import sys
root=Path(sys.argv[1])
p=root/'apps/MTPLXApp/Sources/MTPLXAppCore/Stores/MTPLXBackendStore.swift'
s=p.read_text()
old='''            async let health = client.health()
            async let capabilities = client.capabilities()
            async let sessions = client.sessions()
            let fetchedHealth = try await health
            let fetchedCapabilities = try await capabilities
            let fetchedSessions = try await sessions
'''
new='''            // Keep child-task storage in its own async scope. Swift 6.1 can
            // misorder async-let teardown when large results escape a do block
            // into later suspension points (swiftlang/swift#81771).
            let (fetchedHealth, fetchedCapabilities, fetchedSessions) = try await {
                async let health = client.health()
                async let capabilities = client.capabilities()
                async let sessions = client.sessions()
                return try await (health, capabilities, sessions)
            }()
'''
assert s.count(old)==1
p.write_text(s.replace(old,new))
