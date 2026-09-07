from pathlib import Path
import sys
p=Path(sys.argv[1])/'apps/MTPLXApp/Tests/MTPLXAppCoreTests/MTPLXAppCoreTests.swift'
s=p.read_text()
for name in ('testBackendHeadlineDecodeUsesRawCompletionTPSBeforeDisplayTPS','testBackendHeadlineDecodeIgnoresCumulativeAndStaleSnapshotMaxDuringLiveRequest'):
    a=s.index('    func '+name+'(');b=s.index('\n    }',a)+6
    block=s[a:b]
    needle='        backend.startMetricsStream()'
    assert block.count(needle)==1
    block=block.replace(needle,needle+'\n        defer { backend.stopMetricsStream() }')
    needle='        XCTAssertEqual('
    i=block.index(needle)
    block=block[:i]+'        print("METRICS_DIAGNOSTIC", backend.connectionState, backend.baseURL, backend.latest as Any, process.isRunning)\n'+block[i:]
    s=s[:a]+block+s[b:]
a=s.index('    func testCancelDuringStartupLeavesAppStoppedAndRestoresFans(');b=s.index('\n    }',a)+6
block=s[a:b]
needle='        XCTAssertEqual(backend.daemonState, .starting)'
assert block.count(needle)==1
block=block.replace(needle,'        await backend.refreshLogs()\n        print("START_DIAGNOSTIC", backend.daemonState, backend.startupPhase, backend.runtimeUpdateFailure as Any, backend.pendingModelDownload as Any, backend.logs)\n'+needle)
s=s[:a]+block+s[b:]
p.write_text(s)
