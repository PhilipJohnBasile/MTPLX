import XCTest
@testable import MTPLXAppCore

// Pins the contracts of the shared watchdogged-subprocess plumbing
// behind the #158 sweep: bounded waits reap wedged children, drains
// survive payloads larger than the 64KB kernel pipe buffer without
// losing the final flush, and escalating cancel kills a SIGINT-deaf
// child. Every child here is a stock shell one-liner — fast,
// hermetic, no runtime install required.
final class SubprocessSupportTests: XCTestCase {
    private func shellProcess(_ script: String) -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["-c", script]
        return process
    }

    func testWatchdogWaitReturnsTrueOnFastExit() throws {
        let process = shellProcess("exit 0")
        let watchdog = SubprocessWatchdog(process)
        try process.run()
        XCTAssertTrue(watchdog.wait(for: process, timeout: 10))
        XCTAssertEqual(process.terminationStatus, 0)
    }

    func testWatchdogWaitReapsWedgedChildOnTimeout() throws {
        let process = shellProcess("sleep 30")
        let watchdog = SubprocessWatchdog(process)
        try process.run()
        let start = Date()
        let exited = watchdog.wait(
            for: process,
            timeout: 0.5,
            terminateGrace: 3,
            killGrace: 3
        )
        XCTAssertFalse(exited, "a wedged child must report a timeout, not exit")
        XCTAssertFalse(process.isRunning, "the timeout path must reap the child")
        XCTAssertLessThan(
            Date().timeIntervalSince(start), 8,
            "the bounded wait must not degenerate into waitUntilExit"
        )
    }

    func testPipeDrainCapturesPayloadLargerThanPipeBuffer() throws {
        // 192KB of 'x' — three times the 64KB kernel pipe buffer. The
        // old read-after-exit pattern deadlocks on exactly this child;
        // the drain must capture every byte including the final flush.
        let byteCount = 196_608
        let process = shellProcess(
            "dd if=/dev/zero bs=1024 count=192 2>/dev/null | tr '\\0' 'x'"
        )
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        let watchdog = SubprocessWatchdog(process)
        try process.run()
        let drain = SubprocessPipeDrain(pipe)
        XCTAssertTrue(watchdog.wait(for: process, timeout: 15))
        XCTAssertTrue(drain.join(timeout: 10), "EOF must arrive once the child exits")
        let captured = drain.snapshotData()
        XCTAssertEqual(captured.count, byteCount)
        XCTAssertTrue(captured.allSatisfy { $0 == UInt8(ascii: "x") })
    }

    func testEscalateCancelReapsSigintDeafChild() throws {
        let process = shellProcess("trap '' INT; sleep 30")
        try process.run()
        // Give the shell a beat to install its trap so the test proves
        // escalation, not a lucky early SIGINT.
        Thread.sleep(forTimeInterval: 0.2)
        SubprocessWatchdog.escalateCancel(
            process,
            interruptGrace: 0.5,
            terminateGrace: 5
        )
        let deadline = Date().addingTimeInterval(10)
        while process.isRunning, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        XCTAssertFalse(
            process.isRunning,
            "escalation must terminate a child that ignores SIGINT"
        )
    }
}
