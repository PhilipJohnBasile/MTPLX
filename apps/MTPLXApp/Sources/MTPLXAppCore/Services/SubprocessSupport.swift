import Foundation

// Shared plumbing for every watchdogged subprocess the app spawns.
//
// The #158 postmortem left two invariants no spawn site may violate:
//
// 1. Never wait on a child without a deadline. A wedged child (pip on
//    an unreachable index, a Gatekeeper-stalled first exec, a hung
//    LaunchServices `open`) must cost one bounded timeout window —
//    never park a thread, an async flow, or the main actor forever.
// 2. Never read a pipe only after exit. A child that fills the 64KB
//    kernel pipe buffer before exiting blocks on write while the
//    parent blocks waiting for exit — a mutual deadlock with no error
//    surface.
//
// `SubprocessWatchdog` enforces (1); `SubprocessPipeDrain` and
// `SubprocessTailBuffer` are the two capture flavors that enforce (2).
// Use `SubprocessPipeDrain` when the output is data you parse (JSON
// payloads, version strings, PID lists): its dedicated reader joins on
// EOF, so the final flush of a write-then-exit child is never lost.
// Use `SubprocessTailBuffer` + a readabilityHandler when the output is
// diagnostics for error messages, where a racy final chunk is
// acceptable and a rolling tail is the point.

/// Thread-safe rolling tail of subprocess output. Shared by every
/// watchdogged subprocess runner in the app (runtime installs, fan
/// commands) — pipe reads happen on readabilityHandler threads while
/// the spawning thread waits on the termination semaphore.
final class SubprocessTailBuffer: @unchecked Sendable {
    private let capacity: Int
    private let lock = NSLock()
    private var data = Data()

    init(capacity: Int) {
        self.capacity = max(256, capacity)
    }

    func append(_ chunk: Data) {
        lock.lock()
        data.append(chunk)
        if data.count > capacity {
            data.removeFirst(data.count - capacity)
        }
        lock.unlock()
    }

    func snapshot() -> String {
        String(data: snapshotData(), encoding: .utf8) ?? ""
    }

    func snapshotData() -> Data {
        lock.lock()
        let copy = data
        lock.unlock()
        return copy
    }
}

/// Deadline watchdog for subprocess waits: installs the termination
/// signal before `run()` so a fast exit can never be missed, then
/// bounds the wait with terminate → SIGKILL escalation. Create it
/// BEFORE calling `process.run()`; the watchdog owns the process's
/// `terminationHandler`.
final class SubprocessWatchdog: @unchecked Sendable {
    private let finished = DispatchSemaphore(value: 0)

    init(_ process: Process) {
        process.terminationHandler = { [finished] _ in finished.signal() }
    }

    /// Waits up to `timeout` for the child to exit. On deadline:
    /// terminate, wait `terminateGrace`, SIGKILL, wait `killGrace`.
    /// Returns false on timeout — the child was forcibly reaped (or is
    /// beyond signals); `terminationStatus` is meaningless then.
    @discardableResult
    func wait(
        for process: Process,
        timeout: TimeInterval,
        terminateGrace: TimeInterval = 10,
        killGrace: TimeInterval = 5
    ) -> Bool {
        if finished.wait(timeout: .now() + timeout) == .timedOut {
            process.terminate()
            if finished.wait(timeout: .now() + terminateGrace) == .timedOut {
                kill(process.processIdentifier, SIGKILL)
                _ = finished.wait(timeout: .now() + killGrace)
            }
            return false
        }
        return true
    }

    /// Escalating cancel for user-cancelled unbounded workers (forge
    /// build, HF publish, tune): SIGINT first so the child can clean
    /// up partial artifacts, then terminate, then SIGKILL if it
    /// ignores both — a cancelled worker must never linger as an
    /// invisible GPU/CPU hog. Returns immediately (escalation runs
    /// detached), so it is safe from a stream's onTermination. A child
    /// that honors SIGINT sees exactly the old single-interrupt
    /// behavior.
    static func escalateCancel(
        _ process: Process,
        interruptGrace: TimeInterval = 10,
        terminateGrace: TimeInterval = 10
    ) {
        guard process.isRunning else { return }
        process.interrupt()
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + interruptGrace) {
            guard process.isRunning else { return }
            process.terminate()
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + terminateGrace) {
                if process.isRunning {
                    kill(process.processIdentifier, SIGKILL)
                }
            }
        }
    }
}

/// Lossless single-consumer drain of one subprocess pipe, for output
/// that gets parsed rather than logged. A dedicated blocking reader
/// keeps the child from ever stalling on a full pipe buffer, and
/// `join()` waits for EOF so the final flush of a write-then-exit
/// child is captured — a readabilityHandler tail can lose that last
/// chunk to the termination race, which is fatal when stdout carries a
/// JSON payload. The capacity is a memory fuse against a pathological
/// infinite-spew child, far above any legitimate payload; on overflow
/// the head is dropped (rolling tail) so the child still drains and
/// the parse fails explicitly instead of the app growing without
/// bound. Start the drain after `run()` succeeds — attaching to a pipe
/// whose child never launched would park the reader forever.
final class SubprocessPipeDrain: @unchecked Sendable {
    private let buffer: SubprocessTailBuffer
    private let done = DispatchSemaphore(value: 0)

    init(_ pipe: Pipe, capacity: Int = 8_388_608) {
        let buffer = SubprocessTailBuffer(capacity: capacity)
        self.buffer = buffer
        let handle = pipe.fileHandleForReading
        let done = self.done
        DispatchQueue.global(qos: .userInitiated).async {
            while true {
                let chunk = handle.availableData
                if chunk.isEmpty { break }
                buffer.append(chunk)
            }
            done.signal()
        }
    }

    /// Call after the child has exited: EOF is imminent, so a short
    /// bound suffices. Returns false if EOF never arrived — an orphan
    /// grandchild still holds the write end — in which case
    /// `snapshot()` returns whatever was captured so far.
    @discardableResult
    func join(timeout: TimeInterval = 5) -> Bool {
        done.wait(timeout: .now() + timeout) == .success
    }

    func snapshot() -> String {
        buffer.snapshot()
    }

    func snapshotData() -> Data {
        buffer.snapshotData()
    }
}
