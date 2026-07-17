import Foundation

public enum DaemonState: Equatable, Sendable {
    case stopped
    case starting
    case warming
    case running
    case degraded(String)
    case stopping
    case crashed(Int32?)
}

public enum DaemonSupervisorError: Error, Equatable, CustomStringConvertible, LocalizedError {
    case alreadyRunning
    case launchFailed(String)
    case healthTimeout
    case portOccupied(pid: Int?, launchID: String?)
    case launchIdentityMismatch(expected: String, observed: String?)
    case fanRampTimeout

    public var description: String {
        switch self {
        case .alreadyRunning:
            return "MTPLX is already running."
        case .launchFailed(let detail):
            return "MTPLX couldn't start: \(detail)"
        case .healthTimeout:
            return "MTPLX took too long to start up."
        case .portOccupied(let pid, let launchID):
            let pidText = pid.map { "pid \($0)" } ?? "unknown pid"
            if let launchID {
                return "Port is already used by MTPLX (\(pidText), launch \(launchID))."
            }
            return "Port is already used by another local app (\(pidText))."
        case .launchIdentityMismatch(let expected, let observed):
            return "MTPLX startup didn't match what we expected. Expected \(expected), got \(observed ?? "nothing")."
        case .fanRampTimeout:
            return "Couldn't get your fans to max in time."
        }
    }

    public var errorDescription: String? { description }
}

public enum DaemonStartupPhase: Equatable, Sendable {
    case idle
    case launching
    case waitingForOwnedHealth
    case rampingFans
    case warming
    case ready
    case failed(String)
}

public final class DaemonSupervisor: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var adoptedProcessID: pid_t?
    private let logStore: BoundedLogStore

    public private(set) var state: DaemonState = .stopped

    public init(logStore: BoundedLogStore = BoundedLogStore()) {
        self.logStore = logStore
    }

    public var logs: BoundedLogStore {
        logStore
    }

    public func isRunning() -> Bool {
        lock.withLock {
            process?.isRunning == true || adoptedProcessID != nil
        }
    }

    public func start(
        command: DaemonCommand,
        healthBaseURL: URL,
        apiKey: String? = nil,
        probeHealth: Bool = true,
        timeoutSeconds: TimeInterval = 300,
        expectedLaunchID: String? = nil,
        requireActualFanRamp: Bool = false,
        adoptExistingAppOwnedDaemon: Bool = false,
        onPhase: (@Sendable (DaemonStartupPhase) -> Void)? = nil
    ) async throws -> HealthPayload? {
        try lock.withLock {
            if process?.isRunning == true || adoptedProcessID != nil {
                throw DaemonSupervisorError.alreadyRunning
            }
            state = .starting
        }
        onPhase?(.launching)

        let healthClient = MTPLXAPIClient(baseURL: healthBaseURL, apiKey: apiKey)
        if probeHealth, let existing = try? await healthClient.health(), existing.ok {
            if adoptExistingAppOwnedDaemon,
               canAdopt(existing, for: command, requireActualFanRamp: requireActualFanRamp) {
                await adopt(existing)
                onPhase?(.ready)
                return existing
            }
            throw DaemonSupervisorError.portOccupied(
                pid: existing.startup?.pid,
                launchID: existing.startup?.launchId
            )
        }

        let next = Process()
        next.executableURL = command.executableURL
        next.arguments = command.arguments
        next.environment = MTPLXCommandBuilder.appSubprocessEnvironment(
            environment: ProcessInfo.processInfo.environment.merging(command.environment) { _, new in new }
        )

        let stdout = Pipe()
        let stderr = Pipe()
        next.standardOutput = stdout
        next.standardError = stderr
        attach(pipe: stdout, stream: .stdout)
        attach(pipe: stderr, stream: .stderr)
        next.terminationHandler = { [weak self] process in
            guard let self else { return }
            let status = process.terminationStatus
            Task {
                await self.logStore.append("daemon exited with status \(status)", stream: .system)
            }
            self.lock.withLock {
                if self.state != .stopping {
                    self.state = status == 0 ? .stopped : .crashed(status)
                }
                self.process = nil
            }
        }

        do {
            try next.run()
        } catch {
            lock.withLock { state = .stopped }
            throw DaemonSupervisorError.launchFailed(error.localizedDescription)
        }

        lock.withLock {
            process = next
            adoptedProcessID = nil
            state = .warming
        }
        await logStore.append(
            "launched \(command.executableURL.path) \(command.arguments.joined(separator: " "))",
            stream: .system
        )

        let readyHealth: HealthPayload?
        if probeHealth {
            do {
                readyHealth = try await waitForHealth(
                    baseURL: healthBaseURL,
                    apiKey: apiKey,
                    timeoutSeconds: timeoutSeconds,
                    expectedLaunchID: expectedLaunchID,
                    requireActualFanRamp: requireActualFanRamp,
                    onPhase: onPhase
                )
            } catch {
                await stop()
                throw error
            }
        } else {
            readyHealth = nil
        }
        lock.withLock { state = .running }
        onPhase?(.ready)
        return readyHealth
    }

    /// Whether `start(... adoptExistingAppOwnedDaemon: true)` would adopt
    /// this health payload instead of spawning. Exposed so the store's
    /// port pre-flight can distinguish "leave it for adoption" from "move
    /// to a free port".
    public func canAdoptExisting(
        _ health: HealthPayload,
        for command: DaemonCommand,
        requireActualFanRamp: Bool
    ) -> Bool {
        canAdopt(health, for: command, requireActualFanRamp: requireActualFanRamp)
    }

    public func adoptExistingIfAppOwned(
        command: DaemonCommand,
        healthBaseURL: URL,
        apiKey: String? = nil,
        requireActualFanRamp: Bool = false
    ) async throws -> HealthPayload? {
        try lock.withLock {
            if process?.isRunning == true || adoptedProcessID != nil {
                throw DaemonSupervisorError.alreadyRunning
            }
        }
        let client = MTPLXAPIClient(baseURL: healthBaseURL, apiKey: apiKey)
        guard let existing = try? await client.health(), existing.ok else {
            return nil
        }
        guard canAdopt(existing, for: command, requireActualFanRamp: requireActualFanRamp) else {
            return nil
        }
        await adopt(existing)
        return existing
    }

    public func stop(
        graceSeconds: TimeInterval = 2,
        additionalProcessIDs: [pid_t] = []
    ) async {
        let current = lock.withLock { () -> Process? in
            state = .stopping
            return process
        }
        let adopted = lock.withLock { adoptedProcessID }
        var rootPIDs: [pid_t] = []
        if let currentPID = current?.processIdentifier {
            rootPIDs.append(currentPID)
        }
        if let adopted {
            rootPIDs.append(adopted)
        }
        rootPIDs.append(contentsOf: additionalProcessIDs)
        rootPIDs = rootPIDs.filter { $0 > 1 }
        let family = Self.processFamily(rootPIDs: rootPIDs)

        if !family.isEmpty {
            Self.signal(family, SIGTERM)
            await Self.waitUntilExited(family, timeoutSeconds: graceSeconds)
            let afterTerm = family.filter(Self.pidIsAlive)
            if !afterTerm.isEmpty {
                Self.signal(afterTerm, SIGINT)
                await Self.waitUntilExited(afterTerm, timeoutSeconds: 0.5)
            }
            let afterInt = family.filter(Self.pidIsAlive)
            if !afterInt.isEmpty {
                Self.signal(afterInt, SIGKILL)
                await Self.waitUntilExited(afterInt, timeoutSeconds: 1.0)
            }
        }

        lock.withLock {
            process = nil
            adoptedProcessID = nil
            state = .stopped
        }
        let pidList = family.map(String.init).joined(separator: ",")
        await logStore.append(
            pidList.isEmpty ? "daemon stopped" : "daemon process family stopped: \(pidList)",
            stream: .system
        )
    }

    /// Terminate an MTPLX daemon this supervisor does not own (e.g. a
    /// stale app-owned daemon from a previous app session holding the
    /// configured port with a different model). SIGTERM with a grace
    /// window, then SIGKILL, across the whole process family.
    public func terminateExternalDaemon(
        rootPID: pid_t,
        graceSeconds: TimeInterval = 5
    ) async {
        guard rootPID > 1 else { return }
        let family = Self.processFamily(rootPIDs: [rootPID])
        guard !family.isEmpty else { return }
        Self.signal(family, SIGTERM)
        await Self.waitUntilExited(family, timeoutSeconds: graceSeconds)
        let leftovers = family.filter(Self.pidIsAlive)
        if !leftovers.isEmpty {
            Self.signal(leftovers, SIGKILL)
            await Self.waitUntilExited(leftovers, timeoutSeconds: 1.0)
        }
        await logStore.append(
            "terminated stale MTPLX daemon pid \(rootPID)",
            stream: .system
        )
    }

    public func restart(
        command: DaemonCommand,
        healthBaseURL: URL,
        apiKey: String? = nil,
        probeHealth: Bool = true,
        timeoutSeconds: TimeInterval = 300,
        expectedLaunchID: String? = nil,
        requireActualFanRamp: Bool = false,
        adoptExistingAppOwnedDaemon: Bool = false,
        onPhase: (@Sendable (DaemonStartupPhase) -> Void)? = nil
    ) async throws -> HealthPayload? {
        await stop()
        return try await start(
            command: command,
            healthBaseURL: healthBaseURL,
            apiKey: apiKey,
            probeHealth: probeHealth,
            timeoutSeconds: timeoutSeconds,
            expectedLaunchID: expectedLaunchID,
            requireActualFanRamp: requireActualFanRamp,
            adoptExistingAppOwnedDaemon: adoptExistingAppOwnedDaemon,
            onPhase: onPhase
        )
    }

    public func waitForExistingHealth(
        healthBaseURL: URL,
        apiKey: String? = nil,
        timeoutSeconds: TimeInterval = 300,
        expectedLaunchID: String? = nil,
        requireActualFanRamp: Bool = false,
        onPhase: (@Sendable (DaemonStartupPhase) -> Void)? = nil
    ) async throws -> HealthPayload {
        let health = try await waitForHealth(
            baseURL: healthBaseURL,
            apiKey: apiKey,
            timeoutSeconds: timeoutSeconds,
            expectedLaunchID: expectedLaunchID,
            requireActualFanRamp: requireActualFanRamp,
            onPhase: onPhase
        )
        lock.withLock { state = .running }
        onPhase?(.ready)
        return health
    }

    private func canAdopt(
        _ health: HealthPayload,
        for command: DaemonCommand,
        requireActualFanRamp: Bool
    ) -> Bool {
        guard health.startup?.launchId?.isEmpty == false,
              health.startup?.pid != nil
        else {
            return false
        }
        if requireActualFanRamp,
           health.thermal?.actualRampVerified != true {
            return false
        }
        if let expectedModel = expectedModelPath(from: command.arguments),
           standardizePath(health.modelPath) != standardizePath(expectedModel) {
            return false
        }
        return true
    }

    private func adopt(_ health: HealthPayload) async {
        let pid = health.startup?.pid.map(pid_t.init)
        lock.withLock {
            adoptedProcessID = pid
            state = .running
        }
        await logStore.append(
            "adopted existing app-owned MTPLX daemon pid \(health.startup?.pid.map(String.init) ?? "unknown") launch \(health.startup?.launchId ?? "unknown")",
            stream: .system
        )
    }

    private func expectedModelPath(from arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: "--model"),
              arguments.indices.contains(arguments.index(after: index))
        else {
            return nil
        }
        return arguments[arguments.index(after: index)]
    }

    private func standardizePath(_ path: String) -> String {
        NSString(string: path).standardizingPath
    }

    private static func pidIsAlive(_ pid: pid_t) -> Bool {
        if kill(pid, 0) == 0 {
            return true
        }
        return errno != ESRCH
    }

    private static func processFamily(rootPIDs: [pid_t]) -> [pid_t] {
        var seen: Set<pid_t> = []
        var ordered: [pid_t] = []
        var queue = rootPIDs
        while !queue.isEmpty {
            let pid = queue.removeFirst()
            guard pid > 1, !seen.contains(pid) else { continue }
            seen.insert(pid)
            ordered.append(pid)
            queue.append(contentsOf: childPIDs(of: pid))
        }
        return ordered.reversed()
    }

    private static func childPIDs(of pid: pid_t) -> [pid_t] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        process.arguments = ["-P", String(pid)]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        // Bounded wait + lossless drain (#158): the PID list feeds the
        // reap path, and a wedged pgrep must degrade to "no children
        // found" (roots still get signalled) instead of hanging a
        // daemon stop/restart.
        let watchdog = SubprocessWatchdog(process)
        do {
            try process.run()
        } catch {
            return []
        }
        let drain = SubprocessPipeDrain(output)
        guard watchdog.wait(for: process, timeout: 10) else { return [] }
        guard process.terminationStatus == 0 else { return [] }
        drain.join()
        let text = drain.snapshot()
        return text
            .split(whereSeparator: \.isNewline)
            .compactMap { pid_t(String($0.trimmingCharacters(in: .whitespacesAndNewlines))) }
            .filter { $0 > 1 }
    }

    private static func signal(_ pids: [pid_t], _ signum: Int32) {
        for pid in pids where pid > 1 && pidIsAlive(pid) {
            kill(pid, signum)
        }
    }

    private static func waitUntilExited(
        _ pids: [pid_t],
        timeoutSeconds: TimeInterval
    ) async {
        let deadline = Date().addingTimeInterval(max(0, timeoutSeconds))
        while Date() < deadline {
            if !pids.contains(where: pidIsAlive) {
                return
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
    }

    private func waitForHealth(
        baseURL: URL,
        apiKey: String?,
        timeoutSeconds: TimeInterval,
        expectedLaunchID: String?,
        requireActualFanRamp: Bool,
        onPhase: (@Sendable (DaemonStartupPhase) -> Void)?
    ) async throws -> HealthPayload {
        let client = MTPLXAPIClient(baseURL: baseURL, apiKey: apiKey)
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        var sawHealthyWithUnverifiedFan = false
        onPhase?(.waitingForOwnedHealth)
        while Date() < deadline {
            if !isRunning() {
                let tail = await logStore.snapshot().suffix(8).map(\.message).joined(separator: " | ")
                let detail = tail.isEmpty
                    ? "daemon exited before /health became ready"
                    : "daemon exited before /health became ready: \(tail)"
                throw DaemonSupervisorError.launchFailed(detail)
            }
            if let health = try? await client.health(), health.ok {
                if let expectedLaunchID {
                    guard health.startup?.launchId == expectedLaunchID else {
                        throw DaemonSupervisorError.launchIdentityMismatch(
                            expected: expectedLaunchID,
                            observed: health.startup?.launchId
                        )
                    }
                }
                if requireActualFanRamp,
                   health.thermal?.actualRampVerified != true {
                    sawHealthyWithUnverifiedFan = true
                    onPhase?(.rampingFans)
                    try? await Task.sleep(nanoseconds: 250_000_000)
                    continue
                }
                onPhase?(.warming)
                return health
            }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
        if requireActualFanRamp && sawHealthyWithUnverifiedFan {
            throw DaemonSupervisorError.fanRampTimeout
        }
        throw DaemonSupervisorError.healthTimeout
    }

    private func attach(pipe: Pipe, stream: LogEntry.Stream) {
        pipe.fileHandleForReading.readabilityHandler = { [logStore] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
                Task {
                    await logStore.append(String(line), stream: stream)
                }
            }
        }
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
