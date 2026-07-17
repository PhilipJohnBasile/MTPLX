import Foundation

// MARK: - FanControlSetupResult

public struct FanControlSetupResult: Equatable, Sendable {
    public var ok: Bool
    public var exitCode: Int32?
    public var message: String

    public init(ok: Bool, exitCode: Int32?, message: String) {
        self.ok = ok
        self.exitCode = exitCode
        self.message = message
    }
}

private struct FanControlCommandResult: Sendable {
    var ok: Bool
    var exitCode: Int32?
    var stdout: String
    var stderr: String
    var message: String
}

// MARK: - SubprocessInterruptBox
//
// Tracks the currently running child process so a cancellation from
// the consumer (stream termination, onboarding step retry) can
// interrupt it. Previously AutoTuner's private `TuneSubprocessBox`;
// shared so the runtime-setup step can cancel fan-control installs
// the same way the tuner does.

final class SubprocessInterruptBox: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?

    func set(_ process: Process) {
        lock.lock()
        self.process = process
        lock.unlock()
    }

    func clear(_ process: Process) {
        lock.lock()
        if self.process === process {
            self.process = nil
        }
        lock.unlock()
    }

    func interrupt() {
        lock.lock()
        let current = process
        lock.unlock()
        guard let current, current.isRunning else { return }
        // Escalating cancel (#158 sweep): SIGINT first so the child
        // can clean up, then terminate → SIGKILL if it ignores both —
        // a cancelled tune or fan install must never linger as an
        // invisible worker.
        SubprocessWatchdog.escalateCancel(current)
    }
}

// MARK: - FanControlInstaller
//
// `mtplx tune` requires a CLI-visible fan-pinning helper for honest
// measurements: `~/.mtplx/bin/thermalforge` or a thermalforge/tgpro
// binary on PATH. The app bundles ThermalForge as a resource and
// `mtplx max --install --json` copies it into place. This is shared
// by the onboarding runtime-setup step (installs it up front) and the
// auto-tuner (verifies/repairs right before measuring).

struct FanControlInstaller: Sendable {
    let processEnvironment: [String: String]

    init(processEnvironment: [String: String] = ProcessInfo.processInfo.environment) {
        self.processEnvironment = processEnvironment
    }

    /// The CLI needs `~/.mtplx/bin/thermalforge` or an executable on
    /// the (expanded) PATH; the ThermalForge app bundle alone is not
    /// enough.
    func helperPresent() -> Bool {
        let home = processEnvironment["HOME"] ?? NSHomeDirectory()
        let privateThermalForge = URL(fileURLWithPath: home)
            .appendingPathComponent(".mtplx")
            .appendingPathComponent("bin")
            .appendingPathComponent("thermalforge")
            .path
        if FileManager.default.isExecutableFile(atPath: privateThermalForge) {
            return true
        }
        return executableOnExpandedPath("thermalforge")
            || executableOnExpandedPath("tgpro")
            || executableOnExpandedPath("tgpro-cli")
    }

    /// Check for a detected fan tool and install one when missing.
    /// Never throws — callers decide whether a failure blocks (tune)
    /// or degrades to a warning (runtime setup).
    func ensureReady(
        executable: URL,
        subprocess: SubprocessInterruptBox,
        status: (String) -> Void
    ) -> FanControlSetupResult {
        status("Checking fan control")

        let statusCheck = runCommand(
            executable: executable,
            arguments: ["max", "--status", "--json"],
            subprocess: subprocess
        )
        let hasDetectedTool = statusCheck.ok && Self.payloadBool(
            statusCheck.stdout,
            path: ["detection", "available"]
        ) == true

        if !hasDetectedTool {
            status("Installing fan control")
            let install = install(executable: executable, subprocess: subprocess)
            guard install.ok else { return install }
        }

        status("Fan control ready")
        return FanControlSetupResult(ok: true, exitCode: 0, message: "Fan control ready")
    }

    private func install(
        executable: URL,
        subprocess: SubprocessInterruptBox
    ) -> FanControlSetupResult {
        let result = runCommand(
            executable: executable,
            arguments: ["max", "--install", "--json"],
            subprocess: subprocess
        )
        if result.exitCode == nil {
            return FanControlSetupResult(ok: false, exitCode: nil, message: result.message)
        }
        if result.exitCode == 0, helperPresent() {
            return FanControlSetupResult(ok: true, exitCode: 0, message: "Fan control ready")
        }
        return FanControlSetupResult(
            ok: false,
            exitCode: result.exitCode,
            message: Self.installFailureMessage(stdout: result.stdout, stderr: result.stderr)
        )
    }

    private func runCommand(
        executable: URL,
        arguments: [String],
        subprocess: SubprocessInterruptBox
    ) -> FanControlCommandResult {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.environment = MTPLXCommandBuilder.appSubprocessEnvironment(
            environment: processEnvironment
        )

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        let watchdog = SubprocessWatchdog(process)
        do {
            subprocess.set(process)
            try process.run()
        } catch {
            subprocess.clear(process)
            return FanControlCommandResult(
                ok: false,
                exitCode: nil,
                stdout: "",
                stderr: "",
                message: error.localizedDescription
            )
        }
        // `mtplx max --status/--install` answers with a JSON payload
        // on stdout that gets parsed, so both pipes use the lossless
        // drain, and the wait is bounded (#158): a wedged CLI must
        // fail the fan step after one timeout window, not park
        // onboarding or a tune forever. 120s covers a cold first exec
        // (Gatekeeper scan + Python start) plus the helper copy.
        let stdoutDrain = SubprocessPipeDrain(stdout)
        let stderrDrain = SubprocessPipeDrain(stderr)
        let timeout: TimeInterval = 120
        guard watchdog.wait(for: process, timeout: timeout) else {
            subprocess.clear(process)
            stdoutDrain.join(timeout: 1)
            stderrDrain.join(timeout: 1)
            let commandLine = "\(executable.lastPathComponent) \(arguments.joined(separator: " "))"
            return FanControlCommandResult(
                ok: false,
                exitCode: -2,
                stdout: stdoutDrain.snapshot(),
                // The timeout note rides on stderr so
                // installFailureMessage() surfaces it too.
                stderr: stderrDrain.snapshot()
                    + "\n[\(commandLine) timed out after \(Int(timeout))s and was terminated]",
                message: "\(commandLine) timed out after \(Int(timeout))s and was terminated"
            )
        }
        subprocess.clear(process)
        stdoutDrain.join()
        stderrDrain.join()

        let stdoutText = stdoutDrain.snapshot()
        let stderrText = stderrDrain.snapshot()
        let payloadOK = Self.payloadBool(stdoutText, path: ["ok"])
        let ok = process.terminationStatus == 0 && payloadOK != false
        return FanControlCommandResult(
            ok: ok,
            exitCode: process.terminationStatus,
            stdout: stdoutText,
            stderr: stderrText,
            message: Self.commandMessage(
                stdout: stdoutText,
                stderr: stderrText,
                fallback: "\(executable.lastPathComponent) \(arguments.joined(separator: " ")) failed"
            )
        )
    }

    private func executableOnExpandedPath(_ name: String) -> Bool {
        MTPLXCommandBuilder.expandedPATH(environment: processEnvironment)
            .split(separator: ":")
            .map(String.init)
            .contains { directory in
                FileManager.default.isExecutableFile(
                    atPath: URL(fileURLWithPath: directory)
                        .appendingPathComponent(name)
                        .path
                )
            }
    }

    // MARK: Message extraction

    private static func installFailureMessage(stdout: String, stderr: String) -> String {
        let trimmedStdout = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        if let data = trimmedStdout.data(using: .utf8),
           let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let message = root["message"] as? String,
           !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return message
        }
        let trimmedStderr = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedStderr.isEmpty { return trimmedStderr }
        if !trimmedStdout.isEmpty { return trimmedStdout }
        return "Fan-control install failed."
    }

    private static func commandMessage(
        stdout: String,
        stderr: String,
        fallback: String
    ) -> String {
        let trimmedStdout = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        if let data = trimmedStdout.data(using: .utf8),
           let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            for key in ["message", "error", "actionable"] {
                if let message = root[key] as? String,
                   !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                {
                    return message
                }
            }
            if let attempts = root["attempts"] as? [[String: Any]],
               let last = attempts.last,
               let stderr = last["stderr"] as? String,
               !stderr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                return stderr
            }
        }
        let trimmedStderr = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedStderr.isEmpty { return trimmedStderr }
        if !trimmedStdout.isEmpty { return trimmedStdout }
        return fallback
    }

    private static func payloadBool(_ stdout: String, path: [String]) -> Bool? {
        let trimmed = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        var cursor: Any = root
        for key in path {
            guard let dict = cursor as? [String: Any],
                  let next = dict[key]
            else { return nil }
            cursor = next
        }
        return cursor as? Bool
    }
}

// The private FanControlTailBuffer that lived here moved to the shared
// SubprocessPipeDrain (SubprocessSupport.swift) when runCommand gained
// its deadline watchdog — the parsed JSON payload needs the drain's
// lossless EOF join, not a racy readabilityHandler tail.
