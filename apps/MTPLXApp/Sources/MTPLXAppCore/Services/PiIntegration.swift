import Darwin
import Foundation

public struct PiConfigResult: Equatable, Sendable {
    public let configPath: String
    public let baseURL: String
    public let modelReference: String
    public let launchCommand: String
    public let didChange: Bool
    public let backupPath: String?
}

public enum PiLaunchAction: String, Equatable, Sendable {
    case launched
    case unavailable
}

public struct PiLaunchResult: Equatable, Sendable {
    public let action: PiLaunchAction
    public let command: String
    public let detail: String
    public let launchedProcessIDs: [Int]
    /// Exact ownership of the Terminal-launched Pi process when it reports
    /// its UUID receipt. Callers must use this lease, not process discovery,
    /// for stale cleanup.
    public let terminalHandoffLease: MTPLXTerminalHandoffLease?

    public init(
        action: PiLaunchAction,
        command: String,
        detail: String,
        launchedProcessIDs: [Int] = [],
        terminalHandoffLease: MTPLXTerminalHandoffLease? = nil
    ) {
        self.action = action
        self.command = command
        self.detail = detail
        self.launchedProcessIDs = launchedProcessIDs
        self.terminalHandoffLease = terminalHandoffLease
    }
}

public struct PiIntegration: Sendable {
    public static let providerID = "mtplx"
    public static let localAPIKey = "mtplx-local"
    public static let codingTools = "read,bash,edit,write,grep,find,ls"
    public static let agentOperatingHintsFilename = "pi-agent-operating-hints.md"
    public static let agentOperatingHints = """
    MTPLX agent operating hints:
    - Treat tool calls and long context as expensive user-visible latency. Prefer grep/find/ls first, then read only the exact line ranges needed for the next decision.
    - Do not re-read a file or expand adjacent ranges just to be complete. If you have enough evidence, choose the safest useful change and implement it.
    - For broad project tasks, converge after roughly 10 to 14 tool calls: name the best candidate, edit one focused area, then run the relevant build, typecheck, or smoke check.
    - Keep final answers concise and evidence-based. Mention the files changed and checks run; avoid long inventory summaries.
    - If a shell command appears stuck, stop waiting, explain the command, and choose a narrower verification path.
    """

    public let configURL: URL
    /// Per-invocation scripts, receipts, and cancellation markers live here.
    /// Keeping the directory injectable makes ownership deterministic in
    /// service tests without reusing a fixed command filename.
    public let handoffDirectory: URL

    public init(
        configURL: URL = PiIntegration.defaultConfigURL(),
        handoffDirectory: URL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent("handoffs", isDirectory: true)
    ) {
        self.configURL = configURL
        self.handoffDirectory = handoffDirectory
    }

    public static func defaultConfigURL() -> URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".pi")
            .appendingPathComponent("agent")
            .appendingPathComponent("models.json")
    }

    public static func modelID(for model: String) -> String {
        OpenCodeIntegration.modelID(for: model)
    }

    public static func modelReference(for model: String) -> String {
        "\(providerID)/\(modelID(for: model))"
    }

    public static func launchCommand(for model: String) -> String {
        "pi --model \(modelReference(for: model)) --tools \(codingTools) "
            + "--append-system-prompt \(shellQuote(agentOperatingHintsURL().path))"
    }

    public static func agentOperatingHintsURL(
        homeDirectory: String = NSHomeDirectory()
    ) -> URL {
        URL(fileURLWithPath: homeDirectory)
            .appendingPathComponent(".mtplx")
            .appendingPathComponent(agentOperatingHintsFilename)
    }

    @MainActor
    public func launchInTerminal(
        configuration: MTPLXAppConfiguration,
        isCurrent: (() -> Bool)? = nil
    ) async -> PiLaunchResult {
        let command = Self.terminalLaunchCommand(for: configuration.model)
        guard isCurrent?() ?? true else {
            return staleHandoffResult(command: command)
        }
        #if os(macOS)
        let handoff = makeTerminalHandoffFiles()
        do {
            guard isCurrent?() ?? true else { return staleHandoffResult(command: command) }
            try writeTerminalCommandFile(
                command: command,
                configuration: configuration,
                handoff: handoff
            )
        } catch {
            return PiLaunchResult(
                action: .unavailable,
                command: command,
                detail: "could not prepare Pi terminal command: \(error)"
            )
        }
        guard isCurrent?() ?? true else {
            await cancelPendingTerminalHandoff(handoff)
            return staleHandoffResult(command: command)
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = ["-a", "Terminal", handoff.commandURL.path]
        let stderr = Pipe()
        process.standardError = stderr
        // The backend store calls this from the main actor, so a wedged
        // LaunchServices `open` must cost one bounded window, never
        // beachball the app (#158 pattern).
        let stderrTail = SubprocessTailBuffer(capacity: 4096)
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if !chunk.isEmpty { stderrTail.append(chunk) }
        }
        defer { stderr.fileHandleForReading.readabilityHandler = nil }
        let watchdog = SubprocessWatchdog(process)
        do {
            guard isCurrent?() ?? true else {
                await cancelPendingTerminalHandoff(handoff)
                return staleHandoffResult(command: command)
            }
            try process.run()
            guard watchdog.wait(for: process, timeout: 30) else {
                await cancelPendingTerminalHandoff(handoff)
                return PiLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: "could not open Pi automatically: open timed out after 30s and was terminated"
                )
            }
            guard process.terminationStatus == 0 else {
                await cancelPendingTerminalHandoff(handoff)
                let message = stderrTail.snapshot()
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return PiLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: message.isEmpty
                        ? "could not open Pi automatically: open exited \(process.terminationStatus)"
                        : "could not open Pi automatically: \(message)"
                )
            }
            let receipt = await MTPLXTerminalHandoffLease.awaitReceipt(
                handoffID: handoff.handoffID,
                receiptURL: handoff.receiptURL,
                cancellationMarkerURL: handoff.cancellationMarkerURL,
                commandURL: handoff.commandURL,
                isCurrent: isCurrent
            )
            if receipt.cancellationRequested {
                if let lease = receipt.lease {
                    _ = cancelTerminalHandoff(lease)
                }
                let stale = !(isCurrent?() ?? true)
                return PiLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: stale
                        ? "Pi handoff cancelled because the daemon lifecycle changed."
                        : "Pi Terminal did not report its launch receipt."
                )
            }
            guard let lease = receipt.lease else {
                return PiLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: "Pi Terminal did not report its launch receipt."
                )
            }
            guard isCurrent?() ?? true else {
                _ = cancelTerminalHandoff(lease)
                return PiLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: "Pi handoff cancelled because the daemon lifecycle changed.",
                    launchedProcessIDs: [lease.processID],
                    terminalHandoffLease: lease
                )
            }
            return PiLaunchResult(
                action: .launched,
                command: command,
                detail: "opened Pi in Terminal",
                launchedProcessIDs: [lease.processID],
                terminalHandoffLease: lease
            )
        } catch {
            await cancelPendingTerminalHandoff(handoff)
            return PiLaunchResult(
                action: .unavailable,
                command: command,
                detail: "could not open Pi automatically: \(error)"
            )
        }
        #else
        return PiLaunchResult(
            action: .unavailable,
            command: command,
            detail: "automatic Pi launch currently requires macOS Terminal"
        )
        #endif
    }

    private func staleHandoffResult(command: String) -> PiLaunchResult {
        PiLaunchResult(
            action: .unavailable,
            command: command,
            detail: "Pi handoff cancelled because the daemon lifecycle changed."
        )
    }

    /// Covers every failure after the script is durable. A late Terminal
    /// launch sees the marker; a receipt from the final-check race is reaped
    /// only after proving the exact UUID environment token.
    @MainActor
    private func cancelPendingTerminalHandoff(_ handoff: TerminalHandoffFiles) async {
        let receipt = await MTPLXTerminalHandoffLease.awaitReceipt(
            handoffID: handoff.handoffID,
            receiptURL: handoff.receiptURL,
            cancellationMarkerURL: handoff.cancellationMarkerURL,
            commandURL: handoff.commandURL,
            isCurrent: { false },
            timeoutSeconds: 0,
            delayedCancellationSeconds: 1
        )
        if let lease = receipt.lease {
            _ = cancelTerminalHandoff(lease)
        }
    }

    /// Cancels one exact lease. A stale Store callback must use this method
    /// rather than process discovery or a raw PID list.
    @MainActor
    @discardableResult
    public func cancelTerminalHandoff(_ lease: MTPLXTerminalHandoffLease) -> Bool {
        let cancellationMarked = MTPLXTerminalHandoffLease.writeCancellationMarker(
            at: lease.cancellationMarkerURL
        )
        let commandRemoved = lease.commandURL.map {
            MTPLXTerminalHandoffLease.removeDurableCommandScript(at: $0)
        } ?? true
        let receiptRemoved = lease.receiptURL.map {
            MTPLXTerminalHandoffLease.removeDurableCommandScript(at: $0)
        } ?? true
        let pid = pid_t(lease.processID)
        guard pid > 1,
              MTPLXTerminalHandoffLease.process(pid: pid, hasExactHandoffID: lease.handoffID)
        else { return false }
        Self.terminate(pid: pid)
        return cancellationMarked && commandRemoved && receiptRemoved
    }

    @discardableResult
    public func sync(configuration: MTPLXAppConfiguration) throws -> PiConfigResult {
        let modelID = Self.modelID(for: configuration.model)
        let modelReference = "\(Self.providerID)/\(modelID)"
        let baseURL = OpenCodeIntegration.baseURLString(
            host: configuration.host,
            port: configuration.port
        )
        let apiKey = configuration.apiKey?.isEmpty == false
            ? configuration.apiKey!
            : Self.localAPIKey
        let contextWindow = configuration.effectiveContextWindow(default: 131_072)
        var backupURL: URL?

        var root = try loadRoot()
        var providers = root["providers"]?.objectValue ?? [:]
        providers[Self.providerID] = .object(
            Self.providerConfig(
                modelID: modelID,
                baseURL: baseURL,
                apiKey: apiKey,
                contextWindow: contextWindow,
                reasoningEnabled: OpenCodeIntegration.reasoningEnabled(forModelID: modelID),
                reasoningEffort: OpenCodeIntegration.reasoningEffort(forModelID: modelID)
            )
        )
        root["providers"] = .object(providers)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let nextData = try encoder.encode(root)

        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: configURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let existingData = try? Data(contentsOf: configURL)
        if existingData == nextData {
            try? fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configURL.path)
            return PiConfigResult(
                configPath: configURL.path,
                baseURL: baseURL,
                modelReference: modelReference,
                launchCommand: Self.launchCommand(for: configuration.model),
                didChange: false,
                backupPath: nil
            )
        }

        if existingData != nil {
            let backup = uniqueBackupURL(reason: "bak")
            try fileManager.copyItem(at: configURL, to: backup)
            backupURL = backup
        }
        try nextData.write(to: configURL, options: [.atomic])
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configURL.path)

        return PiConfigResult(
            configPath: configURL.path,
            baseURL: baseURL,
            modelReference: modelReference,
            launchCommand: Self.launchCommand(for: configuration.model),
            didChange: true,
            backupPath: backupURL?.path
        )
    }

    private static func providerConfig(
        modelID: String,
        baseURL: String,
        apiKey: String,
        contextWindow: Int,
        reasoningEnabled: Bool,
        reasoningEffort: String?
    ) -> [String: JSONValue] {
        var compat: [String: JSONValue] = [
            "supportsDeveloperRole": .bool(false),
            "supportsReasoningEffort": .bool(reasoningEffort != nil),
            "maxTokensField": .string("max_tokens"),
        ]
        if let reasoningEffort {
            compat["reasoningEffort"] = .string(reasoningEffort)
        }

        return [
            "baseUrl": .string(baseURL),
            "api": .string("openai-completions"),
            "apiKey": .string(apiKey),
            "authHeader": .bool(true),
            "headers": .object([
                "x-mtplx-client": .string("pi"),
            ]),
            "compat": .object(compat),
            "models": .array([
                .object([
                    "id": .string(modelID),
                    "name": .string("MTPLX \(modelID)"),
                    "reasoning": .bool(reasoningEnabled),
                    "input": .array([.string("text")]),
                    "contextWindow": .number(Double(contextWindow)),
                    "cost": .object([
                        "input": .number(0),
                        "output": .number(0),
                        "cacheRead": .number(0),
                        "cacheWrite": .number(0),
                    ]),
                ]),
            ]),
        ]
    }

    private func loadRoot() throws -> [String: JSONValue] {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: configURL.path) else {
            return [:]
        }
        let data = try Data(contentsOf: configURL)
        guard !data.isEmpty else {
            return [:]
        }
        do {
            return try JSONDecoder().decode([String: JSONValue].self, from: data)
        } catch {
            let backup = uniqueBackupURL(reason: "invalid")
            try fileManager.moveItem(at: configURL, to: backup)
            return [:]
        }
    }

    private func uniqueBackupURL(reason: String) -> URL {
        let directory = configURL.deletingLastPathComponent()
        let timestamp = Self.timestamp()
        let basename = configURL.lastPathComponent
        var candidate = directory.appendingPathComponent("\(basename).\(reason)-\(timestamp).bak")
        var index = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = directory.appendingPathComponent(
                "\(basename).\(reason)-\(timestamp)-\(index).bak"
            )
            index += 1
        }
        return candidate
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }

    private static func terminalLaunchCommand(for model: String) -> String {
        "\(shellQuote(piExecutable())) --model \(shellQuote(modelReference(for: model))) "
            + "--tools \(shellQuote(codingTools)) --append-system-prompt "
            + "\(shellQuote(agentOperatingHintsURL().path))"
    }

    private static func piExecutable() -> String {
        let home = NSHomeDirectory()
        for path in [
            "/opt/homebrew/bin/pi",
            "/usr/local/bin/pi",
            "\(home)/.local/bin/pi",
            "\(home)/.npm-global/bin/pi",
        ] where FileManager.default.isExecutableFile(atPath: path) {
            return path
        }
        return "pi"
    }

    private static func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }

    public static func resolvedWorkspacePath(
        configuration: MTPLXAppConfiguration,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        let configured = MTPLXAppConfiguration.normalizedHermesWorkspacePath(
            configuration.hermesWorkspacePath
        )
        if isDirectory(configured) {
            return configured
        }

        let fallback = MTPLXAppConfiguration.defaultHermesWorkspacePath()
        if isDirectory(fallback) {
            return fallback
        }

        return environment["HOME"] ?? NSHomeDirectory()
    }

    private static func isDirectory(_ path: String) -> Bool {
        var isDirectory = ObjCBool(false)
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }

    private struct TerminalHandoffFiles: Sendable {
        let handoffID: UUID
        let commandURL: URL
        let receiptURL: URL
        let cancellationMarkerURL: URL
    }

    private func makeTerminalHandoffFiles() -> TerminalHandoffFiles {
        let handoffID = UUID()
        let suffix = handoffID.uuidString.lowercased()
        return TerminalHandoffFiles(
            handoffID: handoffID,
            commandURL: handoffDirectory.appendingPathComponent("open-pi-\(suffix).command"),
            receiptURL: handoffDirectory.appendingPathComponent("open-pi-\(suffix).pid"),
            cancellationMarkerURL: handoffDirectory.appendingPathComponent("open-pi-\(suffix).cancelled")
        )
    }

    static func isPiAgentCommand(_ command: String) -> Bool {
        let words = commandWords(command)
        guard let first = words.first else { return false }

        if isPiExecutableToken(first) {
            return true
        }

        guard isNodeExecutableToken(first), words.count >= 2 else {
            return false
        }

        let script = words[1]
        let hasPiScript = isPiExecutableToken(script) || isPiAgentScriptToken(script)
        guard hasPiScript else { return false }

        return hasPiLaunchIntent(words)
    }

    private static func terminate(pid: pid_t) {
        guard kill(pid, 0) == 0 else { return }
        _ = kill(pid, SIGTERM)
        for _ in 0..<20 {
            if kill(pid, 0) != 0 { return }
            Thread.sleep(forTimeInterval: 0.05)
        }
        _ = kill(pid, SIGKILL)
    }

    private static func commandWords(_ command: String) -> [String] {
        command
            .split(whereSeparator: { $0.isWhitespace })
            .map { stripShellTokenQuotes(String($0)) }
            .filter { !$0.isEmpty }
    }

    private static func stripShellTokenQuotes(_ token: String) -> String {
        var value = token.trimmingCharacters(in: .whitespacesAndNewlines)
        while value.count >= 2,
              let first = value.first,
              let last = value.last,
              (first == "'" || first == "\""),
              first == last {
            value.removeFirst()
            value.removeLast()
        }
        return value
    }

    private static func isPiExecutableToken(_ token: String) -> Bool {
        URL(fileURLWithPath: token).lastPathComponent == "pi"
    }

    private static func isNodeExecutableToken(_ token: String) -> Bool {
        URL(fileURLWithPath: token).lastPathComponent == "node"
    }

    private static func isPiAgentScriptToken(_ token: String) -> Bool {
        let normalized = token.replacingOccurrences(of: "\\", with: "/")
        return normalized.contains("@earendil-works/pi-coding-agent/")
            && URL(fileURLWithPath: normalized).lastPathComponent == "cli.js"
    }

    private static func hasPiLaunchIntent(_ words: [String]) -> Bool {
        let normalized = Set(words.map { $0.lowercased() })
        if normalized.contains("--tools") {
            return true
        }
        if normalized.contains("--model"),
           words.contains(where: { $0.lowercased().hasPrefix("\(providerID)/") }) {
            return true
        }
        return false
    }

    private func writeTerminalCommandFile(
        command: String,
        configuration: MTPLXAppConfiguration,
        handoff: TerminalHandoffFiles
    ) throws {
        let directory = handoff.commandURL.deletingLastPathComponent()
        try MTPLXTerminalHandoffLease.prepareArtifactDirectory(directory)
        _ = try Self.writeAgentOperatingHintsFile()
        let workspacePath = Self.resolvedWorkspacePath(configuration: configuration)
        let script = """
        #!/bin/zsh
        _mtplx_handoff_cancel=\(Self.shellQuote(handoff.cancellationMarkerURL.path))
        _mtplx_handoff_receipt=\(Self.shellQuote(handoff.receiptURL.path))
        export MTPLX_APP_HANDOFF_ID=\(Self.shellQuote(handoff.handoffID.uuidString.lowercased()))
        if [[ -e "$_mtplx_handoff_cancel" ]]; then
          exit 0
        fi
        cd \(Self.shellQuote(workspacePath))
        if [[ -e "$_mtplx_handoff_cancel" ]]; then
          exit 0
        fi
        umask 077
        print -r -- "$$" > "${_mtplx_handoff_receipt}.$$.tmp"
        mv -f "${_mtplx_handoff_receipt}.$$.tmp" "$_mtplx_handoff_receipt"
        if [[ -e "$_mtplx_handoff_cancel" ]]; then
          exit 0
        fi
        exec \(command)
        """
        try MTPLXTerminalHandoffLease.writeSecureCommandScript(script, to: handoff.commandURL)
    }

    @discardableResult
    private static func writeAgentOperatingHintsFile(
        homeDirectory: String = NSHomeDirectory()
    ) throws -> URL {
        let url = agentOperatingHintsURL(homeDirectory: homeDirectory)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = Data(agentOperatingHints.utf8)
        if (try? Data(contentsOf: url)) != data {
            try data.write(to: url, options: [.atomic])
        }
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600],
            ofItemAtPath: url.path
        )
        return url
    }
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { value } else { nil }
    }
}
