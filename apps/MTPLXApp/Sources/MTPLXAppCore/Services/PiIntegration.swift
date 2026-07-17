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

    public init(
        action: PiLaunchAction,
        command: String,
        detail: String,
        launchedProcessIDs: [Int] = []
    ) {
        self.action = action
        self.command = command
        self.detail = detail
        self.launchedProcessIDs = launchedProcessIDs
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

    public init(configURL: URL = PiIntegration.defaultConfigURL()) {
        self.configURL = configURL
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

    public func launchInTerminal(configuration: MTPLXAppConfiguration) -> PiLaunchResult {
        let command = Self.terminalLaunchCommand(for: configuration.model)
        #if os(macOS)
        let existingAgentPIDs = Self.runningPiAgentPIDs()
        let scriptURL: URL
        do {
            scriptURL = try writeTerminalCommandFile(
                command: command,
                configuration: configuration
            )
        } catch {
            return PiLaunchResult(
                action: .unavailable,
                command: command,
                detail: "could not prepare Pi terminal command: \(error)"
            )
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = ["-a", "Terminal", scriptURL.path]
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
            try process.run()
            guard watchdog.wait(for: process, timeout: 30) else {
                return PiLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: "could not open Pi automatically: open timed out after 30s and was terminated"
                )
            }
            guard process.terminationStatus == 0 else {
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
            let launchedPIDs = Self.waitForNewPiAgentPIDs(excluding: existingAgentPIDs)
            return PiLaunchResult(
                action: .launched,
                command: command,
                detail: "opened Pi in Terminal",
                launchedProcessIDs: launchedPIDs.map(Int.init).sorted()
            )
        } catch {
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

    @discardableResult
    public func stopLaunchedAgents(processIDs: [Int]) -> Int {
        var stopped = 0
        for processID in Set(processIDs) {
            let pid = pid_t(processID)
            guard pid > 1, Self.isPiAgentProcess(pid: pid) else { continue }
            Self.terminate(pid: pid)
            stopped += 1
        }
        return stopped
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

    private static func waitForNewPiAgentPIDs(excluding existing: Set<pid_t>) -> [pid_t] {
        let deadline = Date().addingTimeInterval(3)
        while Date() < deadline {
            let next = runningPiAgentPIDs().subtracting(existing)
            if !next.isEmpty {
                return Array(next)
            }
            Thread.sleep(forTimeInterval: 0.1)
        }
        return []
    }

    private static func isPiAgentProcess(pid: pid_t) -> Bool {
        runningPiAgentPIDs().contains(pid)
    }

    private static func runningPiAgentPIDs() -> Set<pid_t> {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-axo", "pid=,command="]
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-pi-ps-\(UUID().uuidString).txt")
        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        guard let outputHandle = try? FileHandle(forWritingTo: outputURL) else { return [] }
        defer {
            try? outputHandle.close()
            try? FileManager.default.removeItem(at: outputURL)
        }
        process.standardOutput = outputHandle
        let exitDone = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in
            exitDone.signal()
        }
        do {
            try process.run()
        } catch {
            return []
        }
        if exitDone.wait(timeout: .now() + 1.0) == .timedOut {
            process.terminate()
            if exitDone.wait(timeout: .now() + 0.5) == .timedOut {
                return []
            }
        }
        try? outputHandle.synchronize()
        let outputData = (try? Data(contentsOf: outputURL)) ?? Data()
        guard let output = String(data: outputData, encoding: .utf8) else { return [] }
        let currentPID = getpid()
        return Set(output.split(separator: "\n").compactMap { row -> pid_t? in
            let text = String(row).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let firstSpace = text.firstIndex(where: { $0.isWhitespace }) else {
                return nil
            }
            guard let pid = pid_t(text[..<firstSpace]),
                  pid > 1,
                  pid != currentPID
            else {
                return nil
            }
            let command = String(text[firstSpace...])
            return isPiAgentCommand(command) ? pid : nil
        })
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
        configuration: MTPLXAppConfiguration
    ) throws -> URL {
        let directory = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".mtplx")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let scriptURL = directory.appendingPathComponent("open-pi.command")
        _ = try Self.writeAgentOperatingHintsFile()
        let workspacePath = Self.resolvedWorkspacePath(configuration: configuration)
        let script = """
        #!/bin/zsh
        cd \(Self.shellQuote(workspacePath))
        exec \(command)
        """
        try script.write(to: scriptURL, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: scriptURL.path
        )
        return scriptURL
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
