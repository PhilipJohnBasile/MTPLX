import Combine
import Foundation

public enum HermesConnectionState: Equatable, Sendable {
    case idle
    case checkingInstall
    case needsSetup(String)
    case starting
    case connected
    case failed(String)
}

public enum HermesMessageRole: String, Equatable, Sendable {
    case user
    case assistant
    case system
}

public struct HermesTranscriptMessage: Identifiable, Equatable, Sendable {
    public let id: String
    public var role: HermesMessageRole
    public var text: String
    public var isStreaming: Bool

    public init(
        id: String = UUID().uuidString,
        role: HermesMessageRole,
        text: String,
        isStreaming: Bool = false
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.isStreaming = isStreaming
    }
}

public enum HermesToolStatus: String, Equatable, Sendable {
    case running
    case complete
    case approval
    case waiting
    case failed
}

public struct HermesToolTrace: Identifiable, Equatable, Sendable {
    public let id: String
    public var name: String
    public var status: HermesToolStatus
    public var detail: String

    public init(
        id: String = UUID().uuidString,
        name: String,
        status: HermesToolStatus,
        detail: String
    ) {
        self.id = id
        self.name = name
        self.status = status
        self.detail = detail
    }
}

public struct HermesSavedSession: Identifiable, Equatable, Sendable {
    public let id: String
    public let title: String
    public let preview: String
    public let startedAt: Double
    public let messageCount: Int
    public let source: String

    public init(
        id: String,
        title: String,
        preview: String,
        startedAt: Double,
        messageCount: Int,
        source: String
    ) {
        self.id = id
        self.title = title
        self.preview = preview
        self.startedAt = startedAt
        self.messageCount = messageCount
        self.source = source
    }
}

private struct HermesGatewayEvent: Sendable {
    let type: String
    let sessionID: String?
    let payload: [String: JSONValue]
}

public enum HermesGatewayClientError: Error, LocalizedError {
    case disconnected
    case malformedResponse
    case rpcError(String)
    case sendFailed(String)

    public var errorDescription: String? {
        switch self {
        case .disconnected:
            return "Hermes gateway disconnected."
        case .malformedResponse:
            return "Hermes returned a malformed response."
        case .rpcError(let message):
            return message
        case .sendFailed(let message):
            return "Hermes request could not be sent: \(message)"
        }
    }
}

@MainActor
private final class HermesGatewayClient {
    private let task: URLSessionWebSocketTask
    private var nextID: Int = 1
    private var pending: [Int: CheckedContinuation<JSONValue, Error>] = [:]
    var onEvent: ((HermesGatewayEvent) -> Void)?
    var onDisconnect: ((String) -> Void)?

    init(url: URL) {
        task = URLSession.shared.webSocketTask(with: url)
    }

    func connect() {
        task.resume()
        receiveNext()
    }

    func close() {
        for (_, continuation) in pending {
            continuation.resume(throwing: HermesGatewayClientError.disconnected)
        }
        pending.removeAll()
        task.cancel(with: .goingAway, reason: nil)
    }

    func call(method: String, params: [String: JSONValue] = [:]) async throws -> JSONValue {
        let id = nextID
        nextID += 1
        let request: [String: JSONValue] = [
            "jsonrpc": .string("2.0"),
            "id": .number(Double(id)),
            "method": .string(method),
            "params": .object(params),
        ]
        let data = try JSONEncoder().encode(request)
        guard let text = String(data: data, encoding: .utf8) else {
            throw HermesGatewayClientError.malformedResponse
        }
        return try await withCheckedThrowingContinuation { continuation in
            pending[id] = continuation
            task.send(.string(text)) { [weak self] error in
                guard let error else { return }
                Task { @MainActor in
                    guard let self else { return }
                    self.pending.removeValue(forKey: id)?
                        .resume(throwing: HermesGatewayClientError.sendFailed(error.localizedDescription))
                }
            }
        }
    }

    private func receiveNext() {
        task.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success(let message):
                    self.handle(message)
                    self.receiveNext()
                case .failure(let error):
                    for (_, continuation) in self.pending {
                        continuation.resume(throwing: HermesGatewayClientError.disconnected)
                    }
                    self.pending.removeAll()
                    self.onDisconnect?(error.localizedDescription)
                }
            }
        }
    }

    private func handle(_ message: URLSessionWebSocketTask.Message) {
        let text: String?
        switch message {
        case .string(let raw):
            text = raw
        case .data(let data):
            text = String(data: data, encoding: .utf8)
        @unknown default:
            text = nil
        }
        guard let text,
              let data = text.data(using: .utf8),
              let root = try? JSONDecoder().decode([String: JSONValue].self, from: data)
        else {
            return
        }

        if let id = root["id"]?.intValue {
            let continuation = pending.removeValue(forKey: id)
            if let error = root["error"]?.objectValue {
                continuation?.resume(
                    throwing: HermesGatewayClientError.rpcError(
                        error["message"]?.stringValue ?? "Hermes RPC failed."
                    )
                )
            } else {
                continuation?.resume(returning: root["result"] ?? .null)
            }
            return
        }

        guard root["method"]?.stringValue == "event",
              let params = root["params"]?.objectValue,
              let type = params["type"]?.stringValue
        else {
            return
        }
        onEvent?(
            HermesGatewayEvent(
                type: type,
                sessionID: params["session_id"]?.stringValue,
                payload: params["payload"]?.objectValue ?? [:]
            )
        )
    }
}

@MainActor
public final class HermesAgentStore: ObservableObject {
    @Published public private(set) var connectionState: HermesConnectionState = .idle
    @Published public private(set) var installStatus: HermesInstallStatus?
    @Published public private(set) var profiles: [HermesProfile] = []
    @Published public private(set) var selectedProfile: HermesProfile?
    @Published public private(set) var sessions: [HermesSavedSession] = []
    @Published public private(set) var messages: [HermesTranscriptMessage] = []
    @Published public private(set) var toolTraces: [HermesToolTrace] = []
    @Published public private(set) var activeSessionID: String?
    @Published public private(set) var activeSessionKey: String?
    @Published public private(set) var activeSessionTitle: String?
    @Published public private(set) var isStreaming: Bool = false
    @Published public private(set) var gatewayReady: Bool = false
    @Published public private(set) var gatewayRepairInFlight: Bool = false
    @Published public private(set) var gatewayRepairMessage: String?
    @Published public private(set) var terminalAgentRunning: Bool = false

    private let integration: HermesIntegration
    private var sidecar: HermesSidecar?
    private var sidecarProfileName: String?
    private var sidecarConfigurationSignature: String?
    private var client: HermesGatewayClient?
    private var shuttingDown = false
    private var gatewayGeneration = 0
    /// Manual Terminal launches are owned by their receipt, not by a broad
    /// process scan. Stop and a superseding manual launch invalidate this
    /// generation before awaiting any external work.
    private var terminalHandoffGeneration = 0
    private var terminalHandoffLease: MTPLXTerminalHandoffLease?
    private var terminalHandoffTask: Task<Void, Never>?

    public init(integration: HermesIntegration = HermesIntegration()) {
        self.integration = integration
    }

    public var activeReference: HermesSessionReference? {
        guard let selectedProfile,
              let sessionID = activeSessionKey ?? activeSessionID
        else { return nil }
        return HermesSessionReference(
            profileName: selectedProfile.name,
            sessionID: sessionID,
            title: activeSessionTitle
        )
    }

    public func prepare(configuration: MTPLXAppConfiguration) async {
        connectionState = .checkingInstall
        gatewayRepairMessage = nil
        let status = await integration.installStatus()
        installStatus = status
        terminalAgentRunning = integration.hasLaunchedTerminalAgent()
        profiles = integration.discoverProfiles()
        if let remembered = configuration.lastHermesProfile,
           let profile = profiles.first(where: { $0.name == remembered }) {
            selectedProfile = profile
        } else if selectedProfile == nil {
            selectedProfile = profiles.first
        }
        switch status.kind {
        case .ready:
            connectionState = .idle
        case .missing, .incompatible:
            connectionState = .needsSetup(status.detail)
        }
    }

    /// Open (or re-open) the Hermes chat in a Terminal window. Used by
    /// the handoff panel's "Open in Terminal" action so the user can get
    /// back to the live agent without restarting the daemon.
    public func openTerminal(configuration: MTPLXAppConfiguration) {
        invalidateManualTerminalHandoff()
        let generation = terminalHandoffGeneration
        terminalHandoffTask = Task { @MainActor [weak self] in
            guard let self else { return }
            let result = await self.integration.launchInTerminal(
                configuration: configuration,
                isCurrent: { [weak self] in
                    guard let self else { return false }
                    return self.terminalHandoffGeneration == generation
                }
            )
            guard self.terminalHandoffGeneration == generation else {
                if let lease = result.terminalHandoffLease {
                    _ = self.integration.cancelTerminalHandoff(lease)
                }
                return
            }
            self.terminalHandoffLease = result.terminalHandoffLease
            self.terminalAgentRunning = result.action == .launched
            self.terminalHandoffTask = nil
        }
    }

    public func repairGateway() async {
        guard gatewayRepairInFlight == false else { return }
        gatewayRepairInFlight = true
        gatewayRepairMessage = nil
        defer { gatewayRepairInFlight = false }
        do {
            let result = try await integration.repairGateway()
            if let statusSummary = result.statusSummary {
                gatewayRepairMessage = "\(result.startSummary) \(statusSummary)"
            } else {
                gatewayRepairMessage = result.startSummary
            }
            let status = await integration.installStatus()
            installStatus = status
            terminalAgentRunning = integration.hasLaunchedTerminalAgent()
            profiles = integration.discoverProfiles()
            switch status.kind {
            case .ready:
                connectionState = .idle
            case .missing, .incompatible:
                connectionState = .needsSetup(status.detail)
            }
            if result.statusHealth != .healthy {
                gatewayReady = false
            }
        } catch {
            gatewayRepairMessage = Self.message(for: error)
            connectionState = .failed(Self.message(for: error))
        }
    }

    public func loadSessions(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async {
        selectedProfile = profile
        do {
            try await ensureGateway(profile: profile, configuration: configuration)
            let result = try await rpc("session.list", params: ["limit": .number(200)])
            sessions = Self.parseSessions(result)
            connectionState = .connected
        } catch {
            sessions = []
            connectionState = .failed(Self.message(for: error))
        }
    }

    @discardableResult
    public func startNewAgent(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async throws -> HermesSessionReference {
        selectedProfile = profile
        try await ensureGateway(profile: profile, configuration: configuration)
        let result = try await rpc("session.create", params: ["cols": .number(100)])
        guard let sessionID = result.objectValue?["session_id"]?.stringValue else {
            throw HermesGatewayClientError.malformedResponse
        }
        activeSessionID = sessionID
        activeSessionTitle = "New Hermes Agent"
        messages = []
        toolTraces = []
        activeSessionKey = (try? await liveSessionKey(for: sessionID)) ?? sessionID
        connectionState = .connected
        return HermesSessionReference(
            profileName: profile.name,
            sessionID: activeSessionKey ?? sessionID,
            title: activeSessionTitle
        )
    }

    @discardableResult
    public func resume(
        _ session: HermesSavedSession,
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async throws -> HermesSessionReference {
        selectedProfile = profile
        try await ensureGateway(profile: profile, configuration: configuration)
        let result = try await rpc(
            "session.resume",
            params: [
                "session_id": .string(session.id),
                "cols": .number(100),
            ]
        )
        guard let object = result.objectValue,
              let sessionID = object["session_id"]?.stringValue
        else {
            throw HermesGatewayClientError.malformedResponse
        }
        activeSessionID = sessionID
        activeSessionKey = object["resumed"]?.stringValue ?? session.id
        activeSessionTitle = session.title.isEmpty ? session.preview : session.title
        messages = Self.parseMessages(object["messages"])
        toolTraces = []
        connectionState = .connected
        return HermesSessionReference(
            profileName: profile.name,
            sessionID: activeSessionKey ?? session.id,
            title: activeSessionTitle
        )
    }

    @discardableResult
    public func resumeLast(
        configuration: MTPLXAppConfiguration
    ) async throws -> HermesSessionReference {
        guard let profileName = configuration.lastHermesProfile,
              let sessionID = configuration.lastHermesSessionID,
              let profile = profiles.first(where: { $0.name == profileName })
        else {
            throw HermesGatewayClientError.rpcError("No previous Hermes agent is saved.")
        }
        let session = HermesSavedSession(
            id: sessionID,
            title: configuration.lastHermesSessionTitle ?? "",
            preview: "",
            startedAt: 0,
            messageCount: 0,
            source: ""
        )
        return try await resume(session, profile: profile, configuration: configuration)
    }

    public func send(_ rawText: String) async {
        let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, let sessionID = activeSessionID, !isStreaming else { return }
        messages.append(HermesTranscriptMessage(role: .user, text: text))
        isStreaming = true
        do {
            _ = try await rpc(
                "prompt.submit",
                params: [
                    "session_id": .string(sessionID),
                    "text": .string(text),
                ]
            )
        } catch {
            isStreaming = false
            messages.append(
                HermesTranscriptMessage(
                    role: .system,
                    text: Self.message(for: error)
                )
            )
        }
    }

    public func interrupt() async {
        guard let sessionID = activeSessionID else { return }
        do {
            _ = try await rpc("session.interrupt", params: ["session_id": .string(sessionID)])
        } catch {
            messages.append(
                HermesTranscriptMessage(role: .system, text: Self.message(for: error))
            )
        }
        isStreaming = false
    }

    public func createProfile(named name: String, configuration: MTPLXAppConfiguration) async {
        do {
            let profile = try await integration.createProfile(named: name)
            profiles = integration.discoverProfiles()
            selectedProfile = profiles.first(where: { $0.name == profile.name }) ?? profile
            await loadSessions(profile: selectedProfile ?? profile, configuration: configuration)
        } catch {
            connectionState = .failed(Self.message(for: error))
        }
    }

    public func stop() async {
        shuttingDown = true
        invalidateManualTerminalHandoff()
        gatewayGeneration += 1
        client?.close()
        client = nil
        sidecar?.stop()
        sidecar = nil
        sidecarProfileName = nil
        sidecarConfigurationSignature = nil
        gatewayReady = false
        isStreaming = false
        activeSessionID = nil
        terminalAgentRunning = false
        connectionState = .idle
    }

    public func refreshTerminalAgentState() {
        terminalAgentRunning = integration.hasLaunchedTerminalAgent()
    }

    /// Narrow package test seam for the Store-owned manual Terminal lease.
    /// Production receipts arrive through `openTerminal(configuration:)`.
    func recordManualTerminalHandoffLeaseForTesting(_ lease: MTPLXTerminalHandoffLease) {
        invalidateManualTerminalHandoff()
        terminalHandoffLease = lease
        terminalAgentRunning = true
    }

    var manualTerminalHandoffLeaseIDForTesting: UUID? {
        terminalHandoffLease?.handoffID
    }

    private func invalidateManualTerminalHandoff() {
        terminalHandoffGeneration &+= 1
        terminalHandoffTask?.cancel()
        terminalHandoffTask = nil
        if let lease = terminalHandoffLease {
            _ = integration.cancelTerminalHandoff(lease)
        }
        terminalHandoffLease = nil
        terminalAgentRunning = false
    }

    private func ensureGateway(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async throws {
        let signature = Self.configurationSignature(configuration)
        if selectedProfile?.name == profile.name,
           sidecarProfileName == profile.name,
           sidecarConfigurationSignature == signature,
           sidecar?.process.isRunning == true,
           client != nil {
            return
        }
        shuttingDown = true
        gatewayGeneration += 1
        client?.close()
        client = nil
        sidecar?.stop()
        sidecar = nil
        sidecarProfileName = nil
        sidecarConfigurationSignature = nil
        gatewayReady = false
        let generation = gatewayGeneration
        shuttingDown = false
        connectionState = .starting
        let nextSidecar = try await integration.startDashboard(
            profile: profile,
            configuration: configuration
        )
        let nextClient = HermesGatewayClient(url: nextSidecar.webSocketURL)
        nextClient.onEvent = { [weak self] event in
            self?.handle(event)
        }
        nextClient.onDisconnect = { [weak self] message in
            guard let self,
                  self.gatewayGeneration == generation,
                  !self.shuttingDown
            else { return }
            self.connectionState = .failed(message)
        }
        sidecar = nextSidecar
        sidecarProfileName = profile.name
        sidecarConfigurationSignature = signature
        client = nextClient
        selectedProfile = profile
        nextClient.connect()
        connectionState = .connected
    }

    private func rpc(_ method: String, params: [String: JSONValue] = [:]) async throws -> JSONValue {
        guard let client else {
            throw HermesGatewayClientError.disconnected
        }
        return try await client.call(method: method, params: params)
    }

    private func liveSessionKey(for sessionID: String) async throws -> String? {
        let result = try await rpc(
            "session.active_list",
            params: ["current_session_id": .string(sessionID)]
        )
        guard let rows = result.objectValue?["sessions"]?.arrayValue else { return nil }
        for row in rows {
            guard let object = row.objectValue,
                  object["id"]?.stringValue == sessionID
            else { continue }
            return object["session_key"]?.stringValue
        }
        return nil
    }

    private func handle(_ event: HermesGatewayEvent) {
        if event.type == "gateway.ready" {
            gatewayReady = true
            return
        }
        if let activeSessionID,
           let eventSessionID = event.sessionID,
           eventSessionID != activeSessionID {
            return
        }

        switch event.type {
        case "message.start":
            if messages.last?.role != .assistant || messages.last?.isStreaming == false {
                messages.append(
                    HermesTranscriptMessage(
                        role: .assistant,
                        text: "",
                        isStreaming: true
                    )
                )
            }
            isStreaming = true
        case "message.delta":
            appendAssistantDelta(event.payload["text"]?.stringValue ?? "")
        case "message.complete":
            completeAssistantMessage(
                text: event.payload["text"]?.stringValue,
                reasoning: event.payload["reasoning"]?.stringValue
            )
        case "tool.start":
            toolTraces.append(
                HermesToolTrace(
                    name: Self.toolName(from: event.payload),
                    status: .running,
                    detail: Self.toolDetail(from: event.payload)
                )
            )
        case "tool.progress":
            updateLastTool(
                name: Self.toolName(from: event.payload),
                status: .running,
                detail: event.payload["preview"]?.stringValue ?? Self.toolDetail(from: event.payload)
            )
        case "tool.complete":
            updateLastTool(
                name: Self.toolName(from: event.payload),
                status: .complete,
                detail: Self.toolDetail(from: event.payload)
            )
        case "approval.request":
            toolTraces.append(
                HermesToolTrace(
                    name: "Approval",
                    status: .approval,
                    detail: "Auto-approved by MTPLX Hermes mode."
                )
            )
            if let sessionID = activeSessionID {
                Task {
                    _ = try? await rpc(
                        "approval.respond",
                        params: [
                            "session_id": .string(sessionID),
                            "choice": .string("allow"),
                            "all": .bool(true),
                        ]
                    )
                }
            }
        case "clarify.request", "sudo.request", "secret.request":
            toolTraces.append(
                HermesToolTrace(
                    name: event.type.replacingOccurrences(of: ".request", with: ""),
                    status: .waiting,
                    detail: Self.toolDetail(from: event.payload)
                )
            )
        case "error":
            messages.append(
                HermesTranscriptMessage(
                    role: .system,
                    text: event.payload["message"]?.stringValue ?? "Hermes reported an error."
                )
            )
            isStreaming = false
        case "session.info":
            if let key = event.payload["session_key"]?.stringValue {
                activeSessionKey = key
            }
        default:
            break
        }
    }

    private func appendAssistantDelta(_ delta: String) {
        guard !delta.isEmpty else { return }
        if messages.last?.role != .assistant || messages.last?.isStreaming == false {
            messages.append(
                HermesTranscriptMessage(role: .assistant, text: delta, isStreaming: true)
            )
        } else {
            messages[messages.count - 1].text += delta
        }
        isStreaming = true
    }

    private func completeAssistantMessage(text: String?, reasoning: String?) {
        if let reasoning, !reasoning.isEmpty {
            toolTraces.append(
                HermesToolTrace(name: "Thought", status: .complete, detail: reasoning)
            )
        }
        let finalText = text ?? ""
        if messages.last?.role == .assistant {
            messages[messages.count - 1].text = finalText.isEmpty
                ? messages[messages.count - 1].text
                : finalText
            messages[messages.count - 1].isStreaming = false
        } else if !finalText.isEmpty {
            messages.append(
                HermesTranscriptMessage(role: .assistant, text: finalText, isStreaming: false)
            )
        }
        isStreaming = false
        if let activeSessionID {
            Task {
                activeSessionKey = (try? await liveSessionKey(for: activeSessionID)) ?? activeSessionKey
            }
        }
    }

    private func updateLastTool(name: String, status: HermesToolStatus, detail: String) {
        if let idx = toolTraces.lastIndex(where: { $0.name == name && $0.status == .running }) {
            toolTraces[idx].status = status
            if !detail.isEmpty {
                toolTraces[idx].detail = detail
            }
        } else {
            toolTraces.append(HermesToolTrace(name: name, status: status, detail: detail))
        }
    }

    private static func parseSessions(_ value: JSONValue) -> [HermesSavedSession] {
        guard let rows = value.objectValue?["sessions"]?.arrayValue else { return [] }
        return rows.compactMap { row in
            guard let object = row.objectValue,
                  let id = object["id"]?.stringValue
            else { return nil }
            return HermesSavedSession(
                id: id,
                title: object["title"]?.stringValue ?? "",
                preview: object["preview"]?.stringValue ?? "",
                startedAt: object["started_at"]?.doubleValue ?? 0,
                messageCount: object["message_count"]?.intValue ?? 0,
                source: object["source"]?.stringValue ?? ""
            )
        }
    }

    private static func parseMessages(_ value: JSONValue?) -> [HermesTranscriptMessage] {
        guard let rows = value?.arrayValue else { return [] }
        return rows.compactMap { row in
            guard let object = row.objectValue else { return nil }
            let role = HermesMessageRole(rawValue: object["role"]?.stringValue ?? "") ?? .assistant
            let text = textFromMessageObject(object)
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            return HermesTranscriptMessage(role: role, text: text)
        }
    }

    private static func textFromMessageObject(_ object: [String: JSONValue]) -> String {
        if let text = object["text"]?.stringValue {
            return text
        }
        if let content = object["content"] {
            return text(from: content)
        }
        return ""
    }

    private static func text(from value: JSONValue) -> String {
        switch value {
        case .string(let text):
            return text
        case .array(let array):
            return array.map(text(from:)).joined(separator: "\n")
        case .object(let object):
            if let text = object["text"]?.stringValue {
                return text
            }
            if let content = object["content"] {
                return text(from: content)
            }
            return ""
        default:
            return ""
        }
    }

    private static func toolName(from payload: [String: JSONValue]) -> String {
        payload["name"]?.stringValue
            ?? payload["tool"]?.stringValue
            ?? payload["command"]?.stringValue
            ?? "Tool"
    }

    private static func toolDetail(from payload: [String: JSONValue]) -> String {
        if let preview = payload["preview"]?.stringValue, !preview.isEmpty {
            return preview
        }
        if let message = payload["message"]?.stringValue, !message.isEmpty {
            return message
        }
        if let result = payload["result"]?.stringValue, !result.isEmpty {
            return result
        }
        if let args = payload["args"]?.objectValue {
            return args
                .map { "\($0.key)=\(Self.text(from: $0.value))" }
                .sorted()
                .joined(separator: " ")
        }
        return ""
    }

    private static func message(for error: Error) -> String {
        if let localized = error as? LocalizedError,
           let description = localized.errorDescription {
            return description
        }
        return String(describing: error)
    }

    private static func configurationSignature(_ configuration: MTPLXAppConfiguration) -> String {
        [
            configuration.host,
            String(configuration.port),
            configuration.model,
            configuration.apiKey ?? "",
        ].joined(separator: "|")
    }
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { value } else { nil }
    }

    var arrayValue: [JSONValue]? {
        if case .array(let value) = self { value } else { nil }
    }
}
