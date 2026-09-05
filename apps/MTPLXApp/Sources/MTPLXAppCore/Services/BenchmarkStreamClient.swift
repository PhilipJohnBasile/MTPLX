import Foundation

// MARK: - BenchmarkStreamClient
//
// SSE consumer for `GET /v1/mtplx/benchmarks/aime/{run_id}/stream`.
//
// Mirrors the connect-shape of `MetricsStreamClient` (state + event
// callbacks, reconnect-with-backoff). We reuse `SSEParser` from
// MetricsStreamClient and the same byte-buffer + trailing-delimiter
// scan pattern that solves the macOS `URLSession.bytes` buffering bug
// (documented in MTPLXChatClient.swift). The benchmark SSE schema is
// different from chat-completion chunks (`run_started`,
// `question_started`, `reasoning_delta`, `answer_delta`,
// `answer_verification_started`, `answer_verification_done`,
// `cap_recovery_started`,
// `question_done`, `run_paused`, `run_resumed`, `run_cancelled`,
// `run_done`, `error`) so we cannot reuse MTPLXChatClient directly.

public enum BenchmarkStreamError: Error, Sendable, Equatable {
    case httpStatus(Int, String)
    case invalidResponse
    case daemonUnreachable
    case decode(String)
}

public final class BenchmarkStreamClient: Sendable {
    private let apiClient: MTPLXAPIClient
    private let parser = SSEParser()

    /// Shared JSON decoder configured for the same date format the
    /// Python runner emits (`_iso` returns ISO 8601 with `Z` suffix).
    private let decoder: JSONDecoder = MTPLXAPIClient.makeDefaultDecoder()

    public init(apiClient: MTPLXAPIClient) {
        self.apiClient = apiClient
    }

    /// Connect, parse SSE messages, and forward decoded `BenchEvent`s
    /// until the stream closes naturally (a terminal `run_done`,
    /// `run_cancelled`, or `error` event), the task is cancelled, or
    /// an unrecoverable error fires.
    ///
    /// - Parameters:
    ///   - runId: the active run id returned by `POST /start`.
    ///   - onEvent: called on the awaiting actor for every decoded event.
    ///              `keepAlive` is suppressed (handled internally).
    public func connect(
        runId: String,
        onEvent: @escaping @Sendable (BenchEvent) async -> Void
    ) async throws {
        var request = URLRequest(url: apiClient.aimeStreamURL(runId: runId))
        request.httpMethod = "GET"
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue("no-cache", forHTTPHeaderField: "Cache-Control")
        if let auth = apiClient.authorizationHeader {
            request.setValue(auth, forHTTPHeaderField: "Authorization")
        }

        let (bytes, response): (URLSession.AsyncBytes, URLResponse)
        do {
            (bytes, response) = try await apiClient.session.bytes(for: request)
        } catch let urlError as URLError where
            [URLError.Code.cannotConnectToHost,
             .networkConnectionLost,
             .cannotFindHost,
             .notConnectedToInternet].contains(urlError.code) {
            throw BenchmarkStreamError.daemonUnreachable
        }
        guard let http = response as? HTTPURLResponse else {
            throw BenchmarkStreamError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            // Drain a small body window so we can include a useful
            // error string.
            var body = Data()
            var count = 0
            for try await byte in bytes {
                body.append(byte)
                count += 1
                if count >= 2048 { break }
            }
            let text = String(data: body, encoding: .utf8) ?? ""
            throw BenchmarkStreamError.httpStatus(http.statusCode, text)
        }

        var buffer = Data()
        buffer.reserveCapacity(4096)
        for try await byte in bytes {
            if Task.isCancelled { return }
            buffer.append(byte)
            // Same trailing-delimiter trick as MTPLXChatClient: O(1)
            // per byte instead of an O(N) buffer rescan.
            if buffer.hasSSEDelimiterSuffix {
                let text = String(decoding: buffer, as: UTF8.self)
                buffer.removeAll(keepingCapacity: true)
                for msg in parser.parse(text) {
                    if let event = decode(msg) {
                        await onEvent(event)
                        if event.isTerminal { return }
                    }
                }
            }
        }
        // Final flush in case the server closed without a trailing
        // delimiter.
        if !buffer.isEmpty {
            let text = String(decoding: buffer, as: UTF8.self)
            for msg in parser.parse(text) {
                if let event = decode(msg) {
                    await onEvent(event)
                    if event.isTerminal { return }
                }
            }
        }
    }

    // MARK: - Decoding

    /// Decode a single SSE `(event:, data:)` pair into a `BenchEvent`.
    /// Returns nil for unknown/uninteresting events (keep-alives, etc.)
    /// so the caller can ignore them without inspecting strings.
    private func decode(_ msg: SSEMessage) -> BenchEvent? {
        // Keep-alives are sent as `: keep-alive\n\n` which the parser
        // already filters (no `data:` line) so they never reach here.
        // Defensive: catch a hypothetical `event: keep-alive` too.
        if msg.event == "keep-alive" {
            return .keepAlive
        }

        let data = Data(msg.data.utf8)
        guard let payload = try? decoder.decode(_BenchPayload.self, from: data) else {
            return nil
        }

        switch msg.event {
        case "run_started":
            return .runStarted(
                runID: payload.runID ?? "",
                total: payload.total ?? 0,
                model: payload.model ?? "",
                startedAt: payload.startedAt
            )
        case "question_started":
            return .questionStarted(
                idx: payload.idx ?? 0,
                attempt: payload.attempt ?? 1,
                id: payload.id ?? "",
                set: payload.set ?? "",
                year: payload.year ?? 2026,
                problem: payload.problem ?? ""
            )
        case "reasoning_delta":
            return .reasoningDelta(
                idx: payload.idx ?? 0,
                attempt: payload.attempt ?? 1,
                text: payload.text ?? ""
            )
        case "answer_delta":
            return .answerDelta(
                idx: payload.idx ?? 0,
                attempt: payload.attempt ?? 1,
                text: payload.text ?? ""
            )
        case "question_progress":
            var metrics = payload.progress ?? MetricsLatest()
            if metrics.values["request_id"] == nil, let requestID = payload.requestID {
                metrics.values["request_id"] = .string(requestID)
            }
            return .questionProgress(
                idx: payload.idx ?? 0,
                attempt: payload.attempt ?? 1,
                requestID: payload.requestID ?? "",
                metrics: metrics
            )
        case "answer_verification_started":
            return .answerVerificationStarted(
                idx: payload.idx ?? 0,
                attempt: payload.attempt ?? 1,
                mode: payload.mode ?? "",
                proposedAnswer: payload.proposedAnswer
            )
        case "answer_verification_done":
            return .answerVerificationDone(
                idx: payload.idx ?? 0,
                attempt: payload.attempt ?? 1,
                mode: payload.mode ?? "",
                proposedAnswer: payload.proposedAnswer,
                verifiedAnswer: payload.verifiedAnswer,
                verifierAnswers: payload.verifierAnswers ?? [],
                resolution: payload.resolution,
                durationMs: payload.durationMs
            )
        case "cap_recovery_started":
            return .capRecoveryStarted(
                idx: payload.idx ?? 0,
                attempt: payload.attempt ?? 1,
                requestID: payload.requestID ?? "",
                mode: payload.mode ?? ""
            )
        case "question_done":
            let status = QuestionStatus(rawValue: payload.status ?? "") ?? .abstain
            return .questionDone(
                idx: payload.idx ?? 0,
                attempt: payload.attempt ?? 1,
                id: payload.id ?? "",
                extracted: payload.extracted,
                expected: payload.expected ?? 0,
                status: status,
                durationMs: payload.durationMs,
                reasoningTokenCount: payload.reasoningTokenCount ?? 0,
                answerTokenCount: payload.answerTokenCount ?? 0
            )
        case "run_paused":
            return .runPaused(runID: payload.runID ?? "")
        case "run_resumed":
            return .runResumed(runID: payload.runID ?? "")
        case "run_cancelled":
            return .runCancelled(
                runID: payload.runID ?? "",
                score: payload.score ?? 0,
                total: payload.total ?? 0
            )
        case "run_done":
            return .runDone(
                runID: payload.runID ?? "",
                score: payload.score ?? 0,
                total: payload.total ?? 0,
                accuracy: payload.accuracy,
                durationMs: payload.durationMs
            )
        case "error":
            return .error(
                message: payload.message ?? tr("unknown error"),
                recoverable: payload.recoverable ?? false
            )
        default:
            return nil
        }
    }
}

// MARK: - Convenience

extension BenchEvent {
    public var isTerminal: Bool {
        switch self {
        case .runDone, .runCancelled, .error: return true
        default: return false
        }
    }
}

// MARK: - SSE payload (single struct decodes every event shape)

/// Permissive Codable struct that holds every field any of the SSE
/// event payloads might emit. Each `case` in `BenchmarkStreamClient.decode`
/// picks the subset it cares about; the rest stay nil. Snake-case
/// mappings match the Python emitter exactly.
private struct _BenchPayload: Decodable {
    let runID: String?
    let total: Int?
    let model: String?
    let startedAt: Date?
    let idx: Int?
    let attempt: Int?
    let id: String?
    let set: String?
    let year: Int?
    let problem: String?
    let text: String?
    let mode: String?
    let requestID: String?
    let proposedAnswer: Int?
    let verifiedAnswer: Int?
    let verifierAnswers: [Int?]?
    let resolution: String?
    let extracted: Int?
    let expected: Int?
    let status: String?
    let durationMs: Int?
    let reasoningTokenCount: Int?
    let answerTokenCount: Int?
    let score: Int?
    let accuracy: Double?
    let message: String?
    let recoverable: Bool?
    let progress: MetricsLatest?

    enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case total
        case model
        case startedAt = "started_at"
        case idx
        case attempt
        case id
        case set
        case year
        case problem
        case text
        case mode
        case requestID = "request_id"
        case proposedAnswer = "proposed_answer"
        case verifiedAnswer = "verified_answer"
        case verifierAnswers = "verifier_answers"
        case resolution
        case extracted
        case expected
        case status
        case durationMs = "duration_ms"
        case reasoningTokenCount = "reasoning_token_count"
        case answerTokenCount = "answer_token_count"
        case score
        case accuracy
        case message
        case recoverable
        case progress
    }
}
