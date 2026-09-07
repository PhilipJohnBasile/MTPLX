import Foundation
import Combine

// MARK: - BenchmarkOrchestrator
//
// `@MainActor ObservableObject` that owns the AIME 2026 benchmark run
// lifecycle from the SwiftUI side. Hoisted to MTPLXApp root next to
// `ForgeOrchestrator` so wizard-style state survives tab switches and
// overlay dismiss/re-open without losing in-flight progress.
//
// Mirrors ForgeOrchestrator deliberately: granular `@Published` per UI
// section, stored `Task` handles for explicit cancellation, services
// are injected as closures so the orchestrator can be unit-tested
// without spinning up a real daemon.
//
// Normal daemon telemetry stays in `MTPLXBackendStore`. AIME worker
// telemetry is different: per-question isolation runs the decode in a
// private worker, so the parent daemon's metrics stream can be stale. The
// benchmark stream carries those worker progress frames through
// `AIMELiveTelemetryStore`, which only the small number strip observes.

@MainActor
public final class AIMELiveTelemetryStore: ObservableObject {
    @Published public private(set) var latest: MetricsLatest? = nil

    public init() {}

    public func reset() {
        latest = nil
    }

    public func update(requestID: String, metrics: MetricsLatest) {
        var values = metrics.values
        if values["request_id"] == nil, !requestID.isEmpty {
            values["request_id"] = .string(requestID)
        }
        latest = MetricsLatest(values: values)
    }
}

@MainActor
public final class BenchmarkOrchestrator: ObservableObject {
    // MARK: - Published surface (granular so SwiftUI invalidates per section)

    @Published public private(set) var state: BenchRunState = .idle
    @Published public private(set) var runID: String? = nil
    @Published public private(set) var model: String = ""
    @Published public private(set) var startedAt: Date? = nil
    @Published public private(set) var endedAt: Date? = nil
    @Published public private(set) var pausePending: Bool = false
    @Published public private(set) var skipPending: Bool = false
    @Published public private(set) var lastError: String? = nil

    /// 30 placeholder rows are allocated at start time, then mutated
    /// as `question_started` / `question_done` events arrive. Always
    /// has `total` entries indexed 1..total via `idx`.
    @Published public private(set) var results: [BenchQuestionResult] = []

    /// 1..N while running; 0 when idle or before the first question.
    @Published public private(set) var currentIdx: Int = 0
    @Published public private(set) var currentAttempt: Int = 0
    @Published public private(set) var currentRequestID: String? = nil
    @Published public private(set) var currentVerification: BenchAnswerVerificationState? = nil

    /// Incremental streaming documents for the CURRENT question. SwiftUI
    /// observes these stores directly so the live render path updates stable
    /// blocks instead of repeatedly re-splitting growing strings.
    public let liveReasoningDocument = StreamingDocumentStore(mode: .mathLines)
    public let liveAnswerDocument = StreamingDocumentStore(mode: .plainLines)
    public let liveTelemetry = AIMELiveTelemetryStore()

    /// Raw reconstruction retained for persistence/tests. Not published:
    /// the block stores above are the live UI surface.
    public var liveReasoning: String { liveReasoningDocument.rawText }
    public var liveAnswer: String { liveAnswerDocument.rawText }

    /// `BenchmarkGrader.extractBoxed(liveAnswer)`. Nil while the model
    /// hasn't written a `\boxed{...}` yet. Drives the big tabular hero
    /// numeral on `BenchLiveCard` BEFORE the terminal `question_done`
    /// event arrives.
    @Published public private(set) var liveExtractedAnswer: Int? = nil
    @Published public private(set) var liveAnswerIsGraded: Bool = false

    /// History of finished runs, sorted most-recent-first. Loaded on
    /// overlay open and after each run finishes.
    @Published public private(set) var history: [BenchRunSummary] = []

    // MARK: - Derived values

    public var total: Int { results.count }

    public var score: Int {
        results.reduce(0) { $0 + ($1.status == .correct ? 1 : 0) }
    }

    /// Number of questions that produced a terminal status (correct /
    /// wrong / abstain). Pending questions are excluded.
    public var resolved: Int {
        results.reduce(0) { $0 + ($1.status == .pending ? 0 : 1) }
    }

    public var accuracy: Double? {
        guard resolved > 0 else { return nil }
        return Double(score) / Double(resolved)
    }

    public var currentProblem: BenchProblem? {
        guard currentIdx > 0, currentIdx <= results.count else { return nil }
        return results[currentIdx - 1].problem
    }

    /// Elapsed time since `startedAt`, ticking until `endedAt` arrives
    /// (or "now" if still running). Views call this from a
    /// `TimelineView(.periodic(...))` for live updates.
    public func elapsed(at now: Date = Date()) -> TimeInterval {
        guard let started = startedAt else { return 0 }
        let end = endedAt ?? now
        return max(0, end.timeIntervalSince(started))
    }

    // MARK: - Services (injectable for tests)

    /// Lazy provider so we always read the freshest `MTPLXAPIClient`
    /// from the live `MTPLXBackendStore` (matches the `ChatViewModel`
    /// pattern at `MTPLXApp.swift:30-33`). Mid-run port/key changes
    /// in Settings won't strand the next request.
    public typealias APIClientProvider = @MainActor () -> MTPLXAPIClient
    public typealias DaemonReadinessProvider = @MainActor () async throws -> Void
    public typealias StartOptionsProvider = @MainActor () -> BenchmarkStartOptions
    public typealias StreamFactory = @MainActor (MTPLXAPIClient, String) -> any BenchmarkStreaming

    private var apiClientProvider: APIClientProvider
    private var daemonReadinessProvider: DaemonReadinessProvider?
    private var startOptionsProvider: StartOptionsProvider
    private let streamFactory: StreamFactory

    /// Default initializer used by the SwiftUI host. `apiClientProvider`
    /// is set via `attach(apiClientProvider:)` immediately after the
    /// host has wired up `MTPLXBackendStore`; the orchestrator hoists
    /// to app root with no environment so we can't capture the store
    /// at init time.
    public init() {
        self.apiClientProvider = { fatalError("BenchmarkOrchestrator not attached to a backend") }
        self.daemonReadinessProvider = nil
        self.startOptionsProvider = { BenchmarkStartOptions() }
        self.streamFactory = { client, runId in
            DefaultBenchmarkStreaming(client: BenchmarkStreamClient(apiClient: client), runId: runId)
        }
    }

    /// Test-only initializer.
    public init(
        apiClientProvider: @escaping APIClientProvider,
        daemonReadinessProvider: DaemonReadinessProvider? = nil,
        startOptionsProvider: @escaping StartOptionsProvider = { BenchmarkStartOptions() },
        streamFactory: @escaping StreamFactory
    ) {
        self.apiClientProvider = apiClientProvider
        self.daemonReadinessProvider = daemonReadinessProvider
        self.startOptionsProvider = startOptionsProvider
        self.streamFactory = streamFactory
    }

    public func attach(
        apiClientProvider: @escaping APIClientProvider,
        daemonReadinessProvider: DaemonReadinessProvider? = nil,
        startOptionsProvider: @escaping StartOptionsProvider = { BenchmarkStartOptions() }
    ) {
        self.apiClientProvider = apiClientProvider
        self.daemonReadinessProvider = daemonReadinessProvider
        self.startOptionsProvider = startOptionsProvider
    }

    // MARK: - Task handles

    private var streamTask: Task<Void, Never>?
    private var reasoningFlushTask: Task<Void, Never>?
    private var streamGeneration: Int = 0
    private var flushGeneration: Int = 0

    /// Buffers that absorb reasoning_delta / answer_delta chunks; the
    /// flush task drains them to the `@Published` mirrors at ~6 Hz so
    /// SwiftUI never re-runs the live card body on every SSE byte. Both
    /// the streamed reasoning AND the streamed answer are buffered — the
    /// answer used to append straight to `@Published liveAnswer` on every
    /// token, which fired `objectWillChange` (and re-rendered the whole
    /// overlay) tens of times a second during the answer phase.
    private var _reasoningBuffer: String = ""
    private var _answerBuffer: String = ""
    private var liveAnswerExtractor = LiveAnswerExtractor()
    private static let streamFlushIntervalNs: UInt64 = 250_000_000  // 250 ms = 4 Hz

    // MARK: - Public lifecycle

    /// Kicks off a new AIME 2026 run. No-op if a run is already alive
    /// in this orchestrator (the backend's 409 contract will also
    /// catch a stale daemon-side run separately).
    public func start(year: Int = 2026, questionLimit: Int? = nil) async {
        var startFields: [String: AIMEDiagnosticValue] = ["year": .int(year)]
        if let questionLimit {
            startFields["question_limit"] = .int(questionLimit)
        }
        AIMEDiagnostics.record(
            "start_invoked",
            fields: diagnosticFields(extra: startFields)
        )
        guard state.isTerminal || state == .idle else {
            AIMEDiagnostics.record("start_ignored_live_state", fields: diagnosticFields())
            return
        }
        lastError = nil
        do {
            try await daemonReadinessProvider?()
            let client = apiClientProvider()
            let options = startOptionsProvider()
            let effectiveQuestionLimit = questionLimit ?? options.questionLimit
            let started = try await client.aimeStart(
                year: year,
                temperature: options.temperature,
                topP: options.topP,
                topK: options.topK,
                enableThinking: options.enableThinking,
                questionProcessIsolation: options.questionProcessIsolation,
                questionLimit: effectiveQuestionLimit
            )
            await primeForRun(
                runID: started.runID,
                model: started.model,
                total: started.total,
                startedAt: started.startedAt
            )
            attachStream(runId: started.runID, problems: [], existingResults: nil)
        } catch let MTPLXAPIClientError.httpStatus(409, body) {
            // Daemon already has an active run - surface its id so the
            // overlay can rehydrate against it.
            lastError = tr("Another AIME run is already active.")
            AIMEDiagnostics.record("start_conflict_active_daemon_run", fields: diagnosticFields())
            // Best effort: refresh active+snapshot to rehydrate.
            await refreshActiveRun()
            _ = body
        } catch {
            lastError = tr("Failed to start AIME run: %@", error.localizedDescription)
            AIMEDiagnostics.record(
                "start_failed",
                fields: diagnosticFields(extra: ["error_type": .string(String(describing: type(of: error)))])
            )
        }
    }

    /// Hard-pauses the active decode. The daemon stops the current
    /// attempt, emits `run_paused`, then resume retries the same
    /// question from a fresh prompt.
    public func pause() async {
        guard let runID, state == .running else { return }
        pausePending = true
        do {
            _ = try await apiClientProvider().aimePause(runId: runID)
        } catch {
            lastError = tr("Pause failed: %@", error.localizedDescription)
            pausePending = false
        }
    }

    public func resume() async {
        guard let runID, state == .paused || pausePending else { return }
        pausePending = false
        do {
            _ = try await apiClientProvider().aimeResume(runId: runID)
        } catch {
            lastError = tr("Resume failed: %@", error.localizedDescription)
        }
    }

    /// User-visible escape hatch for a pathological problem. This marks
    /// only the current question as an abstain and lets the run advance.
    public func skipCurrent() async {
        guard let runID, state == .running, currentIdx > 0, !skipPending else { return }
        skipPending = true
        do {
            _ = try await apiClientProvider().aimeSkip(runId: runID)
            refreshSkipSnapshotSoon(runID: runID, skippedIdx: currentIdx)
        } catch {
            lastError = tr("Skip failed: %@", error.localizedDescription)
            skipPending = false
        }
    }

    /// Mid-decode abort. Persists partial results server-side and
    /// surfaces `run_cancelled` via SSE, but the UI cannot wait for
    /// either — the user wants the run to *look* cancelled
    /// immediately. We:
    ///   1. Flip local state to `.cancelled` and tear down the local
    ///      stream so the running tile drops out of the "solving"
    ///      pulse and the panel switches to the summary.
    ///   2. Send the API cancel best-effort so the daemon stops
    ///      decoding the current problem and saves partial results.
    /// If there's no runID we still flip local state — the user sees
    /// "Cancelled" even if no backend record exists yet.
    public func cancel(reason: String = "unspecified") async {
        let activeRunID = runID
        AIMEDiagnostics.record(
            "cancel_invoked",
            fields: diagnosticFields(extra: ["reason": .string(reason)]),
            flushImmediately: true,
            force: true
        )
        streamTask?.cancel()
        reasoningFlushTask?.cancel()
        streamTask = nil
        reasoningFlushTask = nil
        streamGeneration += 1
        flushGeneration += 1
        pausePending = false
        skipPending = false
        currentVerification = nil
        endedAt = Date()
        state = .cancelled

        guard let activeRunID else { return }
        do {
            let snapshot = try await apiClientProvider().aimeCancel(runId: activeRunID)
            await applyCancelledSnapshot(snapshot)
            refreshCancelledSnapshotSoon(runID: activeRunID)
        } catch {
            lastError = tr("Cancel failed: %@", error.localizedDescription)
            AIMEDiagnostics.record(
                "cancel_failed",
                fields: diagnosticFields(extra: ["error_type": .string(String(describing: type(of: error)))])
            )
        }
    }

    /// Clear local state to return to the empty hero. Does NOT abort
    /// any backend run — call `cancel()` first if needed.
    public func reset() {
        AIMEDiagnostics.record("reset_invoked", fields: diagnosticFields())
        streamTask?.cancel()
        reasoningFlushTask?.cancel()
        streamTask = nil
        reasoningFlushTask = nil
        streamGeneration += 1
        flushGeneration += 1
        state = .idle
        runID = nil
        startedAt = nil
        endedAt = nil
        currentIdx = 0
        currentAttempt = 0
        currentRequestID = nil
        currentVerification = nil
        liveReasoningDocument.reset()
        liveAnswerDocument.reset()
        liveTelemetry.reset()
        liveExtractedAnswer = nil
        liveAnswerIsGraded = false
        liveAnswerExtractor.reset()
        pausePending = false
        skipPending = false
        results = []
        _reasoningBuffer = ""
        _answerBuffer = ""
    }

    /// Clears the currently presented terminal run and removes its saved
    /// JSONL history file. Live runs are intentionally not discarded here:
    /// close/collapse keeps them recoverable through the AIME expand handle.
    @discardableResult
    public func discardPresentedRun() -> Bool {
        guard let currentRunID = runID, state.isTerminal else {
            reset()
            return true
        }
        let discarded = Self.removePersistedRunFile(runID: currentRunID)
        reset()
        history.removeAll { $0.runID == currentRunID }
        if discarded {
            lastError = nil
        } else {
            lastError = tr("Could not remove saved AIME run %@.", String(describing: currentRunID))
        }
        return discarded
    }

    public func dismissError() {
        lastError = nil
    }

    public func prepareForPresentation() {
        if state == .idle {
            lastError = nil
        }
    }

    /// Fetch the run already in flight on the daemon, if any. Called
    /// on overlay-open so a benchmark survives app/overlay close.
    public func refreshActiveRun() async {
        do {
            let active = try await apiClientProvider().aimeActive()
            guard let activeID = active.activeRunID else {
                // No live run - if WE thought one was alive, clear it.
                if state.isLive { reset() }
                return
            }
            let snap = try await apiClientProvider().aimeSnapshot(runId: activeID)
            await primeFromSnapshot(snap)
            attachStream(runId: activeID, problems: [], existingResults: results)
        } catch {
            // Quiet failure: the backend may not be running yet.
        }
    }

    public func refreshHistory(limit: Int = 5) async {
        do {
            let resp = try await apiClientProvider().aimeHistory(limit: limit)
            history = resp.runs
        } catch {
            // History is non-critical; failures stay silent.
        }
    }

    // MARK: - Internals

    private func diagnosticFields(
        extra: [String: AIMEDiagnosticValue] = [:]
    ) -> [String: AIMEDiagnosticValue] {
        guard AIMEDiagnostics.isEnabled else { return [:] }
        var fields = AIMEDiagnostics.fields(
            ("state", AIMEDiagnostics.string(state.rawValue)),
            ("run_id", AIMEDiagnostics.string(runID)),
            ("model", AIMEDiagnostics.string(model)),
            ("idx", .int(currentIdx)),
            ("attempt", .int(currentAttempt)),
            ("current_request_id", AIMEDiagnostics.string(currentRequestID)),
            ("reasoning_tokens", .int(liveReasoningDocument.wordCount)),
            ("answer_tokens", .int(liveAnswerDocument.wordCount)),
            ("reasoning_blocks", .int(liveReasoningDocument.blocks.count)),
            ("answer_blocks", .int(liveAnswerDocument.blocks.count)),
            ("reasoning_buffer_bytes", .int(_reasoningBuffer.utf8.count)),
            ("answer_buffer_bytes", .int(_answerBuffer.utf8.count)),
            ("stream_generation", .int(streamGeneration)),
            ("flush_generation", .int(flushGeneration))
        )
        fields.merge(extra) { _, new in new }
        return fields
    }

    private static func removePersistedRunFile(runID: String) -> Bool {
        guard isSafePersistedRunID(runID) else { return false }
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent("benchmarks", isDirectory: true)
            .appendingPathComponent("aime", isDirectory: true)
            .appendingPathComponent("\(runID).jsonl", isDirectory: false)
        guard FileManager.default.fileExists(atPath: url.path) else { return true }
        do {
            try FileManager.default.removeItem(at: url)
            return true
        } catch {
            return false
        }
    }

    private static func isSafePersistedRunID(_ runID: String) -> Bool {
        guard !runID.isEmpty, !runID.contains("..") else { return false }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        return runID.unicodeScalars.allSatisfy { allowed.contains($0) }
    }

    private func primeForRun(
        runID: String,
        model: String,
        total: Int,
        startedAt: Date?
    ) async {
        self.runID = runID
        self.model = model
        self.startedAt = startedAt ?? Date()
        self.endedAt = nil
        self.state = .running
        self.currentIdx = 0
        self.currentAttempt = 0
        self.currentRequestID = nil
        self.currentVerification = nil
        self.liveReasoningDocument.reset()
        self.liveAnswerDocument.reset()
        self.liveTelemetry.reset()
        self.liveExtractedAnswer = nil
        self.liveAnswerIsGraded = false
        self.liveAnswerExtractor.reset()
        self.pausePending = false
        self.skipPending = false
        self._reasoningBuffer = ""
        self._answerBuffer = ""
        AIMEDiagnostics.record(
            "prime_for_run",
            fields: diagnosticFields(extra: ["total": .int(total)])
        )
        // Allocate `total` pending placeholders so the grid renders
        // immediately. `BenchProblem` fields are zeroed until the
        // matching question_started event lands.
        self.results = (1...total).map { idx in
            BenchQuestionResult(
                idx: idx,
                problem: BenchProblem(
                    id: "?-\(idx)",
                    set: idx <= 15 ? "AIME I" : "AIME II",
                    year: 2026,
                    index: idx <= 15 ? idx : idx - 15,
                    problem: "",
                    answer: 0,
                    source: ""
                ),
                status: .pending
            )
        }
    }

    private func primeFromSnapshot(_ snap: BenchSnapshotResponse) async {
        self.runID = snap.runID
        self.model = snap.model
        self.startedAt = snap.startedAt ?? Date()
        self.endedAt = snap.endedAt
        self.state = BenchRunState(rawValue: snap.state) ?? .idle
        self.currentIdx = snap.currentIdx
        self.currentAttempt = snap.currentAttempt ?? 0
        self.currentRequestID = snap.currentRequestID
            ?? Self.requestID(
                runID: snap.runID,
                idx: snap.currentIdx,
                attempt: snap.currentAttempt ?? 0
            )
        self.currentVerification = nil
        self.pausePending = snap.paused
        self.skipPending = false
        self.liveReasoningDocument.reset()
        self.liveAnswerDocument.reset()
        self.liveTelemetry.reset()
        self.liveExtractedAnswer = nil
        self.liveAnswerIsGraded = false
        self.liveAnswerExtractor.reset()
        self._reasoningBuffer = ""
        self._answerBuffer = ""
        AIMEDiagnostics.record(
            "prime_from_snapshot",
            fields: diagnosticFields(extra: [
                "snapshot_current_idx": .int(snap.currentIdx),
                "snapshot_current_attempt": .int(snap.currentAttempt ?? 0),
                "snapshot_question_count": .int(snap.perQuestion.count)
            ])
        )
        // Rebuild results from the snapshot's per_question rows.
        self.results = snap.perQuestion.map { q in
            BenchQuestionResult(
                idx: q.idx,
                problem: BenchProblem(
                    id: q.id,
                    set: q.set,
                    year: snap.year,
                    index: q.idx <= 15 ? q.idx : q.idx - 15,
                    problem: "",  // snapshot doesn't include problem text
                    answer: q.expected,
                    source: ""
                ),
                status: QuestionStatus(rawValue: q.status ?? "") ?? .pending,
                extracted: q.extracted,
                startedAt: nil,
                endedAt: nil,
                reasoningTokenCount: q.reasoningTokenCount ?? 0,
                answerTokenCount: q.answerTokenCount ?? 0
            )
        }
    }

    private func applyCancelledSnapshot(_ snap: BenchSnapshotResponse) async {
        guard runID == nil || snap.runID == runID else { return }
        await primeFromSnapshot(snap)
        state = .cancelled
        endedAt = snap.endedAt ?? endedAt ?? Date()
        pausePending = false
    }

    private func refreshCancelledSnapshotSoon(runID: String) {
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard let self, self.runID == runID, self.state == .cancelled else { return }
            do {
                let snapshot = try await self.apiClientProvider().aimeSnapshot(runId: runID)
                await self.applyCancelledSnapshot(snapshot)
                await self.refreshHistory()
            } catch {
                // The runner may have been garbage-collected or the daemon
                // may already be down. Keep the immediate cancel snapshot.
            }
        }
    }

    private func refreshSkipSnapshotSoon(runID: String, skippedIdx: Int) {
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard let self,
                  self.runID == runID,
                  self.skipPending,
                  self.currentIdx >= skippedIdx,
                  self.state.isLive
            else { return }
            do {
                let snapshot = try await self.apiClientProvider().aimeSnapshot(runId: runID)
                let skippedResolved = snapshot.perQuestion.first { $0.idx == skippedIdx }?.status != nil
                guard skippedResolved || snapshot.currentIdx != skippedIdx else { return }
                await self.primeFromSnapshot(snapshot)
            } catch {
                // The SSE stream remains authoritative; this only rescues
                // stale UI if the skip response races ahead of events.
            }
        }
    }

    private func attachStream(
        runId: String,
        problems: [BenchProblem],
        existingResults: [BenchQuestionResult]?
    ) {
        let hadStreamTask = streamTask != nil
        let hadFlushTask = reasoningFlushTask != nil
        streamTask?.cancel()
        reasoningFlushTask?.cancel()
        streamGeneration += 1
        flushGeneration += 1
        let attachedStreamGeneration = streamGeneration
        let attachedFlushGeneration = flushGeneration
        AIMEDiagnostics.signpost(.streamEvent)
        AIMEDiagnostics.record(
            "stream_attach",
            fields: diagnosticFields(extra: [
                "attach_run_id": .string(runId),
                "had_stream_task": .bool(hadStreamTask),
                "had_flush_task": .bool(hadFlushTask),
                "stream_generation": .int(attachedStreamGeneration),
                "flush_generation": .int(attachedFlushGeneration)
            ])
        )

        let stream = streamFactory(apiClientProvider(), runId)
        streamTask = Task { [weak self] in
            await MainActor.run {
                AIMEDiagnostics.record(
                    "stream_task_started",
                    fields: self?.diagnosticFields(extra: [
                        "stream_generation": .int(attachedStreamGeneration),
                        "attach_run_id": .string(runId)
                    ]) ?? [:]
                )
            }
            defer {
                Task { @MainActor [weak self] in
                    AIMEDiagnostics.record(
                        "stream_task_ended",
                        fields: self?.diagnosticFields(extra: [
                            "stream_generation": .int(attachedStreamGeneration),
                            "attach_run_id": .string(runId)
                        ]) ?? [:]
                    )
                }
            }
            do {
                try await stream.consume { [weak self] event in
                    await self?.handleStreamEvent(event)
                }
            } catch is CancellationError {
                // Expected on reset.
                await MainActor.run {
                    AIMEDiagnostics.record(
                        "stream_task_cancelled",
                        fields: self?.diagnosticFields(extra: [
                            "stream_generation": .int(attachedStreamGeneration)
                        ]) ?? [:]
                    )
                }
            } catch BenchmarkStreamError.daemonUnreachable {
                await MainActor.run {
                    self?.lastError = tr("MTPLX is unreachable — is it running?")
                    self?.state = .error
                    AIMEDiagnostics.record(
                        "stream_task_daemon_unreachable",
                        fields: self?.diagnosticFields(extra: [
                            "stream_generation": .int(attachedStreamGeneration)
                        ]) ?? [:]
                    )
                }
            } catch BenchmarkStreamError.httpStatus(404, _) {
                // Run is gone (probably already finished + GC'd). Treat as done.
                await MainActor.run {
                    self?.state = .done
                    self?.endedAt = Date()
                    AIMEDiagnostics.record(
                        "stream_task_http_404",
                        fields: self?.diagnosticFields(extra: [
                            "stream_generation": .int(attachedStreamGeneration)
                        ]) ?? [:]
                    )
                }
            } catch {
                await MainActor.run {
                    guard let self else { return }
                    if self.state == .cancelled,
                        Self.isCancellationMessage(error.localizedDescription)
                    {
                        self.lastError = nil
                        return
                    }
                    self.lastError = tr("Stream error: %@", error.localizedDescription)
                    self.state = .error
                    AIMEDiagnostics.record(
                        "stream_task_failed",
                        fields: self.diagnosticFields(extra: [
                            "stream_generation": .int(attachedStreamGeneration),
                            "error_type": .string(String(describing: type(of: error)))
                        ])
                    )
                }
            }
        }

        // Flush task drains the reasoning + answer buffers at ~6 Hz so the
        // live panel reads smoothly without re-rendering on every SSE byte.
        reasoningFlushTask = Task { [weak self] in
            await MainActor.run {
                AIMEDiagnostics.record(
                    "flush_task_started",
                    fields: self?.diagnosticFields(extra: [
                        "flush_generation": .int(attachedFlushGeneration)
                    ]) ?? [:]
                )
            }
            defer {
                Task { @MainActor [weak self] in
                    AIMEDiagnostics.record(
                        "flush_task_ended",
                        fields: self?.diagnosticFields(extra: [
                            "flush_generation": .int(attachedFlushGeneration)
                        ]) ?? [:]
                    )
                }
            }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: Self.streamFlushIntervalNs)
                await self?.flushStreamBuffers()
            }
        }
    }

    /// Drains both streaming buffers into their `@Published` mirrors in a
    /// single coalesced update, then re-derives the live boxed answer.
    /// Called on the flush timer and synchronously before any terminal
    /// event reads `liveReasoning` / `liveAnswer`.
    private func flushStreamBuffers() async {
        let diagnosticsEnabled = AIMEDiagnostics.isEnabled
        let reasoningBytes = diagnosticsEnabled ? _reasoningBuffer.utf8.count : 0
        let answerBytes = diagnosticsEnabled ? _answerBuffer.utf8.count : 0
        let flushStarted = ProcessInfo.processInfo.systemUptime
        if diagnosticsEnabled && (reasoningBytes > 0 || answerBytes > 0) {
            AIMEDiagnostics.signpost(.bufferFlush)
        }
        var changed = false
        var reasoningAppendMs: Double = 0
        var answerAppendMs: Double = 0
        var answerExtractMs: Double = 0
        if !_reasoningBuffer.isEmpty {
            let appendStarted = ProcessInfo.processInfo.systemUptime
            liveReasoningDocument.append(_reasoningBuffer)
            reasoningAppendMs = (ProcessInfo.processInfo.systemUptime - appendStarted) * 1000
            _reasoningBuffer = ""
            changed = true
        }
        if !_answerBuffer.isEmpty {
            let answerDelta = _answerBuffer
            let appendStarted = ProcessInfo.processInfo.systemUptime
            liveAnswerDocument.append(answerDelta)
            answerAppendMs = (ProcessInfo.processInfo.systemUptime - appendStarted) * 1000
            let extractStarted = ProcessInfo.processInfo.systemUptime
            if let parsed = liveAnswerExtractor.append(answerDelta) {
                liveExtractedAnswer = parsed
            }
            answerExtractMs = (ProcessInfo.processInfo.systemUptime - extractStarted) * 1000
            _answerBuffer = ""
            changed = true
        }
        if changed && diagnosticsEnabled {
            let flushTotalMs = (ProcessInfo.processInfo.systemUptime - flushStarted) * 1000
            let shouldRecord = flushTotalMs >= 4 || AIMEDiagnostics.shouldRecordCadenced(
                "buffer_flush_finished",
                intervalS: 1,
                tokenCount: liveReasoningDocument.wordCount + liveAnswerDocument.wordCount,
                identity: currentRequestID
            )
            guard shouldRecord else { return }
            AIMEDiagnostics.record(
                "buffer_flush_finished",
                fields: diagnosticFields(extra: [
                    "flushed_reasoning_bytes": .int(reasoningBytes),
                    "flushed_answer_bytes": .int(answerBytes),
                    "boxed_answer_present": .bool(liveExtractedAnswer != nil),
                    "reasoning_append_ms": .double(reasoningAppendMs),
                    "answer_append_ms": .double(answerAppendMs),
                    "answer_extract_ms": .double(answerExtractMs),
                    "flush_total_ms": .double(flushTotalMs)
                ])
            )
        }
    }

    /// Internal so XCTest can drive a scripted event sequence without
    /// spinning up a real HTTP daemon. Tests call this through the
    /// `_testHandleStreamEvent` shim below.
    internal func handleStreamEvent(_ event: BenchEvent) async {
        if AIMEDiagnostics.isEnabled {
            switch event {
            case .reasoningDelta, .answerDelta:
                break
            default:
                AIMEDiagnostics.signpost(.streamEvent)
            }
        }
        switch event {
        case .runStarted(let id, let total, let model, let startedAt):
            self.runID = id
            self.model = model
            self.startedAt = startedAt ?? self.startedAt ?? Date()
            self.state = .running
            // Backfill results if a prior priming step undersized them.
            if results.isEmpty || results.count != total {
                await primeForRun(
                    runID: id,
                    model: model,
                    total: total,
                    startedAt: startedAt
                )
            }
            AIMEDiagnostics.record(
                "stream_run_started",
                fields: diagnosticFields(extra: [
                    "event_run_id": .string(id),
                    "event_total": .int(total),
                    "event_model": .string(model)
                ])
            )

        case .questionStarted(
            let idx,
            let attempt,
            let id,
            let set,
            let year,
            let problem
        ):
            self.currentIdx = idx
            self.currentAttempt = attempt
            self.currentRequestID = Self.requestID(
                runID: runID,
                idx: idx,
                attempt: attempt
            )
            self.currentVerification = nil
            self.liveReasoningDocument.reset()
            self.liveAnswerDocument.reset()
            self.liveTelemetry.reset()
            self.liveExtractedAnswer = nil
            self.liveAnswerIsGraded = false
            self.liveAnswerExtractor.reset()
            self.skipPending = false
            self._reasoningBuffer = ""
            self._answerBuffer = ""
            if let i = results.firstIndex(where: { $0.idx == idx }) {
                var row = results[i]
                row.problem = BenchProblem(
                    id: id,
                    set: set,
                    year: year,
                    index: idx <= 15 ? idx : idx - 15,
                    problem: problem,
                    answer: row.problem.answer,
                    source: row.problem.source.isEmpty
                        ? "https://artofproblemsolving.com/wiki/index.php?title=2026_\(set.replacingOccurrences(of: " ", with: "_"))_Problems"
                        : row.problem.source
                )
                row.startedAt = Date()
                row.status = .pending
                results[i] = row
            }
            AIMEDiagnostics.record(
                "stream_question_started",
                fields: diagnosticFields(extra: [
                    "event_idx": .int(idx),
                    "event_attempt": .int(attempt),
                    "event_problem_id": .string(id),
                    "event_year": .int(year),
                    "event_problem_bytes": .int(problem.utf8.count)
                ])
            )

        case .reasoningDelta(let idx, let attempt, let text):
            guard isCurrentAttempt(idx: idx, attempt: attempt) else {
                AIMEDiagnostics.record(
                    "stream_delta_dropped_stale",
                    fields: diagnosticFields(extra: [
                        "event_kind": .string("reasoning"),
                        "event_idx": .int(idx),
                        "event_attempt": .int(attempt),
                        "event_bytes": .int(text.utf8.count)
                    ])
                )
                return
            }
            _reasoningBuffer.append(text)

        case .answerDelta(let idx, let attempt, let text):
            guard isCurrentAttempt(idx: idx, attempt: attempt) else {
                AIMEDiagnostics.record(
                    "stream_delta_dropped_stale",
                    fields: diagnosticFields(extra: [
                        "event_kind": .string("answer"),
                        "event_idx": .int(idx),
                        "event_attempt": .int(attempt),
                        "event_bytes": .int(text.utf8.count)
                    ])
                )
                return
            }
            // Buffered like reasoning; the flush task drains it and re-runs
            // the boxed-answer extraction at ~6 Hz instead of per token.
            _answerBuffer.append(text)

        case .questionProgress(let idx, let attempt, let requestID, let metrics):
            guard isCurrentAttempt(idx: idx, attempt: attempt) else {
                AIMEDiagnostics.record(
                    "stream_progress_dropped_stale",
                    fields: diagnosticFields(extra: [
                        "event_idx": .int(idx),
                        "event_attempt": .int(attempt),
                        "event_request_id": .string(requestID)
                    ])
                )
                return
            }
            liveTelemetry.update(requestID: requestID, metrics: metrics)
            if AIMEDiagnostics.isEnabled {
                AIMEDiagnostics.record(
                    "stream_question_progress",
                    fields: diagnosticFields(extra: AIMEDiagnostics.metricFields(
                        from: metrics.values,
                        prefix: "progress_"
                    ))
                )
            }

        case .answerVerificationStarted(
            let idx,
            let attempt,
            let mode,
            let proposedAnswer
        ):
            guard isCurrentAttempt(idx: idx, attempt: attempt) else { return }
            currentVerification = BenchAnswerVerificationState(
                idx: idx,
                attempt: attempt,
                mode: mode,
                proposedAnswer: proposedAnswer,
                isRunning: true
            )

        case .answerVerificationDone(
            let idx,
            let attempt,
            let mode,
            let proposedAnswer,
            let verifiedAnswer,
            let verifierAnswers,
            let resolution,
            let durationMs
        ):
            guard isCurrentAttempt(idx: idx, attempt: attempt) else { return }
            currentVerification = BenchAnswerVerificationState(
                idx: idx,
                attempt: attempt,
                mode: mode,
                proposedAnswer: proposedAnswer,
                verifiedAnswer: verifiedAnswer,
                verifierAnswers: verifierAnswers,
                resolution: resolution,
                durationMs: durationMs,
                isRunning: false
            )

        case .capRecoveryStarted(let idx, let attempt, let requestID, _):
            guard isCurrentAttempt(idx: idx, attempt: attempt) else { return }
            currentRequestID = requestID
            currentVerification = nil

        case .questionDone(
            let idx,
            let attempt,
            _,
            let extracted,
            let expected,
            let status,
            let durationMs,
            let rTok,
            let aTok
        ):
            guard !isStaleAttempt(idx: idx, attempt: attempt) else {
                AIMEDiagnostics.record(
                    "stream_question_done_dropped_stale",
                    fields: diagnosticFields(extra: [
                        "event_idx": .int(idx),
                        "event_attempt": .int(attempt),
                        "reasoning_token_count": .int(rTok),
                        "answer_token_count": .int(aTok)
                    ])
                )
                return
            }
            await flushStreamBuffers()
            if idx == currentIdx {
                skipPending = false
            }
            if let i = results.firstIndex(where: { $0.idx == idx }) {
                var row = results[i]
                row.status = status
                row.extracted = extracted
                row.endedAt = Date()
                row.reasoningTokenCount = rTok
                row.answerTokenCount = aTok
                // Capture the live transcript for this question before the
                // next `questionStarted` wipes it, so the user can reopen
                // the tile and review how the model reached its answer.
                if idx == currentIdx {
                    row.reasoning = liveReasoning
                    row.answer = liveAnswer
                }
                if let durationMs, row.startedAt != nil, row.endedAt != nil {
                    // Trust server duration where available.
                    row.endedAt = row.startedAt!.addingTimeInterval(Double(durationMs) / 1000.0)
                }
                row.problem = BenchProblem(
                    id: row.problem.id,
                    set: row.problem.set,
                    year: row.problem.year,
                    index: row.problem.index,
                    problem: row.problem.problem,
                    answer: expected,
                    source: row.problem.source
                )
                results[i] = row
            }
            // Lock in the parsed answer.
            liveExtractedAnswer = extracted
            liveAnswerIsGraded = extracted != nil
            if let currentVerification, currentVerification.idx == idx {
                self.currentVerification = BenchAnswerVerificationState(
                    idx: currentVerification.idx,
                    attempt: currentVerification.attempt,
                    mode: currentVerification.mode,
                    proposedAnswer: currentVerification.proposedAnswer,
                    verifiedAnswer: extracted,
                    verifierAnswers: currentVerification.verifierAnswers,
                    resolution: currentVerification.resolution,
                    durationMs: currentVerification.durationMs,
                    isRunning: false
                )
            }
            AIMEDiagnostics.record(
                "stream_question_done",
                fields: diagnosticFields(extra: AIMEDiagnostics.fields(
                    ("event_idx", .int(idx)),
                    ("event_attempt", .int(attempt)),
                    ("status", .string(status.rawValue)),
                    ("duration_ms", AIMEDiagnostics.int(durationMs)),
                    ("reasoning_token_count", .int(rTok)),
                    ("answer_token_count", .int(aTok)),
                    ("extracted_present", .bool(extracted != nil))
                )),
                flushImmediately: true
            )

        case .runPaused:
            state = .paused
            pausePending = false
            AIMEDiagnostics.record("stream_run_paused", fields: diagnosticFields())

        case .runResumed:
            state = .running
            AIMEDiagnostics.record("stream_run_resumed", fields: diagnosticFields())

        case .runCancelled(let runID, let score, let total):
            state = .cancelled
            endedAt = Date()
            skipPending = false
            currentVerification = nil
            await flushStreamBuffers()
            AIMEDiagnostics.record(
                "stream_run_cancelled",
                fields: diagnosticFields(extra: [
                    "event_run_id": .string(runID),
                    "score": .int(score),
                    "total": .int(total)
                ]),
                flushImmediately: true
            )
            await refreshCancelledSnapshot(runID: runID)
            await refreshHistory()

        case .runDone(let runID, let score, let total, let accuracy, let durationMs):
            state = .done
            endedAt = Date()
            skipPending = false
            currentVerification = nil
            await flushStreamBuffers()
            AIMEDiagnostics.record(
                "stream_run_done",
                fields: diagnosticFields(extra: AIMEDiagnostics.fields(
                    ("event_run_id", .string(runID)),
                    ("score", .int(score)),
                    ("total", .int(total)),
                    ("accuracy", AIMEDiagnostics.double(accuracy)),
                    ("duration_ms", AIMEDiagnostics.int(durationMs))
                )),
                flushImmediately: true
            )
            await refreshHistory()

        case .error(let message, let recoverable):
            if Self.isCancellationMessage(message), state == .cancelled {
                lastError = nil
                endedAt = endedAt ?? Date()
                await flushStreamBuffers()
                AIMEDiagnostics.record(
                    "stream_cancellation_error_acknowledged",
                    fields: diagnosticFields(extra: [
                        "recoverable": .bool(recoverable),
                        "message_bytes": .int(message.utf8.count)
                    ]),
                    flushImmediately: true
                )
                if let runID {
                    await refreshCancelledSnapshot(runID: runID)
                    await refreshHistory()
                }
                return
            }
            state = .error
            endedAt = Date()
            skipPending = false
            lastError = message
            AIMEDiagnostics.record(
                "stream_error",
                fields: diagnosticFields(extra: [
                    "recoverable": .bool(recoverable),
                    "message_bytes": .int(message.utf8.count)
                ]),
                flushImmediately: true
            )

        case .keepAlive:
            AIMEDiagnostics.record("stream_keep_alive", fields: diagnosticFields())
            break
        }
    }

    private static func isCancellationMessage(_ message: String) -> Bool {
        let lowered = message.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return lowered == "cancelled" || lowered == "canceled"
    }

    private func isCurrentAttempt(idx: Int, attempt: Int) -> Bool {
        idx == currentIdx && (currentAttempt == 0 || attempt == currentAttempt)
    }

    private func isStaleAttempt(idx: Int, attempt: Int) -> Bool {
        idx == currentIdx && currentAttempt > 0 && attempt < currentAttempt
    }

    private static func requestID(
        runID: String?,
        idx: Int,
        attempt: Int
    ) -> String? {
        guard let runID, !runID.isEmpty, idx > 0, attempt > 0 else {
            return nil
        }
        return "chatcmpl-\(runID)-q\(idx)-a\(attempt)"
    }

    private func refreshCancelledSnapshot(runID: String) async {
        guard self.runID == runID else { return }
        do {
            let snapshot = try await apiClientProvider().aimeSnapshot(runId: runID)
            await applyCancelledSnapshot(snapshot)
        } catch {
            // The terminal SSE event is still enough to show cancellation;
            // snapshot reconciliation is best-effort for accurate score/grid.
        }
    }

    /// Test-only entry point so XCTest can drive a scripted SSE
    /// sequence without setting up an httpx daemon. Calls the real
    /// internal `handleStreamEvent` so behaviour matches production.
    @MainActor
    internal func _testHandleStreamEvent(_ event: BenchEvent) async {
        await handleStreamEvent(event)
    }
}

// MARK: - Stream abstraction (test seam)

/// Minimal protocol the orchestrator needs from a streaming source.
/// `BenchmarkStreamClient` satisfies this; tests inject a fake
/// implementation that yields canned `BenchEvent`s.
public protocol BenchmarkStreaming: Sendable {
    func consume(_ handler: @escaping @Sendable (BenchEvent) async -> Void) async throws
}

public struct DefaultBenchmarkStreaming: BenchmarkStreaming {
    private let client: BenchmarkStreamClient
    private let runId: String

    public init(client: BenchmarkStreamClient, runId: String) {
        self.client = client
        self.runId = runId
    }

    public func consume(_ handler: @escaping @Sendable (BenchEvent) async -> Void) async throws {
        try await client.connect(runId: runId, onEvent: handler)
    }
}
