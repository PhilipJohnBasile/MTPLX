import Foundation

// MARK: - TuneCandidate
//
// Identifies a model-family tune candidate. Qwen uses AR/D1/D2/D3;
// Gemma uses AR plus draft blocks 2...8. The file names map 1:1 to
// what `mtplx tune` writes so onboarding can show live per-candidate
// progress without guessing.

public enum TuneCandidate: String, CaseIterable, Equatable, Sendable {
    case ar
    case d1
    case d2
    case d3
    case block2
    case block3
    case block4
    case block5
    case block6
    case block7
    case block8

    public static let qwenCandidates: [TuneCandidate] = [.ar, .d1, .d2, .d3]
    public static let gemmaCandidates: [TuneCandidate] = [
        .ar,
        .block2,
        .block3,
        .block4,
        .block5,
        .block6,
        .block7,
        .block8,
    ]

    public static func candidates(forFamily family: String) -> [TuneCandidate] {
        switch family {
        case "qwen3_5", "qwen3_6":
            return qwenCandidates
        case "gemma4":
            return gemmaCandidates
        default:
            return []
        }
    }

    public var fileName: String {
        switch self {
        case .ar: return "ar.json"
        case .d1: return "d1.json"
        case .d2: return "d2.json"
        case .d3: return "d3.json"
        case .block2: return "block2.json"
        case .block3: return "block3.json"
        case .block4: return "block4.json"
        case .block5: return "block5.json"
        case .block6: return "block6.json"
        case .block7: return "block7.json"
        case .block8: return "block8.json"
        }
    }

    public var displayLabel: String {
        switch self {
        case .ar: return "Base speeds"
        case .d1: return "MTP 1"
        case .d2: return "MTP 2"
        case .d3: return "MTP 3"
        case .block2: return "Block 2"
        case .block3: return "Block 3"
        case .block4: return "Block 4"
        case .block5: return "Block 5"
        case .block6: return "Block 6"
        case .block7: return "Block 7"
        case .block8: return "Block 8"
        }
    }

    /// `mtplx tune --_candidate` accepts these exact strings.
    public var cliFlag: String {
        switch self {
        case .ar: return "ar"
        case .d1: return "1"
        case .d2: return "2"
        case .d3: return "3"
        case .block2: return "2"
        case .block3: return "3"
        case .block4: return "4"
        case .block5: return "5"
        case .block6: return "6"
        case .block7: return "7"
        case .block8: return "8"
        }
    }

    public var controlValue: Int {
        switch self {
        case .ar: return 0
        case .d1: return 1
        case .d2: return 2
        case .d3: return 3
        case .block2: return 2
        case .block3: return 3
        case .block4: return 4
        case .block5: return 5
        case .block6: return 6
        case .block7: return 7
        case .block8: return 8
        }
    }

    public var compactLabel: String {
        switch self {
        case .ar: return "Base"
        case .d1: return "MTP 1"
        case .d2: return "MTP 2"
        case .d3: return "MTP 3"
        case .block2: return "B2"
        case .block3: return "B3"
        case .block4: return "B4"
        case .block5: return "B5"
        case .block6: return "B6"
        case .block7: return "B7"
        case .block8: return "B8"
        }
    }
}

// MARK: - TuneCandidateResult
//
// Compact per-candidate summary parsed from `tune.json.results[]`.
// The full per-candidate file is rich (sampling probabilities, draft
// metrics, etc) but the UI only needs the four fields here.

public struct TuneCandidateResult: Equatable, Sendable {
    public var candidate: TuneCandidate
    public var tokS: Double
    public var multiplierVsAR: Double
    public var acceptanceByDepth: [Double]

    public init(candidate: TuneCandidate, tokS: Double, multiplierVsAR: Double, acceptanceByDepth: [Double]) {
        self.candidate = candidate
        self.tokS = tokS
        self.multiplierVsAR = multiplierVsAR
        self.acceptanceByDepth = acceptanceByDepth
    }
}

// MARK: - TuneResult
//
// Final verdict from a tune run. `bestDepth == 0` means AR won; any
// positive value is the selected control value for the family
// (`depth` for Qwen, `draft_block_size` for Gemma).

public struct TuneResult: Equatable, Sendable {
    public var bestCandidate: TuneCandidate
    public var bestDepth: Int          // 0 for AR, 1/2/3 for MTP depths
    public var bestTokS: Double
    public var bestMultiplierVsAR: Double
    public var allCandidates: [TuneCandidateResult]

    public init(
        bestCandidate: TuneCandidate,
        bestDepth: Int,
        bestTokS: Double,
        bestMultiplierVsAR: Double,
        allCandidates: [TuneCandidateResult]
    ) {
        self.bestCandidate = bestCandidate
        self.bestDepth = bestDepth
        self.bestTokS = bestTokS
        self.bestMultiplierVsAR = bestMultiplierVsAR
        self.allCandidates = allCandidates
    }
}

// MARK: - TuneEvent
//
// One frame from the auto-tune subprocess (orchestrator → view).

public enum TuneEvent: Sendable {
    /// Fan control is being installed before the benchmark starts.
    case installingFanControl(String)
    case started(runID: String, outputDir: String)
    /// Fired when `<output-dir>/<run-id>/<candidate>.json` first
    /// appears on disk. The view flips the corresponding checklist
    /// row from spinning to done and stamps the per-candidate tok/s.
    case candidateLanded(TuneCandidateResult)
    /// All four candidates finished and `tune.json` has been parsed.
    case completed(TuneResult)
    /// Subprocess exited non-zero before completing.
    case failed(exitCode: Int32?, stderrTail: String)
    /// Consumer cancelled.
    case cancelled
}

// MARK: - AutoTuner
//
// Shells `mtplx tune --model X --json --yes --output-dir TMP --run-id
// UUID` and surfaces per-candidate progress by polling the run dir.
// We don't trust the default `<cwd>/outputs/cli/tune/<run-id>/` path
// — passing explicit `--output-dir` + `--run-id` makes the poll path
// deterministic regardless of the subprocess's cwd.

public struct AutoTuner: Sendable {
    public init(
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment,
        pollInterval: TimeInterval = 0.75,
        preferDevelopmentWrapper: Bool = false
    ) {
        self.processEnvironment = processEnvironment
        self.pollInterval = pollInterval
        self.preferDevelopmentWrapper = preferDevelopmentWrapper
    }

    private let processEnvironment: [String: String]
    private let pollInterval: TimeInterval
    private let preferDevelopmentWrapper: Bool

    /// `mtplx tune` requires a CLI-visible fan-pinning helper for
    /// honest measurements. The ThermalForge app bundle alone is not
    /// enough; the CLI needs `~/.mtplx/bin/thermalforge` or an
    /// executable on PATH. Delegates to the shared `FanControlInstaller`
    /// (also used by the onboarding runtime-setup step).
    public func thermalHelperPresent() -> Bool {
        FanControlInstaller(processEnvironment: processEnvironment).helperPresent()
    }

    public func stream(
        modelPath: String,
        candidates: [TuneCandidate] = TuneCandidate.qwenCandidates
    ) -> AsyncStream<TuneEvent> {
        AsyncStream { continuation in
            let subprocess = SubprocessInterruptBox()
            let worker = Task.detached(priority: .userInitiated) {
                let runID = "mtplx-onb-" + UUID().uuidString.lowercased().prefix(8)
                let tmpRoot = URL(fileURLWithPath: NSTemporaryDirectory())
                    .appendingPathComponent("mtplx-onboarding-tune", isDirectory: true)
                let outputDir = tmpRoot
                let runDir = outputDir.appendingPathComponent(String(runID), isDirectory: true)
                try? FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)

                let executable: URL
                do {
                    executable = try self.resolveOrBootstrapMtplxExecutable { message in
                        continuation.yield(.installingFanControl(message))
                    }
                    continuation.yield(.installingFanControl("MTPLX runtime ready"))
                } catch {
                    let message = (error as? LocalizedError)?.errorDescription
                        ?? error.localizedDescription
                    continuation.yield(.failed(exitCode: nil, stderrTail: message))
                    continuation.finish()
                    return
                }

                let fanControl = FanControlInstaller(processEnvironment: self.processEnvironment)
                    .ensureReady(executable: executable, subprocess: subprocess) { message in
                        continuation.yield(.installingFanControl(message))
                    }
                if Task.isCancelled {
                    continuation.yield(.cancelled)
                    continuation.finish()
                    return
                }
                if !fanControl.ok {
                    // Fan pinning is a timing nicety, not a tune
                    // prerequisite: a real M5 Max user had onboarding
                    // die here when the helper could not verify a
                    // ramp. The CLI tune now degrades to auto fans on
                    // its own; surface the state and keep going.
                    continuation.yield(.installingFanControl(
                        "Fan control unavailable; tuning with fans on auto"
                    ))
                }

                let process = Process()
                process.executableURL = executable
                let profile = MTPLXCommandBuilder.recommendedProfile(
                    for: modelPath,
                    environment: self.processEnvironment
                )
                process.arguments = [
                    "tune",
                    "--model", modelPath,
                    "--profile", profile,
                    "--json",
                    "--yes",
                    "--retune",
                    "--output-dir", outputDir.path,
                    "--run-id", String(runID),
                ]
                process.environment = MTPLXCommandBuilder.appSubprocessEnvironment(
                    environment: processEnvironment
                )

                let errPipe = Pipe()
                let outPipe = Pipe()
                process.standardError = errPipe
                process.standardOutput = outPipe

                let stderrBuffer = StderrTailBuffer(capacity: 4096)
                let stdoutBuffer = StderrTailBuffer(capacity: 16_384)
                errPipe.fileHandleForReading.readabilityHandler = { handle in
                    let chunk = handle.availableData
                    if !chunk.isEmpty {
                        stderrBuffer.append(chunk)
                    }
                }
                outPipe.fileHandleForReading.readabilityHandler = { handle in
                    let chunk = handle.availableData
                    if !chunk.isEmpty {
                        stdoutBuffer.append(chunk)
                    }
                }

                do {
                    subprocess.set(process)
                    try process.run()
                } catch {
                    subprocess.clear(process)
                    continuation.yield(.failed(exitCode: nil, stderrTail: error.localizedDescription))
                    continuation.finish()
                    return
                }
                continuation.yield(.started(runID: String(runID), outputDir: outputDir.path))

                let pollInterval = self.pollInterval
                let pollTask = Task.detached(priority: .userInitiated) {
                    var seen: Set<TuneCandidate> = []
                    while !Task.isCancelled, process.isRunning {
                        try? await Task.sleep(nanoseconds: UInt64(pollInterval * 1_000_000_000))
                        for candidate in candidates where !seen.contains(candidate) {
                            let path = runDir.appendingPathComponent(candidate.fileName)
                            if FileManager.default.fileExists(atPath: path.path),
                               let result = Self.parseCandidate(at: path, candidate: candidate)
                            {
                                seen.insert(candidate)
                                continuation.yield(.candidateLanded(result))
                            }
                        }
                    }
                }

                process.waitUntilExit()
                subprocess.clear(process)
                errPipe.fileHandleForReading.readabilityHandler = nil
                outPipe.fileHandleForReading.readabilityHandler = nil
                pollTask.cancel()
                if process.terminationStatus == 0 {
                    let tunePath = runDir.appendingPathComponent("tune.json")
                    if let data = try? Data(contentsOf: tunePath),
                        let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                        let result = Self.parseFinal(payload: root, candidates: candidates)
                    {
                        continuation.yield(.completed(result))
                    } else {
                        continuation.yield(.failed(
                            exitCode: 0,
                            stderrTail: Self.missingTuneArtifactMessage(
                                tunePath: tunePath,
                                stdout: stdoutBuffer.snapshot(),
                                stderr: stderrBuffer.snapshot()
                            )
                        ))
                    }
                } else if process.terminationReason == .uncaughtSignal {
                    continuation.yield(.cancelled)
                } else {
                    let tunePath = runDir.appendingPathComponent("tune.json")
                    continuation.yield(.failed(
                        exitCode: process.terminationStatus,
                        stderrTail: Self.tuneFailureMessage(
                            tunePath: tunePath,
                            stdout: stdoutBuffer.snapshot(),
                            stderr: stderrBuffer.snapshot()
                        )
                    ))
                }
                continuation.finish()
            }

            continuation.onTermination = { @Sendable _ in
                worker.cancel()
                subprocess.interrupt()
            }
        }
    }

    // MARK: - JSON parsing

    private static func missingTuneArtifactMessage(
        tunePath: URL,
        stdout: String,
        stderr: String
    ) -> String {
        if let candidateMessage = candidateFailureMessage(tunePath: tunePath) {
            return candidateMessage
        }
        var parts = ["tune.json missing or malformed at \(tunePath.path)"]
        let trimmedStdout = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedStderr = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedStdout.isEmpty {
            parts.append("stdout: \(trimmedStdout)")
        }
        if !trimmedStderr.isEmpty {
            parts.append("stderr: \(trimmedStderr)")
        }
        return parts.joined(separator: "\n")
    }

    private static func tuneFailureMessage(tunePath: URL, stdout: String, stderr: String) -> String {
        if let candidateMessage = candidateFailureMessage(tunePath: tunePath) {
            return candidateMessage
        }
        let trimmedStderr = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedStderr.isEmpty { return trimmedStderr }
        let trimmedStdout = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedStdout.isEmpty { return trimmedStdout }
        return "Tuning failed before MTPLX could write results."
    }

    private static func candidateFailureMessage(tunePath: URL) -> String? {
        guard
            let data = try? Data(contentsOf: tunePath),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let results = root["results"] as? [[String: Any]]
        else { return nil }

        guard let failed = results.first(where: { entry in
            if let returnCode = entry["returncode"] as? Int, returnCode != 0 {
                return true
            }
            if let tokS = entry["tok_s"] as? Double, tokS > 0 {
                return false
            }
            return entry["error"] != nil
        }) else { return nil }

        var parts = ["MTPLX runtime failed while loading the model."]
        if let command = failed["command"] as? [String], let runtime = command.first {
            parts.append("Runtime: \(runtime)")
        }
        if let logPath = failed["stdout"] as? String,
           let diagnostic = diagnosticLine(fromLogAt: logPath) {
            parts.append(diagnostic)
        } else if let error = failed["error"] as? String,
                  !error.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            parts.append(error)
        }
        return parts.joined(separator: "\n")
    }

    /// Last Python exception in the log, with its message. Libraries
    /// like transformers raise errors whose message starts with "\n",
    /// so the line that matches the `Type:` prefix can be empty after
    /// the colon and the real explanation sits on the lines below —
    /// returning only the matched line rendered as a bare
    /// "ImportError:" on the Mac Mini and hid the actual cause.
    static func diagnosticLine(fromLogAt path: String) -> String? {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        // components (not split) so blank lines survive — a blank line
        // is what ends a multiline exception message.
        let lines = text.components(separatedBy: .newlines)
        let prefixes = [
            "ValueError:", "RuntimeError:", "ImportError:",
            "ModuleNotFoundError:", "FileNotFoundError:", "OSError:",
        ]
        for index in lines.indices.reversed() {
            let trimmed = lines[index].trimmingCharacters(in: .whitespacesAndNewlines)
            guard prefixes.contains(where: trimmed.hasPrefix) else { continue }
            var collected = [trimmed]
            for follower in lines[(index + 1)...].prefix(6) {
                let line = follower.trimmingCharacters(in: .whitespacesAndNewlines)
                if line.isEmpty { break }
                collected.append(line)
            }
            return collected.joined(separator: "\n")
        }
        return lines.reversed()
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty }
    }

    static func parseCandidate(at url: URL, candidate: TuneCandidate) -> TuneCandidateResult? {
        guard
            let data = try? Data(contentsOf: url),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        // Per-candidate file shape: depths[].rows[] for MTP candidates,
        // ar_rows[] for AR baseline. Extract the first row's tok_s +
        // acceptance.
        var tokS: Double = 0
        var acceptance: [Double] = []
        if candidate == .ar {
            if let arRows = root["ar_rows"] as? [[String: Any]], let row = arRows.first {
                tokS = (row["tok_s"] as? Double) ?? 0
            }
        } else {
            if let depths = root["depths"] as? [[String: Any]],
                let first = depths.first,
                let rows = first["rows"] as? [[String: Any]],
                let row = rows.first
            {
                tokS = (row["tok_s"] as? Double) ?? 0
                acceptance = (row["acceptance_by_depth"] as? [Double]) ?? []
            }
        }
        return TuneCandidateResult(
            candidate: candidate,
            tokS: tokS,
            multiplierVsAR: 0, // populated only in the final tune.json roll-up
            acceptanceByDepth: acceptance
        )
    }

    static func parseFinal(
        payload: [String: Any],
        candidates: [TuneCandidate] = TuneCandidate.qwenCandidates
    ) -> TuneResult? {
        var byCandidate: [TuneCandidate: TuneCandidateResult] = [:]
        let arTokS = ((payload["best_multiplier"] as? [String: Any])?["ar_tok_s"] as? Double) ?? 0
        if let results = payload["results"] as? [[String: Any]] {
            for entry in results {
                guard
                    let name = entry["candidate"] as? String,
                    let candidate = Self.candidateFor(name: name, candidates: candidates)
                else { continue }
                let tokS = (entry["tok_s"] as? Double) ?? 0
                let multiplier = (entry["multiplier_vs_ar"] as? Double)
                    ?? (arTokS > 0 ? tokS / arTokS : (candidate == .ar ? 1.0 : 0))
                let acceptance = (entry["acceptance_by_depth"] as? [Double]) ?? []
                byCandidate[candidate] = TuneCandidateResult(
                    candidate: candidate,
                    tokS: tokS,
                    multiplierVsAR: multiplier,
                    acceptanceByDepth: acceptance
                )
            }
        }
        let ordered = candidates.compactMap { byCandidate[$0] }
        guard !ordered.isEmpty else { return nil }

        let best = payload["best"] as? [String: Any]
        let bestTokS: Double
        let bestMultiplier: Double
        let bestDepth: Int
        let bestCandidate: TuneCandidate
        if let best {
            bestTokS = (best["tok_s"] as? Double) ?? 0
            bestMultiplier = (best["multiplier_vs_ar"] as? Double) ?? 1.0
            if let depth = best["depth"] as? Int, depth >= 1 {
                bestDepth = depth
                bestCandidate = candidates.first { $0.controlValue == depth && $0 != .ar }
                    ?? (depth == 1 ? .d1 : (depth == 2 ? .d2 : .d3))
            } else {
                bestDepth = 0
                bestCandidate = .ar
            }
        } else if let arResult = byCandidate[.ar] {
            // The backend can complete successfully with `"best": null`
            // when no MTP depth beats AR. That is a real tune verdict,
            // not a malformed payload.
            bestTokS = arResult.tokS
            bestMultiplier = arResult.multiplierVsAR > 0 ? arResult.multiplierVsAR : 1.0
            bestDepth = 0
            bestCandidate = .ar
        } else {
            return nil
        }

        return TuneResult(
            bestCandidate: bestCandidate,
            bestDepth: bestDepth,
            bestTokS: bestTokS,
            bestMultiplierVsAR: bestMultiplier,
            allCandidates: ordered
        )
    }

    private static func candidateFor(
        name: String,
        candidates: [TuneCandidate] = TuneCandidate.qwenCandidates
    ) -> TuneCandidate? {
        let normalized = Self.normalizedCandidateName(name)
        if ["ar", "base", "basespeed", "basespeeds", "baseline"].contains(normalized) {
            return .ar
        }
        if let exact = candidates.first(where: {
            Self.normalizedCandidateName($0.rawValue) == normalized
                || Self.normalizedCandidateName($0.compactLabel) == normalized
                || Self.normalizedCandidateName($0.displayLabel) == normalized
        }) {
            return exact
        }
        if normalized.hasPrefix("mtp"),
           let value = Int(normalized.dropFirst("mtp".count))
        {
            return candidates.first { $0.controlValue == value && $0.rawValue.hasPrefix("d") }
        }
        if normalized.hasPrefix("block"),
           let value = Int(normalized.dropFirst("block".count))
        {
            return candidates.first { $0.controlValue == value && $0.rawValue.hasPrefix("block") }
        }
        if normalized.hasPrefix("d"),
           let value = Int(normalized.dropFirst())
        {
            return candidates.first { $0.controlValue == value && $0.rawValue.hasPrefix("d") }
        }
        if let value = Int(normalized) {
            return candidates.first { $0.controlValue == value && $0 != .ar }
        }
        return nil
    }

    private static func normalizedCandidateName(_ name: String) -> String {
        name
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: " ", with: "")
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "-", with: "")
    }

    // MARK: - Executable resolution

    static func resolveMtplxExecutable(
        env: [String: String],
        preferDevelopmentWrapper: Bool = false
    ) throws -> URL {
        try MTPLXCommandBuilder.resolveInstalledExecutable(
            environment: env,
            allowDevelopmentWrapper: preferDevelopmentWrapper
        )
    }

    private func resolveOrBootstrapMtplxExecutable(
        status: @escaping @Sendable (String) -> Void
    ) throws -> URL {
        if preferDevelopmentWrapper {
            return try Self.resolveMtplxExecutable(
                env: processEnvironment,
                preferDevelopmentWrapper: preferDevelopmentWrapper
            )
        }
        return try MTPLXRuntimeBootstrapper(environment: processEnvironment).installOrUpdate(status: status)
    }
}

// MARK: - Shared StderrTailBuffer
//
// Same shape as the one in `ModelDownloader.swift` — declared here so
// the file is self-contained but kept fileprivate to avoid clashing
// with the downloader's version.

private final class StderrTailBuffer: @unchecked Sendable {
    private let capacity: Int
    private var buffer = Data()
    private let lock = NSLock()

    init(capacity: Int) {
        self.capacity = capacity
    }

    func append(_ chunk: Data) {
        lock.lock()
        defer { lock.unlock() }
        buffer.append(chunk)
        if buffer.count > capacity {
            buffer.removeFirst(buffer.count - capacity)
        }
    }

    func snapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        return String(data: buffer, encoding: .utf8) ?? ""
    }
}
