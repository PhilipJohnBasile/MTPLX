import Foundation

// MARK: - HFPublishEvent

public enum HFPublishEvent: Sendable {
    case started(repo: String)
    case progress(bytesUploaded: Int64, totalBytes: Int64?, mbPerSecond: Double)
    case repoCreated(repo: String, revision: String?)
    case completed(repo: String, revision: String?)
    case failed(exitCode: Int32?, stderrTail: String)
    case cancelled
    case backendNotAvailable
}

// MARK: - HFPublisher
//
// Wraps `mtplx forge publish` (or its eventual real name). Same
// pattern as ForgeBuilder / ModelDownloader: shell the subprocess
// with an explicit --out / --run-id, poll the `publish.json` file
// for progress, surface stderr-tail on failure.
//
// Token is handed to the subprocess via the `--token` flag's
// `stdin` sentinel + a write to stdin so the token never leaks
// into the process listing.

public struct HFPublisher: Sendable {
    public init(
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment,
        pollInterval: TimeInterval = 0.5
    ) {
        self.processEnvironment = processEnvironment
        self.pollInterval = pollInterval
    }

    private let processEnvironment: [String: String]
    private let pollInterval: TimeInterval

    public struct PublishRequest: Equatable, Sendable {
        public var localPath: String
        public var repo: String
        public var visibility: ForgePublishOptions.Visibility
        public var licenseSpdx: String
        public var readmeBody: String?
        public var outputDir: String
        public var runID: String

        public init(
            localPath: String,
            repo: String,
            visibility: ForgePublishOptions.Visibility,
            licenseSpdx: String,
            readmeBody: String? = nil,
            outputDir: String,
            runID: String
        ) {
            self.localPath = localPath
            self.repo = repo
            self.visibility = visibility
            self.licenseSpdx = licenseSpdx
            self.readmeBody = readmeBody
            self.outputDir = outputDir
            self.runID = runID
        }
    }

    public func stream(_ request: PublishRequest, token: String) -> AsyncStream<HFPublishEvent> {
        AsyncStream<HFPublishEvent>(HFPublishEvent.self, bufferingPolicy: .unbounded) { continuation in
            let outputDir = URL(fileURLWithPath: request.outputDir, isDirectory: true)
            let runDir = outputDir.appendingPathComponent(request.runID, isDirectory: true)
            try? FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)

            let executable: URL
            do {
                executable = try ForgeBuilder.resolveMtplxExecutable(env: processEnvironment)
            } catch {
                continuation.yield(.backendNotAvailable)
                continuation.finish()
                return
            }

            var arguments: [String] = [
                "forge", "publish",
                "--path", request.localPath,
                "--repo", request.repo,
                "--visibility", request.visibility.rawValue,
                "--license", request.licenseSpdx,
                "--out", outputDir.path,
                "--run-id", request.runID,
                "--token", "stdin"
            ]
            if let readme = request.readmeBody, !readme.isEmpty {
                let readmePath = runDir.appendingPathComponent("README.md").path
                try? readme.write(toFile: readmePath, atomically: true, encoding: .utf8)
                arguments.append("--readme-path")
                arguments.append(readmePath)
            }

            let process = Process()
            process.executableURL = executable
            process.arguments = arguments
            process.environment = MTPLXCommandBuilder.pythonBytecodeSafeEnvironment(
                environment: processEnvironment
            )

            let stdinPipe = Pipe()
            let errPipe = Pipe()
            let outPipe = Pipe()
            process.standardInput = stdinPipe
            process.standardError = errPipe
            process.standardOutput = outPipe

            let stderrBuffer = HFPublishStderrTail(capacity: 4096)
            errPipe.fileHandleForReading.readabilityHandler = { handle in
                let chunk = handle.availableData
                if !chunk.isEmpty { stderrBuffer.append(chunk) }
            }
            outPipe.fileHandleForReading.readabilityHandler = { handle in
                _ = handle.availableData
            }

            do {
                try process.run()
            } catch {
                continuation.yield(.failed(exitCode: nil, stderrTail: error.localizedDescription))
                continuation.finish()
                return
            }

            // Write the token to stdin then close — the backend reads
            // exactly one line as the token value. Never logged.
            let tokenLine = (token + "\n").data(using: .utf8) ?? Data()
            do {
                try stdinPipe.fileHandleForWriting.write(contentsOf: tokenLine)
                try stdinPipe.fileHandleForWriting.close()
            } catch {
                process.terminate()
                continuation.yield(.failed(exitCode: nil, stderrTail: "Failed to hand token to subprocess: \(error.localizedDescription)"))
                continuation.finish()
                return
            }

            continuation.yield(.started(repo: request.repo))

            let pollInterval = self.pollInterval
            let pollTask = Task.detached(priority: .userInitiated) {
                var lastBytes: Int64 = -1
                var emittedRepoCreated = false
                let yieldEvent: @Sendable (HFPublishEvent) -> Void = { event in
                    continuation.yield(event)
                }
                while !Task.isCancelled, process.isRunning {
                    try? await Task.sleep(nanoseconds: UInt64(pollInterval * 1_000_000_000))
                    Self.scan(
                        runDir: runDir,
                        request: request,
                        lastBytes: &lastBytes,
                        emittedRepoCreated: &emittedRepoCreated,
                        yield: yieldEvent
                    )
                }
                Self.scan(
                    runDir: runDir,
                    request: request,
                    lastBytes: &lastBytes,
                    emittedRepoCreated: &emittedRepoCreated,
                    yield: yieldEvent
                )
            }

            Task.detached(priority: .userInitiated) {
                process.waitUntilExit()
                errPipe.fileHandleForReading.readabilityHandler = nil
                outPipe.fileHandleForReading.readabilityHandler = nil
                pollTask.cancel()
                if process.terminationStatus == 0 {
                    let finalJSON = runDir.appendingPathComponent("publish.json")
                    if let data = try? Data(contentsOf: finalJSON),
                       let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                    {
                        let repo = (root["repo"] as? String) ?? request.repo
                        let revision = root["revision"] as? String
                        continuation.yield(.completed(repo: repo, revision: revision))
                    } else {
                        continuation.yield(.completed(repo: request.repo, revision: nil))
                    }
                } else if process.terminationReason == .uncaughtSignal {
                    continuation.yield(.cancelled)
                } else {
                    let tail = stderrBuffer.snapshot()
                    let invalidChoice = tail.range(of: "invalid choice", options: .caseInsensitive) != nil
                        && tail.range(of: "forge", options: .caseInsensitive) != nil
                    if process.terminationStatus == 2 && invalidChoice {
                        continuation.yield(.backendNotAvailable)
                    } else {
                        continuation.yield(.failed(exitCode: process.terminationStatus, stderrTail: tail))
                    }
                }
                continuation.finish()
            }

            continuation.onTermination = { @Sendable _ in
                // Escalating cancel (#158 sweep): SIGINT alone left a
                // SIGINT-deaf publish running invisibly after the user
                // cancelled, still uploading on their connection.
                SubprocessWatchdog.escalateCancel(process)
                pollTask.cancel()
            }
        }
    }

    private static func scan(
        runDir: URL,
        request: PublishRequest,
        lastBytes: inout Int64,
        emittedRepoCreated: inout Bool,
        yield: @Sendable (HFPublishEvent) -> Void
    ) {
        let path = runDir.appendingPathComponent("publish.json")
        guard let data = try? Data(contentsOf: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }

        if !emittedRepoCreated, let createdRepo = json["repo_created"] as? String {
            emittedRepoCreated = true
            yield(.repoCreated(repo: createdRepo, revision: json["revision"] as? String))
        }
        let bytes = (json["bytes_uploaded"] as? Int).map(Int64.init)
            ?? Int64(json["bytes_uploaded"] as? Double ?? 0)
        let total = (json["total_bytes"] as? Int).map(Int64.init)
            ?? Int64(json["total_bytes"] as? Double ?? 0)
        let mbps = (json["mb_per_s"] as? Double) ?? 0
        if bytes != lastBytes {
            lastBytes = bytes
            yield(.progress(
                bytesUploaded: bytes,
                totalBytes: total > 0 ? total : nil,
                mbPerSecond: mbps
            ))
        }
    }
}

private final class HFPublishStderrTail: @unchecked Sendable {
    private let capacity: Int
    private var buffer = Data()
    private let lock = NSLock()

    init(capacity: Int) { self.capacity = capacity }

    func append(_ chunk: Data) {
        lock.lock(); defer { lock.unlock() }
        buffer.append(chunk)
        if buffer.count > capacity {
            buffer.removeFirst(buffer.count - capacity)
        }
    }

    func snapshot() -> String {
        lock.lock(); defer { lock.unlock() }
        return String(data: buffer, encoding: .utf8) ?? ""
    }
}
