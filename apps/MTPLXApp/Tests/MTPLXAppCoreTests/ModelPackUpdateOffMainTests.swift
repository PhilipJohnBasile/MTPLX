import XCTest

@testable import MTPLXAppCore

/// The model-pack Update button must not run its runtime resolution,
/// directory walks and CLI spawn on the main actor: those happen
/// synchronously inside `AsyncStream`'s build closure, and on the main
/// thread they froze the window for the Python cold start or a venv
/// reinstall.
final class ModelPackUpdateOffMainTests: XCTestCase {
    private final class ThreadRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var observations: [Bool] = []

        func record(isMainThread: Bool) {
            lock.withLock { observations.append(isMainThread) }
        }

        var snapshot: [Bool] {
            lock.withLock { observations }
        }
    }

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-pack-update-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeExecutable(named name: String, body: String) throws -> URL {
        let directory = temporaryDirectory()
        let url = directory.appendingPathComponent(name)
        try body.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    @MainActor
    func testModelPackUpdateBuildsItsStreamOffTheMainThread() async throws {
        let home = temporaryDirectory()
        let pack = home.appendingPathComponent("Youssofal--Pack", isDirectory: true)
        try FileManager.default.createDirectory(at: pack, withIntermediateDirectories: true)
        try Data(repeating: 0, count: 64).write(to: pack.appendingPathComponent("config.json"))
        // A CLI stand-in that exits cleanly: the stream reports completion
        // from the exit status alone.
        let fakeCLI = try makeExecutable(named: "mtplx", body: "#!/bin/sh\nexit 0\n")
        let recorder = ThreadRecorder()
        var downloader = ModelDownloader(
            processEnvironment: [
                "HOME": home.path,
                "PATH": fakeCLI.deletingLastPathComponent().path,
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
            ],
            executableOverride: fakeCLI
        )
        downloader.streamBuildObserver = { recorder.record(isMainThread: Thread.isMainThread) }
        let store = MTPLXBackendStore(
            modelDownloader: downloader,
            modelUpdateChecker: { [] }
        )
        let update = ModelUpdateInfo(
            repoID: "Youssofal/Pack",
            path: pack.path,
            state: "update-available",
            updateBytes: 1024
        )

        store.updateModelPack(update)
        XCTAssertEqual(store.modelPackUpdatingRepoID, "Youssofal/Pack")

        let deadline = ContinuousClock.now + .seconds(15)
        while store.modelPackUpdatingRepoID != nil, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }

        XCTAssertNil(store.modelPackUpdatingRepoID, "the pack update did not finish")
        XCTAssertNil(store.modelPackUpdateStatus)
        XCTAssertEqual(recorder.snapshot, [false], "the stream must be built exactly once, off the main thread")
    }
}
