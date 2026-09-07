import XCTest

@testable import MTPLXAppCore

/// The daemon's API key must never be visible on the process argument list
/// (every local process can read it through `ps`) or in the Logs pane the
/// supervisor fills with the launched command line. The key travels through
/// a user-only file and every logged argv goes through one redaction helper.
final class DaemonAPIKeyFileTests: XCTestCase {
    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-api-key-file-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeExecutable(named name: String) throws -> URL {
        let directory = temporaryDirectory()
        let url = directory.appendingPathComponent(name)
        try "#!/bin/sh\nexit 0\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    private func makeBuilder(home: URL) throws -> MTPLXCommandBuilder {
        let fake = try makeExecutable(named: "mtplx")
        return MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "HOME": home.path,
            "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
        ])
    }

    private func configuration(apiKey: String?) -> MTPLXAppConfiguration {
        MTPLXAppConfiguration(
            model: "/models/qwen",
            profile: "sustained",
            host: "127.0.0.1",
            port: 8123,
            apiKey: apiKey
        )
    }

    private func posixPermissions(of url: URL) throws -> Int {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return try XCTUnwrap(attributes[.posixPermissions] as? Int)
    }

    // MARK: - Builder contract

    func testServeCommandPassesTheKeyThroughAFileNeverOnArgv() throws {
        let home = temporaryDirectory()
        let builder = try makeBuilder(home: home)

        let command = try builder.buildServeCommand(configuration: configuration(apiKey: "sk-live-secret-123"))

        let expectedPath = home
            .appendingPathComponent("Library/Application Support/MTPLX/daemon-api-key")
            .path
        XCTAssertEqual(
            MTPLXCommandBuilder.flagValue("--api-key-file", in: command.arguments),
            expectedPath
        )
        XCTAssertEqual(
            MTPLXCommandBuilder.daemonAPIKeyFileURL(environment: builder.environment).path,
            expectedPath
        )
        XCTAssertFalse(command.arguments.contains("--api-key"), command.arguments.joined(separator: " "))
        XCTAssertFalse(
            command.arguments.contains { $0.contains("sk-live-secret-123") },
            command.arguments.joined(separator: " ")
        )
    }

    func testKeyFileIsUserOnlyAndHoldsExactlyTheKey() throws {
        let home = temporaryDirectory()
        let builder = try makeBuilder(home: home)

        _ = try builder.buildServeCommand(configuration: configuration(apiKey: "sk-live-secret-123"))

        let keyFile = MTPLXCommandBuilder.daemonAPIKeyFileURL(environment: builder.environment)
        XCTAssertEqual(try String(contentsOf: keyFile, encoding: .utf8), "sk-live-secret-123")
        XCTAssertEqual(try posixPermissions(of: keyFile), 0o600)
    }

    func testKeyFileIsRewrittenAndTightenedWhenTheKeyChanges() throws {
        let home = temporaryDirectory()
        let builder = try makeBuilder(home: home)
        let keyFile = MTPLXCommandBuilder.daemonAPIKeyFileURL(environment: builder.environment)

        _ = try builder.buildServeCommand(configuration: configuration(apiKey: "first-key"))
        XCTAssertEqual(try String(contentsOf: keyFile, encoding: .utf8), "first-key")

        // A file somebody loosened (or an older build left world-readable)
        // is tightened again on the next launch, and the new key replaces
        // the old one in full — no stale tail from a longer previous key.
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: keyFile.path)
        _ = try builder.buildServeCommand(configuration: configuration(apiKey: "second"))

        XCTAssertEqual(try String(contentsOf: keyFile, encoding: .utf8), "second")
        XCTAssertEqual(try posixPermissions(of: keyFile), 0o600)
    }

    func testNoConfiguredKeyEmitsNeitherKeyFlagNorKeyFile() throws {
        let home = temporaryDirectory()
        let builder = try makeBuilder(home: home)
        let keyFile = MTPLXCommandBuilder.daemonAPIKeyFileURL(environment: builder.environment)

        for apiKey in [nil, ""] {
            let command = try builder.buildServeCommand(configuration: configuration(apiKey: apiKey))
            XCTAssertFalse(command.arguments.contains("--api-key-file"), command.arguments.joined(separator: " "))
            XCTAssertFalse(command.arguments.contains("--api-key"), command.arguments.joined(separator: " "))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: keyFile.path))
    }

    // MARK: - Redaction helper

    func testRedactionMasksEverySecretFlagValueAndLeavesTheRestIntact() {
        let arguments = [
            "serve",
            "--model", "/models/qwen",
            "--api-key", "sk-live-secret-123",
            "--api-key-file", "/support/MTPLX/daemon-api-key",
            "--hf-token", "hf_abc",
            "--client-secret=shh",
            "--proxy-password", "pw",
            "--top-k", "20",
            "--port", "8123",
        ]

        XCTAssertEqual(
            DaemonCommand.redactingSecrets(in: arguments),
            [
                "serve",
                "--model", "/models/qwen",
                "--api-key", DaemonCommand.redactedValue,
                "--api-key-file", "/support/MTPLX/daemon-api-key",
                "--hf-token", DaemonCommand.redactedValue,
                "--client-secret=" + DaemonCommand.redactedValue,
                "--proxy-password", DaemonCommand.redactedValue,
                "--top-k", "20",
                "--port", "8123",
            ]
        )
    }

    func testSecretFlagDetectionCoversSuffixesAndSkipsPathsAndValues() {
        XCTAssertTrue(DaemonCommand.isSecretFlag("--api-key"))
        XCTAssertTrue(DaemonCommand.isSecretFlag("--API-KEY"))
        XCTAssertTrue(DaemonCommand.isSecretFlag("--api-key=abc"))
        XCTAssertTrue(DaemonCommand.isSecretFlag("--hf-token"))
        XCTAssertTrue(DaemonCommand.isSecretFlag("--some-secret"))
        XCTAssertTrue(DaemonCommand.isSecretFlag("--db-password"))
        XCTAssertFalse(DaemonCommand.isSecretFlag("--api-key-file"))
        XCTAssertFalse(DaemonCommand.isSecretFlag("--top-k"))
        XCTAssertFalse(DaemonCommand.isSecretFlag("--model"))
        XCTAssertFalse(DaemonCommand.isSecretFlag("sk-live-secret-123"))
        XCTAssertFalse(DaemonCommand.isSecretFlag("--"))
    }

    func testRedactedCommandLineJoinsExecutableAndMaskedArguments() {
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/usr/local/bin/mtplx"),
            arguments: ["serve", "--api-key", "sk-live-secret-123", "--port", "8000"]
        )

        XCTAssertEqual(
            command.redactedCommandLine,
            "/usr/local/bin/mtplx serve --api-key \(DaemonCommand.redactedValue) --port 8000"
        )
        XCTAssertEqual(
            command.redactedArguments,
            ["serve", "--api-key", DaemonCommand.redactedValue, "--port", "8000"]
        )
    }

    // MARK: - Supervisor log line

    func testLaunchedLogLineNeverCarriesASecretFlagValue() async throws {
        let supervisor = DaemonSupervisor()
        // A real child so the supervisor reaches its "launched" log line.
        // The extra arguments land in the shell's positional parameters and
        // are ignored; they exist only to be logged.
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "sleep 30", "sh", "--api-key", "sk-live-secret-123", "--port", "8000"]
        )

        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let launched = await supervisor.logs.snapshot()
            .first { $0.message.hasPrefix("launched ") }
        await supervisor.stop(graceSeconds: 0)

        let line = try XCTUnwrap(launched?.message)
        XCTAssertFalse(line.contains("sk-live-secret-123"), line)
        XCTAssertTrue(line.contains("--api-key \(DaemonCommand.redactedValue)"), line)
        XCTAssertTrue(line.contains("--port 8000"), line)
        XCTAssertTrue(line.contains("/bin/sh"), line)
    }
}
