import XCTest
@testable import MTPLXAppCore

// MARK: - RuntimeSetupServiceTests
//
// Exercises the onboarding "Setting up MTPLX" service with injected
// engine/fan-control/brew closures and fake executables under an
// isolated HOME — no real installs, no network, no real brew.

final class RuntimeSetupServiceTests: XCTestCase {
    private final class CallCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0

        func increment() {
            lock.lock()
            value += 1
            lock.unlock()
        }

        func count() -> Int {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    private struct SetupRun {
        var snapshots: [[RuntimeSetupRow]]
        var outcome: RuntimeSetupOutcome?

        func row(_ id: RuntimeSetupRowID) -> RuntimeSetupRow? {
            outcome?.rows.first { $0.id == id } ?? snapshots.last?.first { $0.id == id }
        }
    }

    private func run(_ service: RuntimeSetupService) async -> SetupRun {
        var snapshots: [[RuntimeSetupRow]] = []
        var outcome: RuntimeSetupOutcome?
        for await event in service.stream() {
            switch event {
            case .rows(let rows):
                snapshots.append(rows)
            case .finished(let finished):
                outcome = finished
            }
        }
        return SetupRun(snapshots: snapshots, outcome: outcome)
    }

    private func isolatedEnvironment(home: URL, pathDir: URL) -> [String: String] {
        [
            "HOME": home.path,
            "PATH": pathDir.path,
            "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
        ]
    }

    private func makeFakeCLI(in directory: URL, version: String) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("mtplx")
        try """
        #!/bin/sh
        if [ "$1" = "--version" ]; then
          echo "mtplx \(version) (\(version))"
          exit 0
        fi
        echo ok
        """.data(using: .utf8)!.write(to: url)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    private func assertTerminalWrapper(
        home: URL,
        engine: URL,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        for commandName in ["mtplx", "MTPLX"] {
            let shim = home.appendingPathComponent(".mtplx/bin/\(commandName)")
            XCTAssertNil(
                try? FileManager.default.destinationOfSymbolicLink(atPath: shim.path),
                "terminal command must be a wrapper, not a symlink",
                file: file,
                line: line
            )
            let wrapper = try String(contentsOf: shim, encoding: .utf8)
            XCTAssertTrue(
                wrapper.contains(
                    "export PYTHONPYCACHEPREFIX='\(home.path)/Library/Caches/MTPLX/PythonBytecode'"
                ),
                wrapper,
                file: file,
                line: line
            )
            XCTAssertTrue(
                wrapper.contains("exec '\(engine.path)' \"$@\""),
                wrapper,
                file: file,
                line: line
            )
            XCTAssertTrue(
                FileManager.default.isExecutableFile(atPath: shim.path),
                file: file,
                line: line
            )
        }
    }

    private func temporaryDirectory() -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("runtime-setup-tests-\(UUID().uuidString)", isDirectory: true)
    }

    private func fanControlOK() -> RuntimeSetupService.FanControlEnsurer {
        { _, status in
            status("Fan control ready")
            return FanControlSetupResult(ok: true, exitCode: 0, message: "Fan control ready")
        }
    }

    // MARK: Engine

    func testEngineFailureBlocksSetupAndLeavesLaterRowsPending() async throws {
        struct InstallError: LocalizedError {
            var errorDescription: String? { "Python 3.11 or newer was not found." }
        }
        let home = temporaryDirectory()
        let pathDir = home.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: pathDir, withIntermediateDirectories: true)

        let service = RuntimeSetupService(
            processEnvironment: isolatedEnvironment(home: home, pathDir: pathDir),
            appVersion: "1.0.0",
            engineInstaller: { _ in throw InstallError() },
            fanControlEnsurer: fanControlOK()
        )
        let result = await run(service)

        XCTAssertEqual(result.outcome?.engineReady, false)
        XCTAssertNil(result.outcome?.executablePath)
        XCTAssertEqual(result.row(.engine)?.state, .failed)
        XCTAssertEqual(result.row(.engine)?.detail, "Python 3.11 or newer was not found.")
        XCTAssertEqual(result.row(.fanControl)?.state, .pending)
        XCTAssertEqual(result.row(.globalCLI)?.state, .pending)
    }

    func testFanControlFailureDegradesToWarningAndSetupCompletes() async throws {
        let home = temporaryDirectory()
        let engineDir = home.appendingPathComponent("engine", isDirectory: true)
        let engine = try makeFakeCLI(in: engineDir, version: "1.0.0")
        let pathDir = home.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: pathDir, withIntermediateDirectories: true)

        let service = RuntimeSetupService(
            processEnvironment: isolatedEnvironment(home: home, pathDir: pathDir),
            appVersion: "1.0.0",
            engineInstaller: { status in
                status("Installing MTPLX runtime")
                return engine
            },
            fanControlEnsurer: { _, _ in
                FanControlSetupResult(ok: false, exitCode: 1, message: "No supported fan tool")
            }
        )
        let result = await run(service)

        XCTAssertEqual(result.outcome?.engineReady, true)
        XCTAssertEqual(result.outcome?.executablePath, engine.path)
        XCTAssertEqual(result.row(.engine)?.state, .done)
        XCTAssertEqual(result.row(.engine)?.detail, "MTPLX 1.0.0 ready")
        XCTAssertEqual(result.row(.fanControl)?.state, .warning)
        XCTAssertTrue(result.row(.fanControl)?.detail.contains("safe defaults") == true)
    }

    // MARK: Global CLI sync

    func testUpgradesOldHomebrewCLIExactlyOnce() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        let staleCLI = try makeFakeCLI(in: globalDir, version: "0.3.7")
        let upgraded = try makeFakeCLI(in: home.appendingPathComponent("brew-upgraded"), version: "1.0.0")
        _ = staleCLI

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "homebrew"
        let upgrades = CallCounter()

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK(),
            homebrewUpgrader: {
                upgrades.increment()
                return upgraded
            }
        )
        let result = await run(service)

        XCTAssertEqual(upgrades.count(), 1, "brew upgrade should run exactly once")
        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertTrue(
            result.row(.globalCLI)?.detail.contains("updated to 1.0.0") == true,
            result.row(.globalCLI)?.detail ?? "nil"
        )
        XCTAssertEqual(result.outcome?.engineReady, true)
    }

    func testHomebrewUpgradeFailureFallsBackToShim() async throws {
        struct BrewError: LocalizedError {
            var errorDescription: String? { "brew upgrade exited 1" }
        }
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        _ = try makeFakeCLI(in: globalDir, version: "0.3.7")

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "homebrew"

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK(),
            homebrewUpgrader: { throw BrewError() }
        )
        let result = await run(service)

        XCTAssertEqual(result.outcome?.engineReady, true, "CLI sync must never block setup")
        // brew failing must not leave the user on a stale CLI — the
        // shim shadows it so the terminal still serves the engine.
        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertNil(result.row(.globalCLI)?.command)
        try assertTerminalWrapper(home: home, engine: engine)
    }

    /// The app is not polite about stale CLIs: anything older than the
    /// app gets the shim put in front of it on PATH, automatically.
    /// The old install is shadowed, never modified.
    func testStalePipCLIIsUpdatedAutomaticallyViaShim() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        let stale = try makeFakeCLI(in: globalDir, version: "0.3.7")

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "pipLike"
        let upgrades = CallCounter()

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK(),
            homebrewUpgrader: {
                upgrades.increment()
                return engine
            }
        )
        let result = await run(service)

        XCTAssertEqual(upgrades.count(), 0, "pip installs never go through brew")
        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertNil(result.row(.globalCLI)?.command, "no manual command — the app already fixed it")
        XCTAssertEqual(result.outcome?.engineReady, true)

        try assertTerminalWrapper(home: home, engine: engine)
        let zshrc = try String(
            contentsOf: home.appendingPathComponent(".zshrc"),
            encoding: .utf8
        )
        XCTAssertTrue(zshrc.contains(".mtplx/bin"), "PATH line must put the shim in front")
        XCTAssertTrue(
            FileManager.default.isExecutableFile(atPath: stale.path),
            "the user's old CLI file is shadowed, not deleted"
        )
    }

    /// #292: a symlinked ~/.zshrc (dotfile-repo users) must survive the PATH
    /// append. The old atomic rewrite replaced the link with a plain file and
    /// silently detached the user's dotfiles.
    func testSymlinkedZshrcSurvivesPathAppend() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        _ = try makeFakeCLI(in: globalDir, version: "0.3.7")

        let fileManager = FileManager.default
        let dotfiles = home.appendingPathComponent("dotfiles", isDirectory: true)
        try fileManager.createDirectory(at: dotfiles, withIntermediateDirectories: true)
        let realZshrc = dotfiles.appendingPathComponent("zshrc")
        let userContent = "# user's own config\nalias ll='ls -la'\n"
        try userContent.write(to: realZshrc, atomically: true, encoding: .utf8)
        let zshrcLink = home.appendingPathComponent(".zshrc")
        try fileManager.createSymbolicLink(
            at: zshrcLink,
            withDestinationURL: realZshrc
        )

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "pipLike"
        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK(),
            homebrewUpgrader: { engine }
        )
        _ = await run(service)

        let destination = try? fileManager.destinationOfSymbolicLink(
            atPath: zshrcLink.path
        )
        XCTAssertEqual(
            destination,
            realZshrc.path,
            "~/.zshrc must still be the user's symlink, not a replacement file"
        )
        let target = try String(contentsOf: realZshrc, encoding: .utf8)
        XCTAssertTrue(target.contains("alias ll"), "user content preserved")
        XCTAssertTrue(
            target.contains(".mtplx/bin"),
            "PATH line written through the link into the real dotfile"
        )
    }

    /// The founder's edge case: a CLI newer than the app is the user's
    /// business — no shim, no downgrade, no nagging.
    func testNewerThanAppCLIIsLeftAlone() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        _ = try makeFakeCLI(in: globalDir, version: "1.1.0")

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "pipLike"

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        let result = await run(service)

        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: home.appendingPathComponent(".mtplx/bin/mtplx").path
            ),
            "newer CLI must not be shadowed"
        )
    }

    func testMissingGlobalCLIInstallsTerminalShimAndPATHLine() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let emptyDir = home.appendingPathComponent("empty-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)

        let service = RuntimeSetupService(
            processEnvironment: isolatedEnvironment(home: home, pathDir: emptyDir),
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        let result = await run(service)

        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertTrue(
            result.row(.globalCLI)?.detail.contains("Installed the mtplx command") == true,
            result.row(.globalCLI)?.detail ?? "nil"
        )
        XCTAssertEqual(result.outcome?.engineReady, true)

        try assertTerminalWrapper(home: home, engine: engine)
        let zshrc = try String(
            contentsOf: home.appendingPathComponent(".zshrc"),
            encoding: .utf8
        )
        XCTAssertTrue(zshrc.contains(#"export PATH="$HOME/.mtplx/bin:$PATH""#), zshrc)
    }

    /// A ~/.zshrc symlinked into a dotfiles repo (stow/chezmoi/yadm) must be
    /// written through, never replaced by a plain file that silently detaches
    /// it from version control (#292).
    func testZshrcSymlinkIsPreservedWhenAddingPATHLine() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let emptyDir = home.appendingPathComponent("empty-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        let dotfiles = home.appendingPathComponent("dotfiles", isDirectory: true)
        try FileManager.default.createDirectory(at: dotfiles, withIntermediateDirectories: true)
        let target = dotfiles.appendingPathComponent("zshrc")
        try "# dotfiles-managed\n".write(to: target, atomically: true, encoding: .utf8)
        let link = home.appendingPathComponent(".zshrc")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)

        let service = RuntimeSetupService(
            processEnvironment: isolatedEnvironment(home: home, pathDir: emptyDir),
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        let result = await run(service)
        XCTAssertEqual(result.outcome?.engineReady, true)

        let destination = try FileManager.default.destinationOfSymbolicLink(atPath: link.path)
        XCTAssertEqual(
            destination,
            target.path,
            ".zshrc must remain a symlink into the dotfiles repo"
        )
        let repoCopy = try String(contentsOf: target, encoding: .utf8)
        XCTAssertTrue(repoCopy.contains(".mtplx/bin"), "PATH line must land in the linked target")
        XCTAssertTrue(repoCopy.hasPrefix("# dotfiles-managed"), "existing content preserved")
    }

    func testTerminalShimInstallIsIdempotent() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let emptyDir = home.appendingPathComponent("empty-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        let shimDir = home.appendingPathComponent(".mtplx/bin", isDirectory: true)
        try FileManager.default.createDirectory(at: shimDir, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: shimDir.appendingPathComponent("mtplx"),
            withDestinationURL: engine
        )

        let service = RuntimeSetupService(
            processEnvironment: isolatedEnvironment(home: home, pathDir: emptyDir),
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        _ = await run(service)
        let second = await run(service)

        XCTAssertEqual(second.row(.globalCLI)?.state, .done)
        XCTAssertEqual(second.row(.globalCLI)?.detail, "mtplx command ready.")
        try assertTerminalWrapper(home: home, engine: engine)
        let preservedSymlinks = try FileManager.default.contentsOfDirectory(
            atPath: shimDir.path
        ).filter { $0.hasPrefix("mtplx.pre-wrapper-") }
        XCTAssertEqual(preservedSymlinks.count, 1, "the old symlink should be preserved exactly once")
        let zshrc = try String(
            contentsOf: home.appendingPathComponent(".zshrc"),
            encoding: .utf8
        )
        let occurrences = zshrc.components(separatedBy: ".mtplx/bin:").count - 1
        XCTAssertEqual(occurrences, 1, "PATH line must not be duplicated:\n\(zshrc)")
    }

    func testTerminalWrapperPinsCacheAndForwardsArguments() async throws {
        let home = temporaryDirectory()
        let engineDir = home.appendingPathComponent("engine", isDirectory: true)
        try FileManager.default.createDirectory(at: engineDir, withIntermediateDirectories: true)
        let engine = engineDir.appendingPathComponent("mtplx")
        let log = home.appendingPathComponent("wrapper.log")
        try """
        #!/bin/sh
        if [ "$1" = "--version" ]; then
          echo "mtplx 1.0.0 (1.0.0)"
          exit 0
        fi
        {
          printf '%s\n' "$PYTHONPYCACHEPREFIX"
          printf '%s\n' "$#"
          printf '%s\n' "$1"
          printf '%s\n' "$2"
        } > "$MTPLX_TEST_WRAPPER_LOG"
        """.write(to: engine, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: engine.path
        )
        let emptyDir = home.appendingPathComponent("empty-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        var environment = isolatedEnvironment(home: home, pathDir: emptyDir)
        environment["MTPLX_TEST_WRAPPER_LOG"] = log.path

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        _ = await run(service)

        let process = Process()
        process.executableURL = home.appendingPathComponent(".mtplx/bin/mtplx")
        process.arguments = ["alpha", "two words"]
        environment["PYTHONPYCACHEPREFIX"] = "/Applications/MTPLX.app/Contents/Resources/cache"
        process.environment = environment
        try process.run()
        process.waitUntilExit()

        XCTAssertEqual(process.terminationStatus, 0)
        XCTAssertEqual(
            try String(contentsOf: log, encoding: .utf8).split(separator: "\n").map(String.init),
            [
                home.appendingPathComponent("Library/Caches/MTPLX/PythonBytecode").path,
                "2",
                "alpha",
                "two words",
            ]
        )
    }

    func testCompletedUserLegacyShimMigratesOutsideOnboarding() throws {
        let home = temporaryDirectory()
        let environment = ["HOME": home.path]
        let runtimeBin = URL(
            fileURLWithPath: MTPLXCommandBuilder.appRuntimeBinDirectory(
                environment: environment
            ),
            isDirectory: true
        )
        let engine = try makeFakeCLI(in: runtimeBin, version: "1.0.0")
        let shimDir = home.appendingPathComponent(".mtplx/bin", isDirectory: true)
        try FileManager.default.createDirectory(at: shimDir, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: shimDir.appendingPathComponent("mtplx"),
            withDestinationURL: engine
        )

        XCTAssertTrue(
            try RuntimeSetupService.migrateLegacyTerminalShimIfNeeded(
                processEnvironment: environment
            )
        )
        try assertTerminalWrapper(home: home, engine: engine)
        XCTAssertFalse(
            try RuntimeSetupService.migrateLegacyTerminalShimIfNeeded(
                processEnvironment: environment
            ),
            "the normal-startup migration must be idempotent"
        )
        let preservedSymlinks = try FileManager.default.contentsOfDirectory(
            atPath: shimDir.path
        ).filter { $0.hasPrefix("mtplx.pre-wrapper-") }
        XCTAssertEqual(preservedSymlinks.count, 1)
    }

    func testCompletedUserMigrationLeavesCustomLauncherUntouched() throws {
        let home = temporaryDirectory()
        let custom = try makeFakeCLI(
            in: home.appendingPathComponent("custom-bin"),
            version: "9.9.9"
        )
        let shimDir = home.appendingPathComponent(".mtplx/bin", isDirectory: true)
        try FileManager.default.createDirectory(at: shimDir, withIntermediateDirectories: true)
        let shim = shimDir.appendingPathComponent("mtplx")
        try FileManager.default.createSymbolicLink(at: shim, withDestinationURL: custom)

        XCTAssertFalse(
            try RuntimeSetupService.migrateLegacyTerminalShimIfNeeded(
                processEnvironment: ["HOME": home.path]
            )
        )
        XCTAssertEqual(
            try FileManager.default.destinationOfSymbolicLink(atPath: shim.path),
            custom.path
        )
    }

    func testExistingBrewCLIIsNotShadowedByShim() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        _ = try makeFakeCLI(in: globalDir, version: "1.0.0")

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "homebrew"

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        _ = await run(service)

        XCTAssertFalse(
            FileManager.default.fileExists(atPath: home.appendingPathComponent(".mtplx/bin/mtplx").path),
            "An up-to-date CLI must never be shadowed by the shim"
        )
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: home.appendingPathComponent(".zshrc").path),
            "The shell profile must not be touched when the CLI is already current"
        )
    }

    func testUpToDateGlobalCLIIsLeftAlone() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        _ = try makeFakeCLI(in: globalDir, version: "1.0.0")

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "homebrew"
        let upgrades = CallCounter()

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK(),
            homebrewUpgrader: {
                upgrades.increment()
                return engine
            }
        )
        let result = await run(service)

        XCTAssertEqual(upgrades.count(), 0)
        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertTrue(
            result.row(.globalCLI)?.detail.contains("Up to date (1.0.0)") == true,
            result.row(.globalCLI)?.detail ?? "nil"
        )
    }

    func testRowsArePublishedInCanonicalOrder() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let emptyDir = home.appendingPathComponent("empty-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)

        let service = RuntimeSetupService(
            processEnvironment: isolatedEnvironment(home: home, pathDir: emptyDir),
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        let result = await run(service)

        for snapshot in result.snapshots {
            XCTAssertEqual(snapshot.map(\.id), RuntimeSetupRowID.allCases)
        }
        XCTAssertEqual(result.outcome?.rows.map(\.id), RuntimeSetupRowID.allCases)
    }
}
