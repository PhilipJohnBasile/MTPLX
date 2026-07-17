import Foundation

public struct MTPLXSemanticVersion: Comparable, Codable, Equatable, Sendable, CustomStringConvertible {
    public let components: [Int]

    public init?(_ raw: String) {
        guard let token = Self.firstVersionToken(in: raw) else { return nil }
        let parts = token.split(separator: ".").compactMap { Int($0) }
        guard parts.count >= 2 else { return nil }
        self.components = parts
    }

    public var description: String {
        components.map(String.init).joined(separator: ".")
    }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        let count = max(lhs.components.count, rhs.components.count)
        for index in 0..<count {
            let left = index < lhs.components.count ? lhs.components[index] : 0
            let right = index < rhs.components.count ? rhs.components[index] : 0
            if left != right { return left < right }
        }
        return false
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        !(lhs < rhs) && !(rhs < lhs)
    }

    private static func firstVersionToken(in raw: String) -> String? {
        let separators = CharacterSet.alphanumerics
            .union(CharacterSet(charactersIn: ".-+"))
            .inverted
        for token in raw.components(separatedBy: separators) {
            var candidate = token.trimmingCharacters(in: CharacterSet(charactersIn: "vV"))
            if let suffixIndex = candidate.firstIndex(where: { !($0.isNumber || $0 == ".") }) {
                candidate = String(candidate[..<suffixIndex])
            }
            let dotCount = candidate.filter { $0 == "." }.count
            if dotCount >= 1,
               candidate.split(separator: ".").allSatisfy({ Int($0) != nil }) {
                return candidate
            }
        }
        return nil
    }
}

public struct MTPLXReleaseManifest: Decodable, Equatable, Sendable {
    public var appVersion: String
    public var appBuild: String
    public var minimumCLIVersion: String
    public var recommendedCLIVersion: String
    public var dmgURL: URL
    public var dmgSHA256: String
    public var pypiVersion: String
    public var homebrewFormulaVersion: String
    public var releaseNotesURL: URL
    public var publishedAt: Date?

    enum CodingKeys: String, CodingKey {
        case appVersion = "app_version"
        case appBuild = "app_build"
        case minimumCLIVersion = "minimum_cli_version"
        case recommendedCLIVersion = "recommended_cli_version"
        case dmgURL = "dmg_url"
        case dmgSHA256 = "dmg_sha256"
        case pypiVersion = "pypi_version"
        case homebrewFormulaVersion = "homebrew_formula_version"
        case releaseNotesURL = "release_notes_url"
        case publishedAt = "published_at"
    }

    public static func decode(_ data: Data) throws -> Self {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(Self.self, from: data)
    }
}

public enum MTPLXRuntimeInstallKind: String, Equatable, Sendable {
    case appOwned
    case homebrew
    case sourceCheckout
    case pipLike
    case custom
    case missing

    public var displayName: String {
        switch self {
        case .appOwned: return "App-managed"
        case .homebrew: return "Homebrew"
        case .sourceCheckout: return "Source checkout"
        case .pipLike: return "Python"
        case .custom: return "Custom"
        case .missing: return "Missing"
        }
    }
}

public enum MTPLXRuntimeUpdateAction: Equatable, Sendable {
    case useExisting
    case installHomebrew
    case updateBundledRequired
    case updateHomebrewRequired
    case updateHomebrewRecommended
    case manualUpdateRequired(command: String)
    case homebrewRequired
}

public struct MTPLXRuntimeUpdateSnapshot: Equatable, Sendable {
    public var appVersion: String
    public var appBuild: String
    public var cliVersion: String?
    public var cliPath: String?
    public var cliInstallKind: MTPLXRuntimeInstallKind
    public var latestAppVersion: String?
    public var minimumCLIVersion: String?
    public var recommendedCLIVersion: String?
    public var action: MTPLXRuntimeUpdateAction
    public var title: String
    public var detail: String

    public var canUpdateRuntime: Bool {
        switch action {
        case .installHomebrew, .updateBundledRequired, .updateHomebrewRequired, .updateHomebrewRecommended:
            return true
        default:
            return false
        }
    }
}

public enum MTPLXRuntimeUpdateError: Error, LocalizedError, Equatable, Sendable {
    case manualUpdateRequired(command: String)
    case homebrewRequired

    public var errorDescription: String? {
        switch self {
        case .manualUpdateRequired(let command):
            return "This MTPLX runtime is too old, but it is not managed by Homebrew. Update it manually, then press Retry: \(command)"
        case .homebrewRequired:
            return "MTPLX runtime is missing and Homebrew was not found. Install Homebrew from brew.sh, then press Retry."
        }
    }
}

public struct MTPLXRuntimeUpdateService: Sendable {
    public static let defaultManifestURL = URL(string: "https://mtplx.com/releases/latest.json")!

    public var manifestURL: URL
    public var environment: [String: String]

    public init(
        manifestURL: URL = Self.defaultManifestURL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.manifestURL = manifestURL
        self.environment = environment
    }

    public func fetchManifest() async throws -> MTPLXReleaseManifest {
        let (data, response) = try await URLSession.shared.data(from: manifestURL)
        if let http = response as? HTTPURLResponse,
           !(200..<300).contains(http.statusCode) {
            throw URLError(.badServerResponse)
        }
        return try MTPLXReleaseManifest.decode(data)
    }

    public func snapshot(manifest: MTPLXReleaseManifest? = nil) -> MTPLXRuntimeUpdateSnapshot {
        Self.snapshot(manifest: manifest, environment: environment)
    }

    public func refreshSnapshot() async -> MTPLXRuntimeUpdateSnapshot {
        let manifest = try? await fetchManifest()
        return snapshot(manifest: manifest)
    }

    public func prepareRuntimeForLaunch() async throws -> URL {
        let manifest = try? await fetchManifest()
        guard let manifest else {
            return try MTPLXRuntimeBootstrapper(environment: environment).installOrUpdate()
        }

        if let existing = try? MTPLXCommandBuilder.resolveInstalledExecutable(environment: environment) {
            let kind = Self.installKind(for: existing, environment: environment)
            if kind == .appOwned {
                // The app-owned venv tracks the bundled wheel, not the
                // published manifest: installOrUpdate is a fast no-op when
                // the venv already satisfies the app's version floor and
                // reinstalls from the wheel when the app moved ahead of it
                // (e.g. right after a Sparkle update).
                return try MTPLXRuntimeBootstrapper(environment: environment).installOrUpdate()
            }
            // Whenever this bundle ships a wheel, the daemon runtime is
            // the app-owned venv, full stop. A user-managed CLI on PATH
            // that happens to satisfy the manifest (a stale dev pip
            // install reporting the release version, as on the founder's
            // Mac Mini) is the user's tool, not the engine — its Python
            // and dependency state are unknown. installOrUpdate is a
            // fast no-op once the venv matches the bundled wheel.
            if MTPLXCommandBuilder.bundledRuntimeWheelPath(environment: environment) != nil {
                return try MTPLXRuntimeBootstrapper(environment: environment).installOrUpdate()
            }
            let version = Self.runtimeVersion(executableURL: existing, environment: environment)
            let action = Self.action(
                version: version,
                installKind: kind,
                manifest: manifest,
                hasHomebrew: MTPLXCommandBuilder.resolveHomebrewExecutable(environment: environment) != nil
            )
            switch action {
            case .updateBundledRequired:
                return try MTPLXRuntimeBootstrapper(environment: environment).installOrUpdate()
            case .updateHomebrewRequired:
                if MTPLXCommandBuilder.bundledRuntimeWheelPath(environment: environment) != nil {
                    return try MTPLXRuntimeBootstrapper(environment: environment).installOrUpdate()
                }
                return try MTPLXRuntimeBootstrapper(environment: environment).upgradeHomebrewRuntime()
            case .manualUpdateRequired(let command):
                // A stale pip/source CLI on PATH must never block the app
                // when the bundled wheel can install an app-owned runtime
                // beside it instead.
                if MTPLXCommandBuilder.bundledRuntimeWheelPath(environment: environment) != nil {
                    return try MTPLXRuntimeBootstrapper(environment: environment).installOrUpdate()
                }
                throw MTPLXRuntimeUpdateError.manualUpdateRequired(command: command)
            default:
                return existing
            }
        }

        if MTPLXCommandBuilder.bundledRuntimeWheelPath(environment: environment) == nil,
           MTPLXCommandBuilder.resolveHomebrewExecutable(environment: environment) == nil {
            throw MTPLXRuntimeUpdateError.homebrewRequired
        }
        return try MTPLXRuntimeBootstrapper(environment: environment).installOrUpdate()
    }

    public static func snapshot(
        manifest: MTPLXReleaseManifest?,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> MTPLXRuntimeUpdateSnapshot {
        let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown"
        let appBuild = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "unknown"
        let brewAvailable = MTPLXCommandBuilder.resolveHomebrewExecutable(environment: environment) != nil

        guard let executable = try? MTPLXCommandBuilder.resolveInstalledExecutable(environment: environment) else {
            let wheelAvailable = MTPLXCommandBuilder.bundledRuntimeWheelPath(environment: environment) != nil
            let action: MTPLXRuntimeUpdateAction = wheelAvailable
                ? .updateBundledRequired
                : (brewAvailable ? .installHomebrew : .homebrewRequired)
            return MTPLXRuntimeUpdateSnapshot(
                appVersion: appVersion,
                appBuild: appBuild,
                cliVersion: nil,
                cliPath: nil,
                cliInstallKind: .missing,
                latestAppVersion: manifest?.appVersion,
                minimumCLIVersion: manifest?.minimumCLIVersion,
                recommendedCLIVersion: manifest?.recommendedCLIVersion,
                action: action,
                title: "Runtime missing",
                detail: wheelAvailable
                    ? "MTPLX can install its bundled runtime automatically."
                    : (brewAvailable
                        ? "MTPLX can install the command-line runtime with Homebrew."
                        : "Install Homebrew from brew.sh, then press Retry.")
            )
        }

        let version = runtimeVersion(executableURL: executable, environment: environment)
        let kind = installKind(for: executable, environment: environment)
        let action = manifest.map {
            Self.action(
                version: version,
                installKind: kind,
                manifest: $0,
                hasHomebrew: brewAvailable
            )
        } ?? .useExisting

        return MTPLXRuntimeUpdateSnapshot(
            appVersion: appVersion,
            appBuild: appBuild,
            cliVersion: version,
            cliPath: executable.path,
            cliInstallKind: kind,
            latestAppVersion: manifest?.appVersion,
            minimumCLIVersion: manifest?.minimumCLIVersion,
            recommendedCLIVersion: manifest?.recommendedCLIVersion,
            action: action,
            title: title(for: action),
            detail: detail(for: action, kind: kind, manifest: manifest)
        )
    }

    public static func action(
        version: String?,
        installKind: MTPLXRuntimeInstallKind,
        manifest: MTPLXReleaseManifest,
        hasHomebrew: Bool
    ) -> MTPLXRuntimeUpdateAction {
        guard let current = version.flatMap(MTPLXSemanticVersion.init) else {
            return updateAction(for: installKind)
        }
        let minimum = MTPLXSemanticVersion(manifest.minimumCLIVersion)
        let recommended = MTPLXSemanticVersion(manifest.recommendedCLIVersion)
        if let minimum, current < minimum {
            return updateAction(for: installKind)
        }
        if let recommended, current < recommended, installKind == .homebrew, hasHomebrew {
            return .updateHomebrewRecommended
        }
        return .useExisting
    }

    /// The action that brings a runtime of `installKind` back above the
    /// compatibility floor. App-owned venvs reinstall from the bundled
    /// wheel, Homebrew installs upgrade through brew, and everything else
    /// is the user's own to update.
    private static func updateAction(for installKind: MTPLXRuntimeInstallKind) -> MTPLXRuntimeUpdateAction {
        switch installKind {
        case .appOwned:
            return .updateBundledRequired
        case .homebrew:
            return .updateHomebrewRequired
        case .sourceCheckout, .pipLike, .custom, .missing:
            return .manualUpdateRequired(command: manualUpdateCommand(for: installKind))
        }
    }

    public static func installKind(
        for executableURL: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> MTPLXRuntimeInstallKind {
        if let override = environment["MTPLX_APP_FAKE_INSTALL_KIND"],
           let kind = MTPLXRuntimeInstallKind(rawValue: override) {
            return kind
        }
        let resolved = executableURL.resolvingSymlinksInPath().path
        let appRuntimeBin = MTPLXCommandBuilder.appRuntimeBinDirectory(environment: environment)
        let appRuntimePrefixes = Set([
            appRuntimeBin,
            URL(fileURLWithPath: appRuntimeBin).resolvingSymlinksInPath().path,
        ])
        if appRuntimePrefixes.contains(where: { prefix in
            executableURL.path.hasPrefix(prefix + "/") || resolved.hasPrefix(prefix + "/")
        }) {
            return .appOwned
        }
        if MTPLXCommandBuilder.isDevelopmentWrapperPath(resolved) {
            return .sourceCheckout
        }
        if resolved.contains("/opt/homebrew/") || resolved.contains("/usr/local/Homebrew/")
            || resolved.contains("/usr/local/Cellar/mtplx/")
            || resolved == "/usr/local/bin/mtplx"
            || resolved == "/opt/homebrew/bin/mtplx" {
            return .homebrew
        }
        if resolved.contains("/.local/bin/") || resolved.contains("/site-packages/") {
            return .pipLike
        }
        return .custom
    }

    public static func runtimeVersion(
        executableURL: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        let process = Process()
        process.executableURL = executableURL
        process.arguments = ["--version"]
        var env = environment
        env["PATH"] = MTPLXCommandBuilder.expandedPATH(environment: environment)
        process.environment = MTPLXCommandBuilder.pythonBytecodeSafeEnvironment(
            environment: env
        )
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        let watchdog = SubprocessWatchdog(process)
        do {
            try process.run()
        } catch {
            return nil
        }
        // A version probe must never wedge its caller (#158): a hung
        // CLI (Gatekeeper stall, dead NFS home) reads as "version
        // unknown", not an infinite wait — RuntimeSetupService calls
        // this during onboarding. 30s covers a cold first exec of the
        // Python entry point.
        let stdoutDrain = SubprocessPipeDrain(stdout)
        let stderrDrain = SubprocessPipeDrain(stderr)
        guard watchdog.wait(for: process, timeout: 30) else {
            return nil
        }
        stdoutDrain.join()
        stderrDrain.join()
        var data = stdoutDrain.snapshotData()
        data.append(stderrDrain.snapshotData())
        let output = String(data: data, encoding: .utf8) ?? ""
        return MTPLXSemanticVersion(output)?.description
    }

    private static func title(for action: MTPLXRuntimeUpdateAction) -> String {
        switch action {
        case .useExisting: return "Runtime ready"
        case .installHomebrew: return "Install runtime"
        case .updateBundledRequired: return "Runtime update required"
        case .updateHomebrewRequired: return "Runtime update required"
        case .updateHomebrewRecommended: return "Runtime update available"
        case .manualUpdateRequired: return "Manual runtime update required"
        case .homebrewRequired: return "Homebrew required"
        }
    }

    private static func detail(
        for action: MTPLXRuntimeUpdateAction,
        kind: MTPLXRuntimeInstallKind,
        manifest: MTPLXReleaseManifest?
    ) -> String {
        switch action {
        case .useExisting:
            if manifest == nil {
                return "Couldn't check the latest release, but the installed runtime is usable."
            }
            return "App and runtime are compatible."
        case .installHomebrew:
            return "MTPLX can install the command-line runtime with Homebrew."
        case .updateBundledRequired:
            return "MTPLX reinstalls its app-managed runtime from the bundled wheel automatically."
        case .updateHomebrewRequired:
            return "The installed Homebrew runtime is below the compatibility floor."
        case .updateHomebrewRecommended:
            return "A newer Homebrew runtime is available."
        case .manualUpdateRequired(let command):
            return "\(kind.displayName) runtime needs a manual update: \(command)"
        case .homebrewRequired:
            return "Install Homebrew from brew.sh, then press Retry."
        }
    }

    private static func manualUpdateCommand(for kind: MTPLXRuntimeInstallKind) -> String {
        switch kind {
        // MTPLX is distributed through the Homebrew tap and GitHub
        // releases only — never suggest pip, even to users whose old
        // CLI arrived through pip: the PyPI name is not where updates
        // ship, so that command would install a stale or wrong build.
        case .pipLike, .sourceCheckout, .custom:
            return "brew install youssofal/mtplx/mtplx"
        case .appOwned, .homebrew, .missing:
            return "brew upgrade youssofal/mtplx/mtplx"
        }
    }
}
