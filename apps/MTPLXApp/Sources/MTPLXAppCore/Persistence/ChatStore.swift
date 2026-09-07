import Foundation
import os
import SwiftData

// MARK: - ChatStore
//
// Factory for the SwiftData `ModelContainer` backing the in-app chat
// surface. Crucially this uses an EXPLICIT `ModelConfiguration` URL
// rather than the SwiftData default, because on macOS the default
// store location/name is shared across apps and can collide with any
// other app that also uses SwiftData. The store lives under
// `~/Library/Application Support/MTPLX/chats.store`.
//
// The app opens its one container through `ChatStore.open()`, which
// also decides what to do when the store on disk will not open: the
// files are kept beside themselves under a dated name and a fresh store
// starts, or, if even that fails, the app runs on an in-memory store.
// Either way the caller receives a notice to show, so a reset is never
// silent. Tests can opt into an in-memory container via
// `makeInMemoryContainer()` to avoid touching disk.

/// What `ChatStore.open()` had to do to hand back a working container.
public enum ChatStoreRecoveryNotice: Equatable, Sendable {
    /// The store on disk would not open. It was renamed beside itself
    /// (with its `-wal` and `-shm` siblings) and a fresh, empty store
    /// was created in its place.
    case recoveredFromUnreadableStore(preservedAt: URL)
    /// Neither the existing store nor a fresh one could be opened, so
    /// chats live in memory for this session and are not kept.
    case inMemoryOnly(error: String)
}

/// The container to run on plus the recovery notice, if any.
@MainActor
public struct ChatStoreOpenResult {
    public let container: ModelContainer
    public let notice: ChatStoreRecoveryNotice?
}

/// Holds the launch-time recovery notice for the life of the app so the
/// chat sidebar can show it until the user dismisses it.
@MainActor
public final class ChatStoreRecoveryState: ObservableObject {
    @Published public var notice: ChatStoreRecoveryNotice?

    public init(notice: ChatStoreRecoveryNotice? = nil) {
        self.notice = notice
    }
}

public enum ChatStore {
    private static let log = Logger(subsystem: "com.mtplx.app", category: "ChatStore")

    /// SQLite sidecars SwiftData keeps next to the store file. Moved
    /// together with it so the preserved set stays openable as one
    /// database.
    static let sidecarSuffixes = ["-wal", "-shm"]
    /// Subfolder under `~/Library/Application Support/` where MTPLX
    /// persists user data. Reused for the SwiftData store and any
    /// future chat-related files (e.g. exported transcripts).
    public static let appSupportSubdirectory = "MTPLX"
    /// Filename for the SwiftData store. SwiftData will create three
    /// adjacent files: `chats.store`, `chats.store-shm`, `chats.store-wal`.
    public static let storeFilename = "chats.store"
    /// Optional QA/dev override. Production launches should leave this unset.
    public static let storePathEnvironmentVariable = "MTPLX_CHAT_STORE_PATH"
    public static let storePathArgumentNames = [
        "--mtplx-chat-store",
        "--mtplx-chat-store-path"
    ]

    /// Resolves the on-disk URL of the persistent store, creating the
    /// containing directory if it does not exist.
    public static func storeURL() throws -> URL {
        if let explicit = explicitStorePath(
            environment: ProcessInfo.processInfo.environment,
            arguments: CommandLine.arguments
        ) {
            return try explicitStoreURL(explicit)
        }
        let supportRoot = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let mtplxDir = supportRoot.appendingPathComponent(
            appSupportSubdirectory,
            isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: mtplxDir,
            withIntermediateDirectories: true
        )
        return mtplxDir.appendingPathComponent(storeFilename)
    }

    public static func explicitStorePath(
        environment: [String: String],
        arguments: [String]
    ) -> String? {
        if let value = environment[storePathEnvironmentVariable]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !value.isEmpty {
            return value
        }
        for (index, argument) in arguments.enumerated() {
            for name in storePathArgumentNames {
                if argument == name,
                   arguments.indices.contains(index + 1) {
                    let value = arguments[index + 1].trimmingCharacters(in: .whitespacesAndNewlines)
                    return value.isEmpty ? nil : value
                }
                let prefix = name + "="
                if argument.hasPrefix(prefix) {
                    let value = String(argument.dropFirst(prefix.count))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    return value.isEmpty ? nil : value
                }
            }
        }
        return nil
    }

    public static func explicitStoreURL(_ path: String) throws -> URL {
        let expanded = NSString(string: path).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return url
    }

    /// The app's one entry point. Opens the persistent store at `url`
    /// (the default location when nil). If the store will not open, its
    /// files are kept beside it as `chats.store.unreadable-<yyyyMMdd-HHmmss>`
    /// (plus `-wal`/`-shm`, never deleted) and a fresh store is created;
    /// if that fails too, an in-memory container is returned. The notice
    /// says which of those happened so the sidebar can tell the user.
    @MainActor
    public static func open(at url: URL? = nil, now: Date = Date()) -> ChatStoreOpenResult {
        let storeURL: URL
        do {
            storeURL = try url ?? self.storeURL()
        } catch {
            log.error("chat store location unavailable: \(String(describing: error), privacy: .public)")
            return inMemoryFallback(after: error)
        }
        let firstFailure: Error
        do {
            return ChatStoreOpenResult(container: try makeContainer(at: storeURL), notice: nil)
        } catch {
            firstFailure = error
        }
        log.error("chat store at \(storeURL.path, privacy: .public) would not open: \(String(describing: firstFailure), privacy: .public)")
        let preservedAt: URL
        do {
            preservedAt = try setAsideStore(at: storeURL, now: now)
        } catch {
            log.error("chat store could not be moved aside: \(String(describing: error), privacy: .public)")
            return inMemoryFallback(after: firstFailure)
        }
        do {
            let container = try makeContainer(at: storeURL)
            log.notice("chat store recovered: previous files kept as \(preservedAt.path, privacy: .public)")
            return ChatStoreOpenResult(
                container: container,
                notice: .recoveredFromUnreadableStore(preservedAt: preservedAt)
            )
        } catch {
            log.error("fresh chat store would not open either: \(String(describing: error), privacy: .public)")
            return inMemoryFallback(after: error)
        }
    }

    /// Chats for this session only. The one thing this cannot survive is
    /// the in-memory container failing as well, which only a broken model
    /// schema can cause; that is a build defect every test that builds a
    /// container catches, not a user's disk state.
    @MainActor
    private static func inMemoryFallback(after error: Error) -> ChatStoreOpenResult {
        let description = (error as? LocalizedError)?.errorDescription ?? String(describing: error)
        do {
            return ChatStoreOpenResult(
                container: try makeInMemoryContainer(),
                notice: .inMemoryOnly(error: description)
            )
        } catch let inMemoryError {
            preconditionFailure(
                "The chat model schema cannot be loaded even in memory (\(inMemoryError)); "
                    + "the persistent store had failed with: \(description)"
            )
        }
    }

    /// Rename `chats.store` and its sidecars to
    /// `chats.store.unreadable-<stamp>` (+ `-wal`/`-shm`). Returns the new
    /// store URL. A stamp that already exists gets a numeric suffix.
    static func setAsideStore(at storeURL: URL, now: Date) throws -> URL {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let base = storeURL.path + ".unreadable-" + formatter.string(from: now)
        var destination = URL(fileURLWithPath: base)
        var attempt = 1
        while FileManager.default.fileExists(atPath: destination.path) {
            attempt += 1
            destination = URL(fileURLWithPath: "\(base)-\(attempt)")
        }
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: storeURL.path) {
            try fileManager.moveItem(at: storeURL, to: destination)
        }
        for suffix in sidecarSuffixes {
            let sidecar = URL(fileURLWithPath: storeURL.path + suffix)
            guard fileManager.fileExists(atPath: sidecar.path) else { continue }
            try fileManager.moveItem(at: sidecar, to: URL(fileURLWithPath: destination.path + suffix))
        }
        return destination
    }

    /// Build the persistent SwiftData `ModelContainer` at the default
    /// location.
    /// - Throws: any FileManager / SwiftData error encountered while
    ///   creating the support directory or initializing the container.
    @MainActor
    public static func makeContainer() throws -> ModelContainer {
        try makeContainer(at: storeURL())
    }

    /// Build the persistent container at `url` under the versioned schema
    /// and migration plan.
    @MainActor
    public static func makeContainer(at url: URL) throws -> ModelContainer {
        let schema = Schema(versionedSchema: ChatSchemaV1.self)
        let configuration = ModelConfiguration(
            "MTPLXChats",
            schema: schema,
            url: url,
            allowsSave: true,
            cloudKitDatabase: .none
        )
        return try ModelContainer(
            for: schema,
            migrationPlan: ChatSchemaMigrationPlan.self,
            configurations: [configuration]
        )
    }

    /// In-memory container for tests, previews, and the last-resort
    /// fallback. Does not touch disk.
    @MainActor
    public static func makeInMemoryContainer() throws -> ModelContainer {
        let schema = Schema(versionedSchema: ChatSchemaV1.self)
        let configuration = ModelConfiguration(
            "MTPLXChatsInMemory",
            schema: schema,
            isStoredInMemoryOnly: true,
            allowsSave: true,
            cloudKitDatabase: .none
        )
        return try ModelContainer(
            for: schema,
            migrationPlan: ChatSchemaMigrationPlan.self,
            configurations: [configuration]
        )
    }
}
