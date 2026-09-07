import SwiftData
import XCTest

@testable import MTPLXAppCore

/// A chat store that will not open must never silently become an in-memory
/// store: the files are kept beside themselves, a fresh store starts, and
/// the app is told. Stores written by the previous (unversioned) container
/// keep opening with their rows intact under the versioned schema.
final class ChatStoreRecoveryTests: XCTestCase {
    private func temporaryStoreURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-chat-store-recovery-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent(ChatStore.storeFilename)
    }

    private func fixedDate() throws -> Date {
        var components = DateComponents()
        components.year = 2026; components.month = 9; components.day = 3
        components.hour = 16; components.minute = 4; components.second = 5
        return try XCTUnwrap(Calendar.current.date(from: components))
    }

    /// The container exactly as the app built it before the versioned
    /// schema existed: a plain `Schema` of the four models, no migration
    /// plan. Every store on a user's Mac today was created this way.
    @MainActor
    private func legacyContainer(at url: URL) throws -> ModelContainer {
        let schema = Schema([
            ChatConversation.self,
            ChatMessage.self,
            ChatAttachment.self,
            ToolTraceRecord.self,
        ])
        let configuration = ModelConfiguration(
            "MTPLXChats",
            schema: schema,
            url: url,
            allowsSave: true,
            cloudKitDatabase: .none
        )
        return try ModelContainer(for: schema, configurations: [configuration])
    }

    @MainActor
    private func conversationTitles(in container: ModelContainer) throws -> [String] {
        let context = ModelContext(container)
        let descriptor = FetchDescriptor<ChatConversation>(sortBy: [SortDescriptor(\.createdAt)])
        return try context.fetch(descriptor).map(\.title)
    }

    @MainActor
    func testStoreWrittenByThePreviousContainerOpensUnderTheVersionedSchemaWithRowsIntact() throws {
        let url = try temporaryStoreURL()
        do {
            let legacy = try legacyContainer(at: url)
            let context = ModelContext(legacy)
            let conversation = ChatConversation(title: "Kept across the schema versioning")
            context.insert(conversation)
            let message = ChatMessage(role: .user, visibleContent: "hello", conversation: conversation)
            context.insert(message)
            conversation.messages.append(message)
            try context.save()
        }

        let opened = ChatStore.open(at: url, now: try fixedDate())

        XCTAssertNil(opened.notice, "a healthy store is not a recovery")
        XCTAssertEqual(try conversationTitles(in: opened.container), ["Kept across the schema versioning"])
        let context = ModelContext(opened.container)
        let messages = try context.fetch(FetchDescriptor<ChatMessage>())
        XCTAssertEqual(messages.map(\.visibleContent), ["hello"])
        XCTAssertEqual(messages.first?.conversation?.title, "Kept across the schema versioning")
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: url.deletingLastPathComponent().path)
            .filter { $0.contains(".unreadable-") }
        XCTAssertTrue(leftovers.isEmpty, "nothing is moved aside when the store opens: \(leftovers)")
    }

    @MainActor
    func testUnreadableStoreIsKeptAsideWithItsSidecarsAndAFreshStoreOpensWithTheNotice() throws {
        let url = try temporaryStoreURL()
        let garbage = Data("this is not a sqlite database".utf8)
        let walBytes = Data("wal bytes".utf8)
        let shmBytes = Data("shm bytes".utf8)
        try garbage.write(to: url)
        try walBytes.write(to: URL(fileURLWithPath: url.path + "-wal"))
        try shmBytes.write(to: URL(fileURLWithPath: url.path + "-shm"))

        let opened = ChatStore.open(at: url, now: try fixedDate())

        let preserved = URL(fileURLWithPath: url.path + ".unreadable-20260903-160405")
        XCTAssertEqual(opened.notice, .recoveredFromUnreadableStore(preservedAt: preserved))
        XCTAssertEqual(try Data(contentsOf: preserved), garbage, "the original store bytes are kept untouched")
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: preserved.path + "-wal")), walBytes)
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: preserved.path + "-shm")), shmBytes)

        // The fresh store at the original path is real and writable.
        let context = ModelContext(opened.container)
        context.insert(ChatConversation(title: "After recovery"))
        try context.save()
        XCTAssertEqual(try conversationTitles(in: opened.container), ["After recovery"])
        XCTAssertNotEqual(try Data(contentsOf: url), garbage, "a fresh store now lives at the original path")
    }

    @MainActor
    func testSecondRecoveryInTheSameSecondDoesNotOverwriteTheFirstPreservedStore() throws {
        let url = try temporaryStoreURL()
        let now = try fixedDate()
        try Data("first garbage".utf8).write(to: url)
        let first = ChatStore.open(at: url, now: now)
        guard case .recoveredFromUnreadableStore(let firstPreservedAt)? = first.notice else {
            return XCTFail("expected a recovery, got \(String(describing: first.notice))")
        }

        try Data("second garbage".utf8).write(to: url)
        let second = ChatStore.open(at: url, now: now)
        guard case .recoveredFromUnreadableStore(let secondPreservedAt)? = second.notice else {
            return XCTFail("expected a recovery, got \(String(describing: second.notice))")
        }

        XCTAssertEqual(secondPreservedAt.path, firstPreservedAt.path + "-2")
        XCTAssertEqual(try String(contentsOf: firstPreservedAt, encoding: .utf8), "first garbage")
        XCTAssertEqual(try String(contentsOf: secondPreservedAt, encoding: .utf8), "second garbage")
    }

    @MainActor
    func testWhenNoStoreCanBeCreatedTheAppRunsInMemoryWithTheNotice() throws {
        // A store path whose parent is a regular file: neither the existing
        // store nor a fresh one can be opened there.
        let blockingFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-chat-store-blocker-\(UUID().uuidString)")
        try Data("not a directory".utf8).write(to: blockingFile)
        let url = blockingFile.appendingPathComponent(ChatStore.storeFilename)

        let opened = ChatStore.open(at: url, now: try fixedDate())

        guard case .inMemoryOnly(let error)? = opened.notice else {
            return XCTFail("expected the in-memory notice, got \(String(describing: opened.notice))")
        }
        XCTAssertFalse(error.isEmpty)
        let context = ModelContext(opened.container)
        context.insert(ChatConversation(title: "Memory only"))
        try context.save()
        XCTAssertEqual(try conversationTitles(in: opened.container), ["Memory only"])
        XCTAssertEqual(try Data(contentsOf: blockingFile), Data("not a directory".utf8), "the blocking file is left alone")
    }
}
