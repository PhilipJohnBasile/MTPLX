import Foundation
import SwiftData
import XCTest

@testable import MTPLXAppCore

// MARK: - LocalizedAutoTitleTests
//
// A conversation is created with the active language's "New Chat" and
// takes its name from the first message the user sends. The guard that
// decided "still untitled" compared against the English literal, so in
// eleven of the twelve shipped languages it never fired and every chat
// stayed a placeholder forever. Placeholder detection now comes from
// the string tables (every language's translation plus the English
// literal older rows were seeded with), rows that stayed untitled are
// named at launch, and these tests run the titling path under
// non-English languages.

final class LocalizedAutoTitleTests: XCTestCase {
    override func setUp() {
        super.setUp()
        L10n.activate(.english)
    }

    override func tearDown() {
        L10n.activate(.english)
        super.tearDown()
    }

    /// A client whose daemon is not there: titling happens before the
    /// request leaves, and the turn fails fast afterwards.
    private static let unreachableClient = MTPLXChatClient(
        apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:1")!)
    )

    @MainActor
    private func makeViewModel(container: ModelContainer) -> ChatViewModel {
        ChatViewModel(
            container: container,
            chatClientProvider: { Self.unreachableClient },
            modelName: { "mtplx-test-model" }
        )
    }

    // MARK: Placeholder detection

    func testEveryShippedLanguagePlaceholderAndTheEnglishLiteralArePlaceholders() {
        XCTAssertTrue(ChatConversationTitle.isPlaceholder("New Chat"))
        var seen: Set<String> = []
        for language in AppLanguage.allCases {
            let localized = L10n.string("New Chat", language: language)
            XCTAssertTrue(ChatConversationTitle.isPlaceholder(localized), language.code)
            XCTAssertTrue(ChatConversationTitle.isPlaceholder("  \(localized)\n"), "\(language.code): whitespace")
            seen.insert(localized)
        }
        XCTAssertGreaterThan(seen.count, 6, "the placeholder really is translated")
        XCTAssertFalse(ChatConversationTitle.isPlaceholder("Neuer Chat über MTPLX"))
        XCTAssertFalse(ChatConversationTitle.isPlaceholder("How do I install MTPLX"))
        XCTAssertFalse(ChatConversationTitle.isPlaceholder(""))
    }

    func testDerivedTitleKeepsFiveWordsAndFallsBackToTheActivePlaceholder() {
        XCTAssertEqual(
            ChatConversationTitle.derived(from: "  Wie   funktioniert MTPLX auf dem Mac heute?\n"),
            "Wie funktioniert MTPLX auf dem"
        )
        L10n.activate(.japanese)
        XCTAssertEqual(ChatConversationTitle.derived(from: "MTPLXの使い方を教えて"), "MTPLXの使い方を教えて")
        XCTAssertEqual(ChatConversationTitle.derived(from: "   "), "新規チャット")
        XCTAssertTrue(ChatConversationTitle.isPlaceholder(ChatConversationTitle.derived(from: "")))
        XCTAssertTrue(ChatConversation(title: ChatConversationTitle.placeholder).titleIsPlaceholder)
    }

    // MARK: Titling under a non-English language

    @MainActor
    func testSendUnderGermanTitlesTheConversationFromItsFirstWords() async throws {
        L10n.activate(.german)
        let container = try ChatStore.makeInMemoryContainer()
        let viewModel = makeViewModel(container: container)

        let conversation = viewModel.createNewConversation()
        XCTAssertEqual(conversation.title, "Neuer Chat")
        XCTAssertTrue(conversation.titleIsPlaceholder)

        viewModel.send("Hallo Welt, wie geht es dir heute?")
        XCTAssertEqual(conversation.title, "Hallo Welt, wie geht es")
        XCTAssertFalse(conversation.titleIsPlaceholder)
        XCTAssertEqual(viewModel.conversations.first?.title, "Hallo Welt, wie geht es")

        try await pollUntil("the unreachable turn settles") { !viewModel.isStreaming }
        // A second message never renames a titled conversation.
        viewModel.send("Und noch eine Frage")
        XCTAssertEqual(conversation.title, "Hallo Welt, wie geht es")
        try await pollUntil("the second turn settles") { !viewModel.isStreaming }
    }

    @MainActor
    func testRowCreatedWithTheEnglishLiteralIsTitledUnderJapanese() async throws {
        // A conversation from before localisation (or from the model's
        // default initialiser) carries the English literal; a Japanese
        // user's next message must still name it.
        let container = try ChatStore.makeInMemoryContainer()
        let legacy = ChatConversation(title: "New Chat")
        container.mainContext.insert(legacy)
        try container.mainContext.save()

        L10n.activate(.japanese)
        let viewModel = makeViewModel(container: container)
        viewModel.select(legacy)
        viewModel.send("MTPLXの使い方を教えて")
        XCTAssertEqual(legacy.title, "MTPLXの使い方を教えて")
        try await pollUntil("the unreachable turn settles") { !viewModel.isStreaming }
    }

    @MainActor
    func testPlaceholderRowsFromAnotherLanguageAreStillPlaceholdersAfterSwitching() async throws {
        // Created while the app was in French, sent after switching to Korean.
        let container = try ChatStore.makeInMemoryContainer()
        let french = ChatConversation(title: L10n.string("New Chat", language: .french))
        container.mainContext.insert(french)
        try container.mainContext.save()

        L10n.activate(.korean)
        let viewModel = makeViewModel(container: container)
        viewModel.select(french)
        XCTAssertTrue(french.titleIsPlaceholder)
        viewModel.send("안녕하세요 오늘 날씨 어때요 정말 궁금해요")
        XCTAssertEqual(french.title, "안녕하세요 오늘 날씨 어때요 정말")
        try await pollUntil("the unreachable turn settles") { !viewModel.isStreaming }
    }

    // MARK: Existing untitled rows are named at launch

    @MainActor
    func testConversationsLeftUntitledByTheOldGuardAreNamedWhenTheViewModelLoads() throws {
        let container = try ChatStore.makeInMemoryContainer()
        let context = container.mainContext

        let german = ChatConversation(title: "Neuer Chat", createdAt: Date(timeIntervalSince1970: 100))
        context.insert(german)
        let germanQuestion = ChatMessage(
            role: .user, visibleContent: "Wie funktioniert MTPLX auf dem Mac",
            createdAt: Date(timeIntervalSince1970: 101), conversation: german
        )
        context.insert(germanQuestion)
        german.messages.append(germanQuestion)
        let germanAnswer = ChatMessage(
            role: .assistant, visibleContent: "So geht es",
            createdAt: Date(timeIntervalSince1970: 102), conversation: german
        )
        context.insert(germanAnswer)
        german.messages.append(germanAnswer)

        let empty = ChatConversation(title: "New Chat", createdAt: Date(timeIntervalSince1970: 200))
        context.insert(empty)

        let named = ChatConversation(title: "Release notes draft", createdAt: Date(timeIntervalSince1970: 300))
        context.insert(named)
        let namedQuestion = ChatMessage(
            role: .user, visibleContent: "unrelated first message",
            createdAt: Date(timeIntervalSince1970: 301), conversation: named
        )
        context.insert(namedQuestion)
        named.messages.append(namedQuestion)
        try context.save()

        _ = makeViewModel(container: container)

        XCTAssertEqual(german.title, "Wie funktioniert MTPLX auf dem")
        XCTAssertEqual(empty.title, "New Chat", "nothing to derive a title from yet")
        XCTAssertEqual(named.title, "Release notes draft", "a real title is never touched")

        // The rename reached the store, not just the loaded object.
        let descriptor = FetchDescriptor<ChatConversation>(sortBy: [SortDescriptor(\.createdAt)])
        let reloaded = try ModelContext(container).fetch(descriptor)
        XCTAssertEqual(
            reloaded.map(\.title),
            ["Wie funktioniert MTPLX auf dem", "New Chat", "Release notes draft"]
        )
    }
}
