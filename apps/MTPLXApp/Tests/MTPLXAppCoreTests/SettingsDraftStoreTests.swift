import XCTest

@testable import MTPLXAppCore

/// The Settings draft lives above the tab. These tests drive the store the
/// way the tab does — `adoptIfUnedited` on every appearance and on every
/// persisted-configuration change, `reset` after a save or a revert — with
/// the tab itself modelled as a throwaway object that is destroyed and
/// recreated between appearances, exactly as the dashboard does.
@MainActor
final class SettingsDraftStoreTests: XCTestCase {
    /// Stand-in for the tab view: it has no state of its own any more,
    /// only a reference to the shared store, and it syncs on appearance.
    @MainActor
    private struct TabScoped {
        let drafts: SettingsDraftStore

        func appear(with persisted: MTPLXAppConfiguration) {
            drafts.adoptIfUnedited(persisted)
        }
    }

    private func persisted() -> MTPLXAppConfiguration {
        MTPLXAppConfiguration(model: "/models/qwen", profile: "sustained", host: "127.0.0.1", port: 8000)
    }

    func testEditedFieldSurvivesTheTabBeingDestroyedAndRecreated() {
        let drafts = SettingsDraftStore()
        let saved = persisted()
        var tab: TabScoped? = TabScoped(drafts: drafts)
        tab?.appear(with: saved)
        XCTAssertEqual(drafts.draft, saved, "first appearance seeds the draft from the saved configuration")
        XCTAssertFalse(drafts.hasUnsavedChanges)

        drafts.draft.port = 9001
        drafts.draft.host = "0.0.0.0"
        XCTAssertTrue(drafts.hasUnsavedChanges)

        // The user looks at another tab: the Settings tab is torn down...
        tab = nil
        // ...and rebuilt when they come back.
        tab = TabScoped(drafts: drafts)
        tab?.appear(with: saved)

        XCTAssertEqual(drafts.draft.port, 9001, "the pending edit is still on screen")
        XCTAssertEqual(drafts.draft.host, "0.0.0.0")
        XCTAssertTrue(drafts.hasUnsavedChanges, "and it still reads as unsaved")
        XCTAssertEqual(drafts.lastSynced, saved, "while the saved values are remembered for Revert")
    }

    func testUneditedDraftFollowsAConfigurationChangedElsewhere() {
        let drafts = SettingsDraftStore()
        let first = persisted()
        TabScoped(drafts: drafts).appear(with: first)

        var moved = first
        moved.port = 8001 // a port fallback persisted from the launch path
        drafts.adoptIfUnedited(moved)

        XCTAssertEqual(drafts.draft, moved)
        XCTAssertFalse(drafts.hasUnsavedChanges)
    }

    func testPendingEditsAreKeptOverAnExternalChangeUntilSavedOrReverted() {
        let drafts = SettingsDraftStore()
        let first = persisted()
        TabScoped(drafts: drafts).appear(with: first)
        drafts.draft.contextWindow = 32_768

        var moved = first
        moved.port = 8001
        drafts.adoptIfUnedited(moved)

        XCTAssertEqual(drafts.draft.contextWindow, 32_768, "the user's edit is not thrown away")
        XCTAssertEqual(drafts.draft.port, 8000, "the draft is left as the user sees it")
        XCTAssertTrue(drafts.hasUnsavedChanges)

        drafts.reset(to: moved)
        XCTAssertEqual(drafts.draft, moved)
        XCTAssertFalse(drafts.hasUnsavedChanges)
    }

    func testResetAfterSaveAlignsBothHalvesAndClearsUnsaved() {
        let drafts = SettingsDraftStore()
        TabScoped(drafts: drafts).appear(with: persisted())
        drafts.draft.memoryLimitGB = 64
        XCTAssertTrue(drafts.hasUnsavedChanges)

        let savedNow = drafts.draft
        drafts.reset(to: savedNow)

        XCTAssertEqual(drafts.draft, savedNow)
        XCTAssertEqual(drafts.lastSynced, savedNow)
        XCTAssertFalse(drafts.hasUnsavedChanges)
        // A later appearance with the same saved values changes nothing.
        TabScoped(drafts: drafts).appear(with: savedNow)
        XCTAssertEqual(drafts.draft.memoryLimitGB, 64)
        XCTAssertFalse(drafts.hasUnsavedChanges)
    }
}
