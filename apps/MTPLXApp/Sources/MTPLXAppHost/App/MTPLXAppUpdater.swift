import Combine
import SwiftUI
import MTPLXAppCore
@preconcurrency import Sparkle

/// Sparkle wiring with gentle scheduled-update reminders.
///
/// Scheduled (non-user-initiated) checks do NOT show Sparkle's modal
/// dialog; they publish `pendingUpdate` so the app can render its own
/// bottom-right toast. "Install Now" resumes the Sparkle session,
/// which brings the standard, battle-tested install flow into focus.
/// User-initiated checks (the menu item) keep the standard dialog.
@MainActor
final class MTPLXAppUpdater: NSObject, ObservableObject {
    struct PendingUpdate: Equatable {
        /// Marketing version, e.g. "1.0.0".
        let version: String
        /// Build number Sparkle compares, e.g. "10003".
        let build: String
    }

    @Published private(set) var canCheckForUpdates: Bool = false
    /// Non-nil while a scheduled check has found an update the user
    /// has not yet engaged with. Drives the in-app toast.
    @Published private(set) var pendingUpdate: PendingUpdate?

    private var updaterController: SPUStandardUpdaterController!
    private var cancellable: AnyCancellable?

    init(startingUpdater: Bool = true) {
        super.init()
        let controller = SPUStandardUpdaterController(
            startingUpdater: startingUpdater,
            updaterDelegate: nil,
            userDriverDelegate: self
        )
        self.updaterController = controller
        self.canCheckForUpdates = controller.updater.canCheckForUpdates
        self.cancellable = controller.updater
            .publisher(for: \.canCheckForUpdates)
            .receive(on: RunLoop.main)
            .sink { [weak self] canCheck in
                self?.canCheckForUpdates = canCheck
            }

        guard startingUpdater else { return }
        // The toast IS the scheduled-check UX, so checks must run
        // without Sparkle's first-launch permission prompt, and a
        // fresh check should happen on every app open — not only when
        // the daily timer happens to elapse.
        let updater = controller.updater
        updater.automaticallyChecksForUpdates = true
        updater.updateCheckInterval = 21_600
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            updater.checkForUpdatesInBackground()
        }
    }

    func checkForUpdates() {
        updaterController.checkForUpdates(nil)
    }

    /// Resume the paused update session in focus — Sparkle takes over
    /// from here with its standard download/install/relaunch flow.
    func installPendingUpdate() {
        pendingUpdate = nil
        updaterController.checkForUpdates(nil)
    }

    /// Hide the toast for this session; the next scheduled check (or
    /// app open) will surface the update again.
    func dismissPendingUpdate() {
        pendingUpdate = nil
    }
}

extension MTPLXAppUpdater: SPUStandardUserDriverDelegate {
    nonisolated var supportsGentleScheduledUpdateReminders: Bool { true }

    nonisolated func standardUserDriverShouldHandleShowingScheduledUpdate(
        _ update: SUAppcastItem,
        andInImmediateFocus immediateFocus: Bool
    ) -> Bool {
        // Scheduled finds are always ours to present (the toast);
        // returning false hands presentation to the delegate.
        false
    }

    nonisolated func standardUserDriverWillHandleShowingUpdate(
        _ handleShowingUpdate: Bool,
        forUpdate update: SUAppcastItem,
        state: SPUUserUpdateState
    ) {
        guard !state.userInitiated, !handleShowingUpdate else { return }
        let version = update.displayVersionString
        let build = update.versionString
        Task { @MainActor in
            self.pendingUpdate = PendingUpdate(version: version, build: build)
        }
    }

    nonisolated func standardUserDriverDidReceiveUserAttention(
        forUpdate update: SUAppcastItem
    ) {
        Task { @MainActor in
            self.pendingUpdate = nil
        }
    }

    nonisolated func standardUserDriverWillFinishUpdateSession() {
        Task { @MainActor in
            self.pendingUpdate = nil
        }
    }
}

struct CheckForUpdatesCommand: View {
    @ObservedObject var updater: MTPLXAppUpdater
    @ObservedObject var languageStore: LanguageStore

    var body: some View {
        Button(L10n.string("Check for Updates...", language: languageStore.language)) {
            updater.checkForUpdates()
        }
        .disabled(!updater.canCheckForUpdates)
    }
}
