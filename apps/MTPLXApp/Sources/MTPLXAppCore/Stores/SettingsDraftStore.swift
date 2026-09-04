import Foundation

/// The Settings tab's editing session: the working copy of the persisted
/// configuration, the configuration it was last synced to, and the state
/// of a save in flight.
///
/// It lives above the tab (owned for the app's lifetime and handed down
/// through the environment) because the dashboard rebuilds the Settings
/// tab every time it is selected. When this was the tab's own `@State`,
/// switching to another tab and back silently threw away every pending
/// edit and showed the saved values again, which read as "the app failed
/// to save".
@MainActor
public final class SettingsDraftStore: ObservableObject {
    /// The values the controls edit. Committed by Save, discarded by Revert.
    @Published public var draft: MTPLXAppConfiguration
    /// The persisted configuration the draft was last aligned with. The
    /// difference between the two is exactly the user's pending edits.
    @Published public var lastSynced: MTPLXAppConfiguration
    /// A save/apply is running; the buttons show progress and stay
    /// disabled even if the user leaves and comes back mid-apply.
    @Published public var isApplying = false
    /// The last save's failure, shown until the next save attempt.
    @Published public var lastSaveError: String?

    public init(configuration: MTPLXAppConfiguration = MTPLXAppConfiguration()) {
        draft = configuration
        lastSynced = configuration
    }

    /// True while the draft carries edits the user has not saved.
    public var hasUnsavedChanges: Bool {
        draft != lastSynced
    }

    /// Align the draft with `configuration`, dropping any pending edits.
    /// Used after a successful save (with the saved values) and by Revert.
    public func reset(to configuration: MTPLXAppConfiguration) {
        draft = configuration
        lastSynced = configuration
    }

    /// Take a configuration that changed outside the tab (a port fallback,
    /// a model pick, another session's save) unless the user has pending
    /// edits, which are kept until they save or revert. Called whenever
    /// the tab appears and whenever the persisted configuration changes
    /// while it is visible.
    public func adoptIfUnedited(_ configuration: MTPLXAppConfiguration) {
        if draft == configuration {
            lastSynced = configuration
        } else if draft == lastSynced {
            draft = configuration
            lastSynced = configuration
        }
    }
}
