import Foundation

// MARK: - ChatEscapePolicy
//
// What Esc does on the chat surface, decided in one place. Esc used to
// be bound twice — the close-chat button and the Stop Generating menu
// item both claimed it, the menu item only while a reply streamed — so
// which one fired depended on responder order and the key could close
// the surface when the user meant to stop a runaway generation, or the
// reverse. Now a single binding asks this policy: a streaming reply is
// stopped first; an idle surface is closed. Stop Generating keeps its
// own shortcut so the menu never competes for Esc.

public enum ChatEscapePolicy {
    public enum Action: Equatable, Sendable {
        /// A reply is streaming in the visible conversation: Esc stops
        /// it and leaves the surface open so the partial can be read.
        case stopGenerating
        /// Nothing is streaming: Esc collapses the chat surface.
        case closeSurface
    }

    /// The one decision Esc makes, from the only input that matters.
    public static func action(isStreaming: Bool) -> Action {
        isStreaming ? .stopGenerating : .closeSurface
    }
}
