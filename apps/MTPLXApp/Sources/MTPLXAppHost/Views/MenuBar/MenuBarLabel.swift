import SwiftUI
import MTPLXAppCore

// MARK: - MenuBarLabel
//
// Live status glyph for the MenuBarExtra. The icon shape tracks the
// daemon lifecycle (running / warming / degraded / stopped) so the menu
// bar reflects the engine state at a glance instead of a static bolt.
// Menu bar icons are template-rendered, so state reads from the glyph
// shape rather than color.

struct MenuBarLabel: View {
    @EnvironmentObject private var backend: MTPLXBackendStore

    var body: some View {
        Image(systemName: iconName)
            .accessibilityLabel(tr("MTPLX — %@", backend.daemonState.kind.label))
    }

    private var iconName: String {
        switch backend.daemonState.kind {
        case .running:
            return "bolt.fill"
        case .starting, .warming, .stopping:
            return "bolt"
        case .degraded, .crashed:
            return "bolt.trianglebadge.exclamationmark"
        case .stopped:
            return "bolt.slash.fill"
        }
    }
}
