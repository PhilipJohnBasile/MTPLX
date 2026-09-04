import SwiftUI
import MTPLXAppCore

// MARK: - ChatOverlay
//
// Wraps `ChatView` so it presents as a real overlay over the
// dashboard. The overlay slides UP from the bottom edge of the
// dashboard area on open and collapses DOWN on close (the
// transition is wired by the call-site `.move(edge: .bottom)`
// on the chat slot). Carries a `ChatCloseButton` pinned to the
// top-left — the Mac convention for "close this surface"
// (mirrors the red traffic-light position on a window). A
// peer `ChatExpandTab` (in `ContentView`'s ZStack) renders at
// the bottom-centre of the dashboard area when chat is closed
// so the user can pull the drawer back up.

struct ChatOverlay: View, Equatable {
    let daemonState: DaemonState
    let startupPhase: DaemonStartupPhase
    let selectedModel: String
    let visionEnabled: Bool
    let performanceLock: Bool
    let onCollapse: () -> Void

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.daemonState == rhs.daemonState
            && lhs.startupPhase == rhs.startupPhase
            && lhs.selectedModel == rhs.selectedModel
            && lhs.visionEnabled == rhs.visionEnabled
            && lhs.performanceLock == rhs.performanceLock
    }

    var body: some View {
        VStack(spacing: 0) {
            closeBar
            ChatView(
                daemonState: daemonState,
                startupPhase: startupPhase,
                selectedModel: selectedModel,
                visionEnabled: visionEnabled,
                performanceLock: performanceLock
            )
                .background(Brand.bgOuter)
        }
    }

    /// Thin top bar that anchors `ChatCloseButton` at the leftmost
    /// position so the close affordance lives in the top-left of
    /// the chat overlay regardless of whether the sidebar is open
    /// or collapsed. Single hairline separator below the bar so the
    /// chat content reads as a layered surface. The surface's one Esc
    /// binding lives here too, beside the button rather than on it,
    /// because a click always closes while Esc first stops a
    /// streaming reply.
    private var closeBar: some View {
        HStack(spacing: 8) {
            ChatCloseButton(action: onCollapse)
            ChatEscapeShortcut(onCloseSurface: onCollapse)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            Brand.bgInner
                .overlay(
                    Rectangle()
                        .fill(Brand.separator)
                        .frame(height: Brand.hairline),
                    alignment: .bottom
                )
        )
    }
}

// MARK: - ChatEscapeShortcut
//
// The chat surface's ONLY Esc binding. It carries no visible control:
// Esc is a decision, not a button — while a reply streams it stops the
// generation (the user's usual reason for reaching for Esc mid-reply);
// otherwise it closes the surface, as the close button does on click.
// The decision itself is `ChatEscapePolicy`, shared with nothing else
// on purpose: the Stop Generating menu item has its own shortcut, so
// there is exactly one Esc binding on the surface and nothing for
// responder order to arbitrate.

private struct ChatEscapeShortcut: View {
    @EnvironmentObject private var viewModel: ChatViewModel
    let onCloseSurface: () -> Void

    var body: some View {
        Button(action: perform) {
            Color.clear
        }
        .buttonStyle(.plain)
        .frame(width: 0, height: 0)
        .opacity(0)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
        .keyboardShortcut(.escape, modifiers: [])
    }

    private func perform() {
        switch ChatEscapePolicy.action(isStreaming: viewModel.isStreaming) {
        case .stopGenerating:
            Task { await viewModel.cancel() }
        case .closeSurface:
            onCloseSurface()
        }
    }
}

// MARK: - ChatCloseButton
//
// Chrome pill at the top-left of the chat overlay. Single click
// collapses the chat surface back down to the dashboard; Esc does the
// same when nothing is streaming (see `ChatEscapeShortcut`).
// `chevron.down` glyph so the icon matches the collapse direction
// (chat is about to slide down and out of view).

struct ChatCloseButton: View {
    let action: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.down")
                    .font(.system(size: 10, weight: .heavy))
                Text(tr("Close chat"))
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .tracking(0.3)
            }
            .foregroundStyle(hovering ? Brand.typeBody : Brand.typeSecondary)
            .padding(.horizontal, 12)
            .padding(.vertical, 5)
            .background {
                Capsule(style: .continuous)
                    .fill(Brand.wash.opacity(0.04))
                    .overlay {
                        Capsule(style: .continuous)
                            .strokeBorder(
                                hovering ? Brand.separatorStrong : Brand.separator,
                                lineWidth: Brand.hairlineStrong
                            )
                    }
            }
            .contentShape(Capsule(style: .continuous))
        }
        .buttonStyle(.plain)
        .help(tr("Close chat (Esc)"))
        .accessibilityLabel(tr("Close chat"))
        .onHover { hovering = $0 }
        .animation(
            reduceMotion ? nil : .easeOut(duration: 0.16),
            value: hovering
        )
    }
}

// MARK: - ChatExpandTab
//
// Counter-affordance to `ChatCloseButton`. A small chrome pill
// pinned to the bottom-centre of the dashboard area, visible only
// when chat is closed. Single click pulls the chat drawer up from
// below. Panel-surface gradient + hairlineStrong stroke + soft
// elevation so it reads as a chrome handle attached to the
// surrounding chrome system, not as a separate cheap button.

struct SurfaceExpandTab: View {
    let surface: AppExpandableSurface
    let action: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.up")
                    .font(.system(size: 10, weight: .heavy))
                Text(tr("Expand %@", surface.title))
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .tracking(0.3)
            }
            .foregroundStyle(hovering ? Brand.typeBody : Brand.typeSecondary)
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background {
                Capsule(style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [Brand.panelSurfaceTop, Brand.panelSurfaceBottom],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .overlay {
                        Capsule(style: .continuous)
                            .strokeBorder(
                                hovering ? Brand.separatorStrong : Brand.separator,
                                lineWidth: Brand.hairlineStrong
                            )
                    }
                    .shadow(
                        color: Brand.Elevation.mid.color,
                        radius: Brand.Elevation.mid.radius,
                        x: 0,
                        y: Brand.Elevation.mid.y
                    )
            }
            .contentShape(Capsule(style: .continuous))
        }
        .buttonStyle(.plain)
        .help(tr("Expand %@", surface.title))
        .accessibilityLabel(tr("Expand %@", surface.title))
        .onHover { hovering = $0 }
        .animation(
            reduceMotion ? nil : .easeOut(duration: 0.16),
            value: hovering
        )
    }
}
