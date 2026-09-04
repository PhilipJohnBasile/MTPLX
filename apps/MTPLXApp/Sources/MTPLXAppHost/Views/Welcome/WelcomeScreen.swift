import SwiftUI
import AppKit
import MTPLXAppCore

// MARK: - WelcomeScreen
//
// The resting state of the app whenever the daemon is not running. Big
// chrome wordmark, tracked subtitle, and an enormous ON button. Hitting
// the button calls `backend.startDaemon()` which then drives the daemon
// state into `.starting → .warming → .running`. ContentView reacts to
// those state changes and crossfades to `WarmingScreen` or
// `DashboardSurface`.

struct WelcomeScreen: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore

    var body: some View {
        ZStack {
            Brand.pianoRadial
                .ignoresSafeArea()

            VStack(spacing: 28) {
                WordmarkView(height: 96)
                WordmarkSubtitle(dividerWidth: 320)

                Spacer().frame(height: 12)

                StartButton(
                    daemonState: backend.daemonState.kind,
                    soundEnabled: themeStore.soundEnabled,
                    motionEnabled: !themeStore.reduceMotionPreference
                ) {
                    Task { await backend.startDaemon() }
                }

                Text(tr("RUNS ON YOUR MAC · 127.0.0.1:%@", String(backend.configuration.port)))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .tracking(4)
                    .foregroundStyle(Brand.textHighlight.opacity(0.55))
                    .padding(.top, 6)
            }
            .padding(.horizontal, 48)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(tr("Welcome to MTPLX"))
    }
}

// MARK: - WarmingScreen
//
// What you see between clicking ON and the daemon being healthy. Same
// layout as WelcomeScreen but the StartButton enters its `.warming` look
// (pulsing chrome ring) and the subtitle swaps to "LOADING MODEL…" with
// an animated progress dot.

struct WarmingScreen: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore
    @EnvironmentObject private var stopCoordinator: AppStopCoordinator

    var body: some View {
        ZStack {
            Brand.pianoRadial
                .ignoresSafeArea()

            VStack(spacing: 28) {
                WordmarkView(height: 96)

                VStack(spacing: 12) {
                    Rectangle()
                        .fill(Brand.wash.opacity(0.10))
                        .frame(width: 320, height: 1)
                    Text(progressLabel)
                        .font(BrandFont.subtitle())
                        .tracking(4)
                        .foregroundStyle(Brand.textHighlight)
                        .contentTransition(.opacity)
                }

                Spacer().frame(height: 12)

                StartButton(
                    daemonState: backend.daemonState.kind,
                    soundEnabled: false,
                    motionEnabled: !themeStore.reduceMotionPreference
                ) {
                    Task { await stopCoordinator.stopAll(reason: "welcome_loading_stop") }
                }

                if let lastLog = backend.logs.last {
                    Text(lastLog.message)
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundStyle(Brand.textHighlight.opacity(0.50))
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: 460)
                        .padding(.top, 6)
                }
            }
            .padding(.horizontal, 48)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(tr("MTPLX is loading"))
    }

    private var progressLabel: String {
        switch backend.daemonState.kind {
        case .starting: return tr("STARTING…")
        case .warming: return tr("LOADING MODEL…")
        case .stopping: return tr("STOPPING…")
        default: return tr("PLEASE WAIT…")
        }
    }
}

// MARK: - StartButton
//
// 120pt chrome ring with a centered power glyph. The visual centerpiece
// of the welcome state. Three states:
// - resting (.stopped)  — solid chrome ring, glyph dim until hover
// - hover               — chrome ring fills, glyph brightens, soft halo
// - press               — spring scale down, optional click sound
// - warming             — phase-animated breathing pulse, click → stop

struct StartButton: View {
    let daemonState: DaemonStateKind
    let soundEnabled: Bool
    let motionEnabled: Bool
    let action: () -> Void

    @State private var hovering: Bool = false
    @State private var pressing: Bool = false

    var body: some View {
        Button(action: handleClick) {
            ZStack {
                outerHalo
                ring
                glyph
            }
            .frame(width: 132, height: 132)
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .scaleEffect(pressing ? 0.94 : (hovering ? 1.04 : 1.0))
        .animation(motionEnabled ? .spring(response: 0.30, dampingFraction: 0.7) : nil,
                   value: hovering)
        .animation(motionEnabled ? .spring(response: 0.18, dampingFraction: 0.6) : nil,
                   value: pressing)
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in pressing = true }
                .onEnded { _ in pressing = false }
        )
        .accessibilityLabel(accessibilityLabel)
    }

    @ViewBuilder
    private var outerHalo: some View {
        Circle()
            .fill(
                RadialGradient(
                    colors: [
                        Brand.accent.opacity(hovering ? 0.16 : 0.06),
                        Brand.accent.opacity(0.0)
                    ],
                    center: .center,
                    startRadius: 30,
                    endRadius: 120
                )
            )
            .scaleEffect(isWarming ? 1.0 : (hovering ? 1.05 : 1.0))
            .phaseAnimator(
                isWarming ? [0.85, 1.05, 0.85] : [1.0, 1.0, 1.0],
                trigger: isWarming
            ) { content, scale in
                content.scaleEffect(scale)
            } animation: { _ in
                motionEnabled
                    ? .easeInOut(duration: 1.6)
                    : nil
            }
    }

    @ViewBuilder
    private var ring: some View {
        Circle()
            .fill(Brand.raisedSurface)
            .overlay {
                Circle()
                    .stroke(
                        Brand.chromeFill,
                        lineWidth: 4
                    )
            }
            .overlay {
                Circle()
                    .stroke(
                        Brand.shineGradient,
                        lineWidth: 1.5
                    )
                    .blendMode(.overlay)
                    .opacity(0.6)
            }
            .frame(width: 120, height: 120)
            .shadow(
                color: Brand.Depth.ambient.color,
                radius: 16,
                x: 0,
                y: 8
            )
            .shadow(
                color: Brand.accent.opacity(hovering ? 0.25 : 0.08),
                radius: 24,
                x: 0,
                y: 0
            )
    }

    @ViewBuilder
    private var glyph: some View {
        Image(systemName: glyphName)
            .font(.system(size: 44, weight: .semibold))
            .foregroundStyle(Brand.chromeFill)
            .symbolRenderingMode(.monochrome)
            .opacity(glyphOpacity)
            .scaleEffect(isWarming ? 0.92 : 1.0)
            .phaseAnimator(
                isWarming ? [0.85, 1.0, 0.85] : [1.0, 1.0, 1.0],
                trigger: isWarming
            ) { content, opacity in
                content.opacity(opacity)
            } animation: { _ in
                motionEnabled ? .easeInOut(duration: 1.6) : nil
            }
    }

    private var glyphName: String {
        switch daemonState {
        case .stopped, .crashed, .degraded: return "power"
        case .starting, .warming: return "circle.dotted"
        case .running, .stopping: return "stop.fill"
        }
    }

    private var glyphOpacity: Double {
        if isWarming { return 0.85 }
        return hovering ? 1.0 : 0.78
    }

    private var isWarming: Bool {
        switch daemonState {
        case .starting, .warming: return true
        default: return false
        }
    }

    private var accessibilityLabel: String {
        switch daemonState {
        case .stopped: return tr("Start MTPLX")
        case .crashed: return tr("Restart MTPLX after crash")
        case .degraded: return tr("Restart MTPLX")
        case .starting: return tr("Starting MTPLX")
        case .warming: return tr("Loading model")
        case .running, .stopping: return tr("Stop MTPLX")
        }
    }

    private func handleClick() {
        if soundEnabled {
            NSSound(named: NSSound.Name("Tink"))?.play()
        }
        action()
    }
}
