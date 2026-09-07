import SwiftUI
import AppKit
import MTPLXAppCore

// MARK: - NewMaxToast
//
// Top-right toast that fires when a new all-time max TPS is observed.
// V1 styling: piano-black surface with chrome hairline and a warm-yellow
// bolt glyph. Soft-volume system sound (not the harsh beep) if the user
// enabled the toggle.

struct NewMaxToast: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore
    @State private var lastSeen: Double = 0
    @State private var current: ToastInfo? = nil
    @State private var dismissTask: Task<Void, Never>? = nil

    struct ToastInfo: Equatable {
        let value: Double
        let timestamp: Date
    }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.clear
            if let current {
                toastView(info: current)
                    .padding(.top, 56)
                    .padding(.trailing, 18)
                    .transition(
                        .move(edge: .top).combined(with: .opacity)
                    )
            }
        }
        .allowsHitTesting(false)
        .onChange(of: backend.rolling?.stickyAllTimeMax) { _, newValue in
            guard let max = newValue, max > lastSeen, max > 0 else {
                if let v = newValue, v > lastSeen { lastSeen = v }
                return
            }
            if lastSeen == 0 {
                lastSeen = max
                return
            }
            lastSeen = max
            present(value: max)
        }
    }

    private func toastView(info: ToastInfo) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "bolt.badge.automatic.fill")
                .font(.title3)
                .foregroundStyle(Brand.warning)
                .shadow(color: Brand.warning.opacity(0.5), radius: 8)
            VStack(alignment: .leading, spacing: 2) {
                Text(tr("NEW SPEED RECORD"))
                    .font(.system(size: 10, weight: .heavy, design: .monospaced))
                    .tracking(2)
                    .foregroundStyle(Brand.textHighlight)
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(Format.tps(info.value))
                        .font(.system(.title3, design: .rounded).weight(.heavy))
                        .monospacedDigit()
                        .foregroundStyle(Brand.chromeFill)
                    Text(tr("TPS"))
                        .font(.system(size: 10, weight: .heavy, design: .monospaced))
                        .tracking(1.5)
                        .foregroundStyle(Brand.textHighlight.opacity(0.65))
                }
            }
            Spacer(minLength: 4)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(width: 240)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(Brand.warning.opacity(0.55), lineWidth: 1)
                )
                .shadow(color: Brand.Depth.ambient.color, radius: 18, x: 0, y: 10)
        )
    }

    private func present(value: Double) {
        let info = ToastInfo(value: value, timestamp: Date())
        let motionEnabled = !backend.configuration.performanceLock && !themeStore.reduceMotionPreference
        if motionEnabled {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.85)) {
                current = info
            }
        } else {
            current = info
        }
        if themeStore.soundEnabled {
            // Quieter than NSSound.beep() — "Tink" is short and pleasant.
            NSSound(named: NSSound.Name("Tink"))?.play()
        }
        dismissTask?.cancel()
        dismissTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(2.4))
            if motionEnabled {
                withAnimation(.easeInOut(duration: 0.4)) {
                    current = nil
                }
            } else {
                current = nil
            }
        }
    }
}
