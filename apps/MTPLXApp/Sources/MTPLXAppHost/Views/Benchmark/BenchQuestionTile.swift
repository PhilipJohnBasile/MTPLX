import SwiftUI
import MTPLXAppCore

/// One tile in the question grid. State-driven Brand chrome + phase
/// animator pulse while running + Motion.metricBar landing +
/// `.levelChange` haptic when the result lands. Hover tooltip shows
/// expected/extracted. The "running" tile reads as a quiet
/// chromeAccent ring instead of the V0 cool-blue ring so the grid
/// stays on-palette during a run.
struct BenchQuestionTile: View {
    let result: BenchQuestionResult
    @EnvironmentObject private var themeStore: ThemeStore
    @State private var isHovering: Bool = false
    @State private var pulse: Int = 0

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                .fill(fillColor)
            RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                .strokeBorder(borderColor, lineWidth: borderWidth)
            content
        }
        .frame(height: 36)
        .modifier(BenchTilePulseModifier(active: isRunning, motionEnabled: !themeStore.reduceMotionPreference))
        .scaleEffect(isHovering ? 1.04 : 1.0)
        .onHover { hovering in
            withAnimation(
                themeStore.reduceMotionPreference
                    ? nil
                    : .spring(response: 0.22, dampingFraction: 0.78)
            ) {
                isHovering = hovering
            }
        }
        .help(tooltip)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityValue(accessibilityValue)
        .onChange(of: result.status) { old, new in
            // Fire a firm haptic the instant a question lands. Only
            // fire on a real flip from `.pending`.
            guard old == .pending, new != .pending else { return }
            Haptics.tick(.levelChange)
        }
        .animation(
            themeStore.reduceMotionPreference
                ? nil
                : .easeInOut(duration: 0.30),
            value: result.status
        )
    }

    @ViewBuilder
    private var content: some View {
        switch result.status {
        case .pending:
            Text("\(result.idx)")
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(0.3)
                .foregroundStyle(Brand.typeTertiary)
        case .correct:
            Image(systemName: "checkmark")
                .font(.system(size: 13, weight: .heavy))
                .foregroundStyle(Brand.success)
        case .wrong:
            Image(systemName: "xmark")
                .font(.system(size: 12, weight: .heavy))
                .foregroundStyle(Brand.danger)
        case .abstain:
            Image(systemName: "questionmark")
                .font(.system(size: 12, weight: .heavy))
                .foregroundStyle(Brand.warning)
        }
    }

    private var isRunning: Bool {
        // The currently-streaming question has empty status (.pending)
        // AND a populated problem text. The orchestrator clears the
        // status on questionStarted; non-yet-attempted tiles have empty
        // problem text.
        result.status == .pending && !result.problem.problem.isEmpty
    }

    private var fillColor: Color {
        switch result.status {
        case .pending:
            return isRunning
                ? Brand.accentChrome.opacity(0.12)
                : Brand.cardSurface
        case .correct: return Brand.success.opacity(0.14)
        case .wrong: return Brand.danger.opacity(0.12)
        case .abstain: return Brand.warning.opacity(0.10)
        }
    }

    private var borderColor: Color {
        switch result.status {
        case .pending:
            return isRunning ? Brand.accentChrome : Brand.separator
        case .correct: return Brand.success.opacity(0.55)
        case .wrong: return Brand.danger.opacity(0.55)
        case .abstain: return Brand.warning.opacity(0.55)
        }
    }

    private var borderWidth: CGFloat {
        switch result.status {
        case .pending where isRunning: return 1.2
        case .pending: return Brand.hairline
        default: return Brand.hairlineHeavy
        }
    }

    private var tooltip: String {
        let base = tr("Q%lld · %@ #%lld", result.idx, result.problem.set, result.problem.index)
        switch result.status {
        case .pending:
            return isRunning ? tr("%@ — solving…", base) : base
        case .correct:
            return tr("%@ — correct (answer %lld)", base, result.problem.answer)
        case .wrong:
            if let extracted = result.extracted {
                return tr("%@ — wrong: expected %lld, got %lld", base, result.problem.answer, extracted)
            }
            return tr("%@ — wrong: expected %lld", base, result.problem.answer)
        case .abstain:
            return tr("%@ — no parseable answer (expected %lld)", base, result.problem.answer)
        }
    }

    private var accessibilityLabel: String {
        tr("Problem %lld", result.idx)
    }

    private var accessibilityValue: String {
        switch result.status {
        case .pending: return isRunning ? tr("solving") : tr("pending")
        case .correct: return tr("correct")
        case .wrong:
            if let extracted = result.extracted {
                return tr("wrong, expected %lld, got %lld", result.problem.answer, extracted)
            }
            return tr("wrong")
        case .abstain: return tr("no answer")
        }
    }
}
