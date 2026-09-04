import SwiftUI
import MTPLXAppCore

// MARK: - BenchQuestionDetail
//
// Review card for a finished AIME question, reached by tapping a tile in
// the question grid. Shows a satisfying correct / wrong reveal (the
// model's answer vs the expected answer), the problem statement, and the
// full reasoning transcript captured live during the run.
//
// Jet Chrome: piano-black surfaces, off-white type, semantic status
// accents (success / danger / warning) reserved for the verdict only —
// no purple, no AI-slop gradient. The verdict hero springs in on appear
// for a bit of dopamine without turning the panel into a toy.

struct BenchQuestionDetail: View {
    let result: BenchQuestionResult
    let total: Int
    let onClose: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var revealed = false

    private var displayTotal: Int {
        max(total, result.idx)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(Brand.separator)
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    verdictHero
                    problemBlock
                    reasoningBlock
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(width: 680, height: 660)
        .background(Brand.bgOuter)
        .onAppear {
            guard !reduceMotion else { revealed = true; return }
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7).delay(0.05)) {
                revealed = true
            }
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(tr("Problem %lld of %lld", result.idx, displayTotal))
                    .font(.system(size: 15, weight: .heavy, design: .rounded))
                    .foregroundStyle(Brand.typeBody)
                Text("\(result.problem.set) #\(result.problem.index)")
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(Brand.typeSecondary)
            }
            Spacer()
            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .heavy))
                    .foregroundStyle(Brand.typeBody)
                    .frame(width: 30, height: 30)
                    .background {
                        Circle()
                            .fill(Brand.cardSurface)
                            .overlay {
                                Circle().strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                            }
                    }
            }
            .buttonStyle(.plain)
            .accessibilityLabel(tr("Close"))
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .background(Brand.bgInner)
    }

    // MARK: Verdict hero

    private var verdictHero: some View {
        HStack(spacing: 18) {
            ZStack {
                Circle()
                    .fill(accent.opacity(0.14))
                    .overlay {
                        Circle().strokeBorder(accent.opacity(0.35), lineWidth: Brand.hairline)
                    }
                Image(systemName: verdictGlyph)
                    .font(.system(size: 30, weight: .bold))
                    .foregroundStyle(accent)
                    .symbolRenderingMode(.hierarchical)
            }
            .frame(width: 72, height: 72)
            .scaleEffect(revealed ? 1 : 0.6)
            .opacity(revealed ? 1 : 0)

            VStack(alignment: .leading, spacing: 8) {
                Text(verdictTitle)
                    .font(.system(size: 20, weight: .heavy, design: .rounded))
                    .foregroundStyle(accent)
                HStack(alignment: .top, spacing: 10) {
                    answerChip(label: tr("ANSWER"), value: result.extracted.map(String.init) ?? "—", tint: accent)
                    answerChip(label: tr("EXPECTED"), value: String(result.problem.answer), tint: Brand.typeSecondary)
                    if let ms = result.durationMs {
                        answerChip(label: tr("TIME"), value: formattedDuration(ms), tint: Brand.typeSecondary)
                    }
                }
            }
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilitySummary)
    }

    private func answerChip(label: String, value: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.2)
                .foregroundStyle(Brand.typeTertiary)
            Text(value)
                .font(.system(size: 24, weight: .heavy, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(tint)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(minWidth: 84, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                .fill(Brand.bgInner.opacity(0.6))
                .overlay {
                    RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                        .strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                }
        }
    }

    // MARK: Problem + reasoning

    private var problemBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel(tr("PROBLEM"))
            MathProblemRender(text: cleanedProblem, expanded: true)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
                .background {
                    RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous)
                        .fill(Brand.bgInner.opacity(0.6))
                        .overlay {
                            RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous)
                                .strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                        }
                }
        }
    }

    private var reasoningBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel(tr("REASONING"))
            if result.reasoning.isEmpty {
                Text(tr("The reasoning for this problem wasn't saved — reopen a problem you solved in this session to read its full working."))
                    .font(.system(size: 12, weight: .regular))
                    .foregroundStyle(Brand.typeTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                MathReasoningRender(text: result.reasoning)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(14)
                    .background {
                        RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous)
                            .fill(Brand.bgInner.opacity(0.6))
                            .overlay {
                                RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous)
                                    .strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                            }
                    }
            }
        }
    }

    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .heavy, design: .monospaced))
            .tracking(1.4)
            .foregroundStyle(Brand.typeTertiary)
    }

    // MARK: Derived

    private var accent: Color {
        switch result.status {
        case .correct: return Brand.success
        case .wrong: return Brand.danger
        case .abstain: return Brand.warning
        case .pending: return Brand.accentChrome
        }
    }

    private var verdictGlyph: String {
        switch result.status {
        case .correct: return "checkmark.seal.fill"
        case .wrong: return "xmark.octagon.fill"
        case .abstain: return "questionmark.diamond.fill"
        case .pending: return "hourglass"
        }
    }

    private var verdictTitle: String {
        switch result.status {
        case .correct: return tr("Correct")
        case .wrong: return tr("Wrong")
        case .abstain: return tr("No answer")
        case .pending: return tr("Not solved yet")
        }
    }

    private var accessibilitySummary: String {
        let answer = result.extracted.map(String.init) ?? tr("no answer")
        return tr("%@. Answer %@, expected %lld.", verdictTitle, answer, result.problem.answer)
    }

    /// Strips `[asy]...[/asy]` figure source so the math renderer never tries
    /// to typeset Asymptote markup (matches the live card's cleaning).
    private var cleanedProblem: String {
        let raw = result.problem.problem
        guard let range = raw.range(of: #"\[asy\][\s\S]*?\[/asy\]"#, options: .regularExpression) else {
            return raw
        }
        var cleaned = raw
        cleaned.replaceSubrange(range, with: "")
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func formattedDuration(_ ms: Int) -> String {
        let seconds = Double(ms) / 1000.0
        if seconds < 60 { return String(format: "%.0fs", seconds) }
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return tr("%lldm %llds", m, s)
    }
}
