import SwiftUI
import MTPLXAppCore

// MARK: - BenchEmptyState
//
// The pre-run hero (function chip + headline + paragraph + Run AIME
// 2026 CTA) followed by an optional previous-runs list. Hero chip
// flips from cool-blue to the polished chromeAccent so the empty
// state reads as Jet Chrome; the primary CTA routes through
// MTPLXPillButton for cross-app visual coherence.

struct BenchEmptyState: View {
    let history: [BenchRunSummary]
    let runTitle: String
    let runIcon: String
    let isRunEnabled: Bool
    let onRun: () -> Void
    let onQuickCheck: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            heroBlock
            if !history.isEmpty {
                historyBlock
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var heroBlock: some View {
        HStack(alignment: .top, spacing: 22) {
            VStack(alignment: .leading, spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: Brand.Radii.l, style: .continuous)
                        .fill(Brand.accentChrome.opacity(0.12))
                        .overlay {
                            RoundedRectangle(cornerRadius: Brand.Radii.l, style: .continuous)
                                .strokeBorder(Brand.accentChrome.opacity(0.30), lineWidth: Brand.hairline)
                        }
                    Image(systemName: "function")
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundStyle(Brand.accentChrome)
                        .symbolRenderingMode(.hierarchical)
                }
                .frame(width: 56, height: 56)

                Text(tr("Run AIME 2026"))
                    .font(.system(size: 22, weight: .heavy, design: .rounded))
                    .foregroundStyle(Brand.typeBody)

                Text(tr("Watch your Mac solve all 30 AIME 2026 problems live — reasoning, decode speed, and a per-question score grid as it goes."))
                    .font(.system(size: 13, weight: .regular))
                    .foregroundStyle(Brand.typeSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 520, alignment: .leading)

                HStack(spacing: 10) {
                    Button(action: onQuickCheck) {
                        HStack(spacing: 8) {
                            Image(systemName: "bolt.fill")
                                .font(.system(size: 11, weight: .heavy))
                            Text(tr("Quick Check"))
                        }
                    }
                    .buttonStyle(.mtplxPrimary)
                    .disabled(!isRunEnabled)
                    .accessibilityHint(tr("Runs the first 3 AIME 2026 problems as a bounded health check."))

                    Button(action: onRun) {
                        HStack(spacing: 8) {
                            Image(systemName: runIcon)
                                .font(.system(size: 11, weight: .heavy))
                            Text(runTitle)
                        }
                    }
                    .buttonStyle(.mtplxGhost)
                    .disabled(!isRunEnabled)
                    .accessibilityHint(tr("Runs all 30 AIME 2026 problems."))
                }
                .padding(.top, 4)
            }
            Spacer(minLength: 0)
        }
    }

    private var historyBlock: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(tr("PREVIOUS RUNS"))
                    .font(.system(size: 10, weight: .heavy, design: .monospaced))
                    .tracking(1.5)
                    .foregroundStyle(Brand.typeTertiary)
                Spacer()
                Text(tr("Saved on your Mac"))
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
            }

            VStack(spacing: 0) {
                ForEach(Array(history.enumerated()), id: \.element.id) { idx, run in
                    if idx > 0 {
                        Divider().overlay(Brand.separator)
                    }
                    BenchHistoryRow(run: run)
                }
            }
            .background {
                RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous)
                    .fill(Brand.cardSurface)
                    .overlay {
                        RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous)
                            .strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                    }
            }
        }
    }
}
