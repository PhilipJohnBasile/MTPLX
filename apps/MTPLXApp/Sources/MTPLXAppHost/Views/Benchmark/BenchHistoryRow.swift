import SwiftUI
import MTPLXAppCore

/// Single row in the previous-runs list. Score color uses
/// `acceptanceColor`-style banding (90%+ success, 70%+ chrome,
/// 50%+ warning, else danger) so the eye picks up wins and losses
/// without parsing the percentage. The 70% band swaps from cool-blue
/// to chromeAccent — Jet Chrome by default; only the failing bands
/// keep the loud semantic colors.
struct BenchHistoryRow: View {
    let run: BenchRunSummary

    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 2) {
                Text(formattedDate)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(Brand.typeBody)
                Text(formattedSubtitle)
                    .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .lineLimit(1)
            }
            Spacer()
            scorePill
            accuracyTag
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(historyAccessibilityLabel)
    }

    private var historyAccessibilityLabel: String {
        tr("Run %@. Score %lld of %lld. Accuracy %@.", formattedDate, run.score, run.total, accuracyLabel)
    }

    private var scorePill: some View {
        HStack(spacing: 6) {
            Text("\(run.score)")
                .font(.system(size: 14, weight: .heavy, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(scoreColor)
            Text("/ \(run.total)")
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background {
            Capsule().fill(scoreColor.opacity(0.10))
                .overlay { Capsule().strokeBorder(scoreColor.opacity(0.35), lineWidth: Brand.hairline) }
        }
    }

    private var accuracyTag: some View {
        Text(accuracyLabel)
            .font(.system(size: 10, weight: .heavy, design: .monospaced))
            .tracking(0.5)
            .foregroundStyle(Brand.typeTertiary)
            .frame(width: 48, alignment: .trailing)
    }

    private var scoreColor: Color {
        let rate = run.total > 0 ? Double(run.score) / Double(run.total) : 0
        switch rate {
        case 0.90...: return Brand.success
        case 0.70...: return Brand.accentChrome
        case 0.50...: return Brand.warning
        default: return Brand.danger
        }
    }

    private var accuracyLabel: String {
        guard let acc = run.accuracy else { return "—" }
        return Self.percentFormatter.string(from: NSNumber(value: acc)) ?? "—"
    }

    private static let percentFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .percent
        f.minimumFractionDigits = 0
        f.maximumFractionDigits = 0
        return f
    }()

    private var formattedDate: String {
        guard let ended = run.endedAt else { return run.runID }
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: ended)
    }

    private var formattedSubtitle: String {
        let modelShort = run.model
            .split(separator: "/").last.map(String.init) ?? run.model
        if let durationMs = run.durationMs {
            let mins = max(1, Int(round(Double(durationMs) / 60_000)))
            return tr("%@ · %lld min · state %@", modelShort, mins, run.state)
        }
        return tr("%@ · state %@", modelShort, run.state)
    }
}
