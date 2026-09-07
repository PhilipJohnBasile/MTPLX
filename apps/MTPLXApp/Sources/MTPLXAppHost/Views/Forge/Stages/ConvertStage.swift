import SwiftUI
import MTPLXAppCore

// MARK: - ConvertStage
//
// Live progress while `mtplx forge build` is in its download +
// body-quantisation phase. Three substages rendered as a small
// checklist that flips items from spinning to checked as the
// orchestrator's per-phase progress lands:
//
//   • Download source       — driven by orchestrator.downloadProgress
//   • Convert to MLX        — driven by orchestrator.convertPhases[.convert].label == "to_mlx"
//   • Quantise body         — driven by orchestrator.convertPhases[.convert].label == "quantize_body"
//
// The download bar is the same shape as DownloadStep.swift's; the
// substage checklist is the new ingredient. Both surfaces share the
// raisedSurface card chrome with a thin separator inside.

struct ConvertStage: View {
    @EnvironmentObject private var orchestrator: ForgeOrchestrator

    var body: some View {
        ForgeStageShell(
            title: tr("Converting to MLX"),
            subtitle: tr("Downloading the model and converting it to run on your Mac. You can cancel anytime — your progress is saved."),
            step: .convert,
            symbol: "arrow.triangle.2.circlepath",
            symbolTint: Brand.accentChrome
        ) {
            VStack(alignment: .leading, spacing: 16) {
                downloadCard
                substageChecklist
                if let failure = orchestrator.buildFailure {
                    ForgeFailureBanner(message: failure)
                }
                Spacer(minLength: 0)
            }
        } footer: {
            ForgePrimaryButton(
                tr("Cancel build"),
                icon: "xmark",
                isEnabled: orchestrator.isBuilding
            ) {
                orchestrator.cancelBuild()
            }
        }
    }

    // MARK: - Download card

    @ViewBuilder
    private var downloadCard: some View {
        ForgePhaseCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 6) {
                    Text(tr("DOWNLOAD"))
                        .font(.system(size: 10, weight: .heavy, design: .monospaced))
                        .tracking(1.5)
                        .foregroundStyle(Brand.typeTertiary)
                    Spacer(minLength: 0)
                    if let progress = orchestrator.downloadProgress,
                       let total = progress.totalBytes,
                       total > 0
                    {
                        Text(tr("%@ of %@", formatBytesShort(progress.bytesOnDisk), formatBytesShort(total)))
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundStyle(Brand.typeSecondary)
                    } else if let progress = orchestrator.downloadProgress {
                        Text(formatBytesShort(progress.bytesOnDisk))
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundStyle(Brand.typeSecondary)
                    }
                }
                progressBar
                telemetryRow
            }
        }
    }

    private var progressBar: some View {
        ForgeLinearProgressBar(
            fraction: downloadFraction,
            height: 8,
            minimumFillWidth: 8
        )
        .accessibilityLabel(tr("Download progress"))
        .accessibilityValue(tr("%lld percent", Int(downloadFraction * 100)))
    }

    private var downloadFraction: Double {
        guard let progress = orchestrator.downloadProgress,
              let total = progress.totalBytes,
              total > 0
        else { return 0 }
        return min(1, Double(progress.bytesOnDisk) / Double(total))
    }

    private var telemetryRow: some View {
        let progress = orchestrator.downloadProgress
        let rate = progress?.bytesPerSecond ?? 0
        let eta = progress?.etaSeconds
        let status = downloadStatusText(progress)
        return HStack(spacing: 14) {
            Text(status)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(rate > 50_000 ? Brand.typeSecondary : Brand.typeTertiary)
            if rate > 0, let eta, eta > 0 {
                Text(tr("ETA %@", formatDuration(eta)))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
            }
            Spacer(minLength: 0)
        }
    }

    private func downloadStatusText(_ progress: ForgeDownloadProgress?) -> String {
        guard let progress else { return "-" }
        if progress.bytesPerSecond > 50_000 {
            return formatRate(progress.bytesPerSecond)
        }
        let label = progress.label?.replacingOccurrences(of: "_", with: " ") ?? "waiting"
        if let stalled = progress.stalledSeconds, stalled >= 2 {
            return "\(label) \(formatDuration(stalled))"
        }
        return label
    }

    // MARK: - Sub-phase checklist (Convert)

    private var substageChecklist: some View {
        let phase = orchestrator.convertPhases[.convert]
        return ForgePhaseChecklist(
            heading: "Steps",
            rows: [
                ForgePhaseRow(
                    label: tr("Download model"),
                    state: downloadRowState
                ),
                ForgePhaseRow(
                    label: tr("Convert to MLX"),
                    state: phaseRowState(phase: phase, matchingLabel: "to_mlx", anyProgress: true)
                ),
                ForgePhaseRow(
                    label: tr("Compress for size"),
                    state: phaseRowState(phase: phase, matchingLabel: "quantize_body", anyProgress: false)
                )
            ]
        )
    }

    private var downloadRowState: ForgePhaseRowState {
        guard let progress = orchestrator.downloadProgress else {
            return orchestrator.buildPhase == .download ? .inProgress : .pending
        }
        if let total = progress.totalBytes, total > 0, progress.bytesOnDisk >= total {
            return .done
        }
        if progress.bytesOnDisk > 0 { return .inProgress }
        return .pending
    }

    private func phaseRowState(
        phase: ForgePhaseProgress?,
        matchingLabel: String,
        anyProgress: Bool
    ) -> ForgePhaseRowState {
        guard let phase else { return .pending }
        if phase.finished { return .done }
        if phase.label?.lowercased() == matchingLabel { return .inProgress }
        // anyProgress: render this row as in-progress whenever the
        // generic convert phase is active and the more specific
        // label hasn't been emitted yet. Used for the first row in
        // the chain.
        if anyProgress && phase.progress > 0 && phase.progress < 1 { return .inProgress }
        if phase.progress >= 1 { return .done }
        return .pending
    }

    // MARK: - Formatters

    private func formatBytesShort(_ bytes: Int64) -> String {
        let gib = Double(bytes) / 1_073_741_824.0
        if gib >= 1 { return tr("%.2f GB", gib) }
        let mib = Double(bytes) / 1_048_576.0
        return tr("%.0f MB", mib)
    }

    private func formatRate(_ bytesPerSecond: Double) -> String {
        if bytesPerSecond < 1024 { return "—" }
        let mbps = bytesPerSecond / 1_048_576.0
        return tr("%.1f MB/s", mbps)
    }

    private func formatDuration(_ seconds: Double) -> String {
        let total = Int(max(0, seconds))
        let h = total / 3600
        let m = (total % 3600) / 60
        let s = total % 60
        if h > 0 { return tr("%lldh %lldm", h, m) }
        if m > 0 { return tr("%lldm %llds", m, s) }
        return tr("%llds", s)
    }
}

// MARK: - Shared substage primitives
//
// Used by both ConvertStage and CalibrateStage. Kept here rather than
// in a separate Components file because they're small and only have
// two consumers.

public enum ForgePhaseRowState: Equatable, Sendable {
    case pending
    case inProgress
    case done
    case failed
}

public struct ForgePhaseRow: Identifiable, Equatable, Sendable {
    public let id = UUID()
    public var label: String
    public var state: ForgePhaseRowState
    public var detail: String?

    public init(label: String, state: ForgePhaseRowState, detail: String? = nil) {
        self.label = label
        self.state = state
        self.detail = detail
    }
}

struct ForgePhaseChecklist: View {
    let heading: String
    let rows: [ForgePhaseRow]

    var body: some View {
        ForgePhaseCard {
            VStack(alignment: .leading, spacing: 12) {
                Text(heading.uppercased())
                    .font(.system(size: 10, weight: .heavy, design: .monospaced))
                    .tracking(1.5)
                    .foregroundStyle(Brand.typeTertiary)
                VStack(spacing: 8) {
                    ForEach(rows) { row in
                        ForgePhaseRowView(row: row)
                    }
                }
            }
        }
    }
}

struct ForgePhaseRowView: View {
    let row: ForgePhaseRow

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            stateGlyph
            VStack(alignment: .leading, spacing: 2) {
                Text(row.label)
                    .font(.system(size: 12, weight: row.state == .inProgress ? .semibold : .medium))
                    .foregroundStyle(textTint)
                if let detail = row.detail {
                    Text(detail)
                        .font(.caption2)
                        .foregroundStyle(Brand.typeTertiary)
                }
            }
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder
    private var stateGlyph: some View {
        switch row.state {
        case .pending:
            Circle()
                .strokeBorder(Brand.separator, lineWidth: 1)
                .frame(width: 14, height: 14)
        case .inProgress:
            ProgressView()
                .controlSize(.mini)
                .progressViewStyle(.circular)
                .frame(width: 14, height: 14)
        case .done:
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(Brand.success)
        case .failed:
            Image(systemName: "xmark.octagon.fill")
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(Brand.danger)
        }
    }

    private var textTint: Color {
        switch row.state {
        case .pending: return Brand.typeTertiary
        case .inProgress: return Brand.typeHi
        case .done: return Brand.typeBody
        case .failed: return Brand.danger
        }
    }
}

struct ForgePhaseCard<Content: View>: View {
    @ViewBuilder let content: () -> Content

    var body: some View {
        content()
            .padding(12)
            .frame(width: ForgeStageLayout.contentWidth, alignment: .leading)
            .clipped()
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Brand.bgInner.opacity(0.6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .stroke(Brand.separator, lineWidth: 0.5)
                    )
            )
    }
}

struct ForgeLinearProgressBar: View {
    let fraction: Double
    var height: CGFloat
    var minimumFillWidth: CGFloat
    var fill: Color = Brand.accentChrome
    var track: Color = Brand.separator.opacity(0.4)

    var body: some View {
        GeometryReader { geometry in
            let width = max(0, geometry.size.width)
            let clampedFraction = min(1, max(0, fraction))
            let fillWidth = clampedFraction <= 0
                ? minimumFillWidth
                : min(width, max(minimumFillWidth, width * CGFloat(clampedFraction)))
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(track)
                    .frame(width: width, height: height)
                Capsule()
                    .fill(fill)
                    .frame(width: min(width, fillWidth), height: height)
                    .animation(.easeInOut(duration: 0.3), value: clampedFraction)
            }
        }
        .frame(width: ForgeStageLayout.contentWidth - 24, height: height, alignment: .leading)
        .clipped()
    }
}

struct ForgeFailureBanner: View {
    let message: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Brand.danger)
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 4) {
                Text(tr("Build failed"))
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                Text(message)
                    .font(.caption2)
                    .foregroundStyle(Brand.typeSecondary)
                    .lineLimit(6)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.danger.opacity(0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.danger.opacity(0.40), lineWidth: 0.5)
                )
        )
    }
}
