import SwiftUI
import MTPLXAppCore

// MARK: - CalibrateStage
//
// Live progress while `mtplx forge build` is in its MTP head
// packaging phase. Substages mirror what build_flat4_cyankiwi
// _mtp_requant.py does internally — the reference pipeline the
// backend agent is generalising:
//
//   • Extract MTP weights  — pull mtp.fc + mtp transformer layer
//   • Requantise MTP body  — applies bit / group-size choice
//   • Pack sidecar         — writes mtp.safetensors + config patch
//
// When the backend ships loss / PPL metrics from a real calibration
// pass (DWQ-style refit; not in V1 reference pipeline), they surface
// underneath the checklist as a thin metric row.

struct CalibrateStage: View {
    @EnvironmentObject private var orchestrator: ForgeOrchestrator

    var body: some View {
        ForgeStageShell(
            title: tr("Calibrating"),
            subtitle: tr("Packing the speed-prediction head. We'll measure the real speed up next."),
            step: .calibrate,
            symbol: "slider.horizontal.3",
            symbolTint: Brand.accentChrome
        ) {
            VStack(alignment: .leading, spacing: 16) {
                substageChecklist
                metricsRow
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

    // MARK: - Sub-phase checklist (Calibrate)

    private var substageChecklist: some View {
        let phase = orchestrator.convertPhases[.calibrate]
        return ForgePhaseChecklist(
            heading: "Calibrate pipeline",
            rows: [
                ForgePhaseRow(
                    label: tr("Extract MTP weights"),
                    state: rowState(phase: phase, matchingLabel: "extract_mtp", anyProgress: true)
                ),
                ForgePhaseRow(
                    label: tr("Requantise MTP body"),
                    state: rowState(phase: phase, matchingLabel: "requantize_mtp", anyProgress: false)
                ),
                ForgePhaseRow(
                    label: tr("Pack sidecar"),
                    state: rowState(phase: phase, matchingLabel: "pack_sidecar", anyProgress: false)
                )
            ]
        )
    }

    private func rowState(
        phase: ForgePhaseProgress?,
        matchingLabel: String,
        anyProgress: Bool
    ) -> ForgePhaseRowState {
        guard let phase else { return .pending }
        if phase.finished { return .done }
        if phase.label?.lowercased() == matchingLabel { return .inProgress }
        if anyProgress && phase.progress > 0 && phase.progress < 1 { return .inProgress }
        if phase.progress >= 1 { return .done }
        return .pending
    }

    // MARK: - Optional loss / PPL metrics

    @ViewBuilder
    private var metricsRow: some View {
        // Reserved space: when the backend ships DWQ-style calibration
        // with loss / PPL metrics, the orchestrator surfaces them via
        // a dedicated published property and this row renders them.
        // V1 reference pipeline (requant-only) doesn't, so this is a
        // no-op for now — but the slot stays in the layout to avoid
        // re-flowing the card when it eventually appears.
        if let phase = orchestrator.convertPhases[.calibrate],
           phase.progress > 0 && phase.progress < 1
        {
            ForgePhaseCard {
                HStack(spacing: 18) {
                    metricCell(label: tr("Progress"), value: percent(phase.progress))
                    if let label = phase.label, !label.isEmpty {
                        metricCell(label: tr("Phase"), value: humanizeLabel(label))
                    }
                    Spacer(minLength: 0)
                }
            }
        }
    }

    private func metricCell(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label.uppercased())
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1.2)
                .foregroundStyle(Brand.typeTertiary)
            Text(value)
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundStyle(Brand.typeHi)
                .monospacedDigit()
        }
    }

    private func percent(_ v: Double) -> String {
        String(format: "%.0f%%", v * 100)
    }

    private func humanizeLabel(_ raw: String) -> String {
        raw.split(separator: "_")
            .map { $0.capitalized }
            .joined(separator: " ")
    }
}
