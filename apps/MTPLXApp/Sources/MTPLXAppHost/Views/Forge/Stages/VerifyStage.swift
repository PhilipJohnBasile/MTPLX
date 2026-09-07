import SwiftUI
import AppKit
import MTPLXAppCore

// MARK: - VerifyStage
//
// Measures AR + D1/D2/D3 acceptance on the freshly forged artifact
// by relying on the backend's `mtplx forge build` pipeline (which
// itself shells `mtplx tune` internally). The frontend's job is to:
//
//   1. Render a four-row checklist (AR, D1, D2, D3) that flips
//      checked as each ForgeVerifyRow lands via the orchestrator's
//      `verifyRows` map.
//   2. Once every row has landed (or when state.verification is
//      populated from a completed build), reveal the
//      AcceptanceRevealTPSPanel + AcceptanceBarGrid extracted to
//      Views/Common/AcceptanceReveal.swift.
//
// Cancel pill in the footer cancels the whole build, not just
// verify — the artifact only becomes usable after verify, so abandoning
// here means abandoning the build.

struct VerifyStage: View {
    @EnvironmentObject private var orchestrator: ForgeOrchestrator

    var body: some View {
        ForgeStageShell(
            title: tr("Measuring real speed"),
            subtitle: tr("Running your new model on this Mac to see how fast it actually is."),
            step: .verify,
            symbol: "checkmark.seal.fill",
            symbolTint: Brand.success
        ) {
            VStack(alignment: .leading, spacing: 16) {
                if let outcome = failedSpeedOutcome {
                    failedSpeedCard(outcome)
                        .transition(.opacity)
                } else if let data = revealData {
                    revealCard(data)
                        .transition(.opacity)
                } else {
                    checklistView
                        .transition(.opacity)
                }
                if failedSpeedOutcome == nil, let failure = orchestrator.buildFailure {
                    ForgeFailureBanner(message: failure)
                }
                Spacer(minLength: 0)
            }
            .animation(.smooth(duration: 0.30), value: revealData)
            .animation(.smooth(duration: 0.30), value: failedSpeedOutcome)
        } footer: {
            footerButton
        }
    }

    // MARK: - Live checklist

    private var checklistView: some View {
        ForgePhaseChecklist(
            heading: "Verify candidates",
            rows: [
                ForgePhaseRow(
                    label: tr("Baseline"),
                    state: candidateState(depth: 0),
                    detail: detail(for: 0)
                ),
                ForgePhaseRow(
                    label: tr("Depth 1"),
                    state: candidateState(depth: 1),
                    detail: detail(for: 1)
                ),
                ForgePhaseRow(
                    label: tr("Depth 2"),
                    state: candidateState(depth: 2),
                    detail: detail(for: 2)
                ),
                ForgePhaseRow(
                    label: tr("Depth 3"),
                    state: candidateState(depth: 3),
                    detail: detail(for: 3)
                )
            ]
        )
    }

    private func candidateState(depth: Int) -> ForgePhaseRowState {
        if orchestrator.verifyRows[depth] != nil { return .done }
        // The first depth with no landed row is in-progress; the rest
        // are pending. We can't infer this perfectly without
        // backend-side ordering, but greedy "first missing == active"
        // matches how the build actually runs (one candidate at a time
        // for accurate timing under max fans).
        if orchestrator.isBuilding {
            for d in 0..<depth where orchestrator.verifyRows[d] == nil {
                return .pending
            }
            return .inProgress
        }
        return .pending
    }

    private func detail(for depth: Int) -> String? {
        guard let row = orchestrator.verifyRows[depth] else { return nil }
        let acceptance = row.acceptanceByPosition.isEmpty
            ? 0
            : row.acceptanceByPosition.reduce(0, +) / Double(row.acceptanceByPosition.count)
        return String(
            format: "%.1f tok/s · mean accept %.0f%% · verify %.1fs",
            row.tokS, acceptance * 100, row.verifyTimeSeconds
        )
    }

    // MARK: - Reveal card

    private var revealData: AcceptanceRevealData? {
        if let outcome = orchestrator.buildOutcome, !outcome.isSpeedWin {
            return nil
        }
        // Prefer the orchestrator's structured verification when the
        // build has completed; otherwise synthesise one from the live
        // verifyRows so the reveal can light up the moment every
        // candidate has landed (even before forge.json is read).
        if let verification = orchestrator.state.verification,
           let bestTokS = verification.tokSByDepth.values.max(),
           bestTokS > 0
        {
            return AcceptanceRevealData.from(verification)
        }
        let rowsByDepth = orchestrator.verifyRows
        let requiredDepths = [0, 1, 2, 3]
        guard requiredDepths.allSatisfy({ rowsByDepth[$0] != nil }),
              let ar = rowsByDepth[0],
              ar.tokS > 0
        else { return nil }
        let mtpDepths = [1, 2, 3]
        guard let bestDepth = mtpDepths.max(by: { (rowsByDepth[$0]?.tokS ?? 0) < (rowsByDepth[$1]?.tokS ?? 0) }),
              let bestRow = rowsByDepth[bestDepth]
        else { return nil }
        let multiplierVsAr = ar.tokS > 0 ? bestRow.tokS / ar.tokS : 1
        let winningDepth = multiplierVsAr > 1.0 ? bestDepth : 0
        let synthVerification = ForgeVerification(
            arTokS: ar.tokS,
            tokSByDepth: Dictionary(uniqueKeysWithValues: mtpDepths.compactMap { d in
                guard let r = rowsByDepth[d] else { return nil }
                return (d, r.tokS)
            }),
            acceptanceByDepth: Dictionary(uniqueKeysWithValues: mtpDepths.compactMap { d in
                guard let r = rowsByDepth[d] else { return nil }
                return (d, r.acceptanceByPosition)
            }),
            bestDepth: winningDepth,
            multiplierVsAr: winningDepth == 0 ? 1.0 : multiplierVsAr,
            verifiedOnHardware: orchestrator.state.hardware?.chipName ?? tr("Apple Silicon"),
            sampler: ForgeSampler()
        )
        return AcceptanceRevealData.from(synthVerification)
    }

    @ViewBuilder
    private var footerButton: some View {
        if failedSpeedOutcome != nil, !orchestrator.isBuilding {
            HStack(spacing: 10) {
                ForgePrimaryButton("Retry verify", icon: "arrow.clockwise", isEnabled: true) {
                    orchestrator.startBuild()
                }
                Button {
                    openDiagnostics()
                } label: {
                    Label(tr("Diagnostics"), systemImage: "folder")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Brand.typeBody)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 9)
                        .background(
                            Capsule(style: .continuous)
                                .fill(Brand.cardSurface)
                                .overlay(
                                    Capsule(style: .continuous)
                                        .stroke(Brand.separator, lineWidth: 0.5)
                                )
                        )
                }
                .buttonStyle(.plain)
                .disabled(orchestrator.buildRunDir == nil)
                ForgePrimaryButton("Discard", icon: "trash", isEnabled: true) {
                    orchestrator.resetWizard()
                }
            }
        } else if revealData != nil, !orchestrator.isBuilding, orchestrator.completedLocalPath != nil {
            ForgePrimaryButton(
                tr("Continue"),
                icon: "arrow.right",
                isEnabled: orchestrator.state.hasSpeedWinningVerification
            ) {
                orchestrator.continueAfterVerify()
            }
        } else {
            ForgePrimaryButton(
                tr("Cancel build"),
                icon: "xmark",
                isEnabled: orchestrator.isBuilding
            ) {
                orchestrator.cancelBuild()
            }
        }
    }

    private var failedSpeedOutcome: ForgeBuildOutcome? {
        guard let outcome = orchestrator.buildOutcome, !outcome.isSpeedWin else { return nil }
        return outcome
    }

    private func openDiagnostics() {
        guard let runDir = orchestrator.buildRunDir else { return }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: runDir)])
    }

    @ViewBuilder
    private func failedSpeedCard(_ outcome: ForgeBuildOutcome) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(Brand.warning)
                VStack(alignment: .leading, spacing: 5) {
                    Text(tr("Converted, not accelerated"))
                        .font(.system(.title3, design: .rounded).weight(.semibold))
                        .foregroundStyle(Brand.typeHi)
                    Text(outcome.message)
                        .font(.system(size: 12))
                        .foregroundStyle(Brand.typeSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            failedSpeedRows(outcome)
            if let path = outcome.convertedPath {
                Text(path.replacingOccurrences(of: NSHomeDirectory(), with: "~"))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
                    .truncationMode(.middle)
                    .lineLimit(1)
            }
            Text(tr("No picker registration was made."))
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Brand.typeBody)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Brand.warning.opacity(0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Brand.warning.opacity(0.40), lineWidth: 0.5)
                )
        )
    }

    @ViewBuilder
    private func failedSpeedRows(_ outcome: ForgeBuildOutcome) -> some View {
        VStack(spacing: 0) {
            ForEach(outcome.verifyRows.sorted(by: { $0.depth < $1.depth }), id: \.depth) { row in
                HStack(spacing: 10) {
                    Text(row.depth == 0 ? tr("Base") : tr("D%lld", row.depth))
                        .font(.system(size: 12, weight: .bold, design: .monospaced))
                        .foregroundStyle(row.depth == 0 ? Brand.typeHi : Brand.warning)
                        .frame(width: 34, alignment: .leading)
                    Text(tr("%.2f tok/s", row.tokS))
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Brand.typeBody)
                        .frame(width: 96, alignment: .leading)
                    Text(String(format: "%.2fx", row.multiplierVsAr))
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(row.multiplierVsAr > 1.0 ? Brand.success : Brand.typeSecondary)
                        .frame(width: 56, alignment: .leading)
                    Text(acceptanceText(row))
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Brand.typeSecondary)
                    Spacer(minLength: 0)
                }
                .padding(.vertical, 8)
                if row.depth != outcome.verifyRows.map(\.depth).max() {
                    Rectangle()
                        .fill(Brand.separator.opacity(0.7))
                        .frame(height: 0.5)
                }
            }
        }
        .padding(.horizontal, 12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.bgOuter.opacity(0.40))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    private func acceptanceText(_ row: ForgeVerifyRow) -> String {
        guard row.depth > 0 else { return tr("baseline") }
        guard !row.acceptanceByPosition.isEmpty else { return tr("accept n/a") }
        let parts = row.acceptanceByPosition.enumerated().map { index, value in
            tr("p%lld %lld%%", index + 1, Int((value * 100).rounded()))
        }
        return parts.joined(separator: " · ")
    }

    /// Reveal card is a 3-act composition: headline → BIG hero
    /// (BEFORE → NOW + multiplier) → per-depth acceptance breakdown
    /// as supporting evidence. The earlier layout buried the hero
    /// under a small bar grid + a duplicate metrics table, which
    /// flattened the celebratory moment and produced the stray
    /// horizontal-band visual that read as a "random white line."
    @ViewBuilder
    private func revealCard(_ data: AcceptanceRevealData) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            Text(data.headline)
                .font(.system(.title3, design: .rounded).weight(.semibold))
                .foregroundStyle(Brand.typeHi)
                .frame(maxWidth: .infinity, alignment: .leading)
            AcceptanceRevealTPSPanel(data: data)
                .padding(.vertical, 4)
            Rectangle()
                .fill(Brand.separator)
                .frame(height: 0.5)
            AcceptanceBarGrid(data: data)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }
}
