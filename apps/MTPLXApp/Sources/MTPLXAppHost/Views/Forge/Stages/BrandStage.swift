import SwiftUI
import MTPLXAppCore

// MARK: - BrandStage
//
// Final pre-registration review. The actual name is chosen on
// PlanStage before the backend starts so the artifact path, picker
// entry, runtime metadata, and default HF repo all agree. This stage
// is intentionally read-only: changing the name after conversion
// would lie about the folder that was already written.

struct BrandStage: View {
    @EnvironmentObject private var orchestrator: ForgeOrchestrator

    var body: some View {
        ForgeStageShell(
            title: tr("Review your model"),
            subtitle: tr("The MTPLX suffix is locked and this build has passed the speed gate."),
            step: .brand,
            symbol: "checkmark.seal.fill",
            symbolTint: Brand.success,
            onBack: { orchestrator.goBack() }
        ) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    nameSection
                    runtimeMetadataPreview
                    Spacer(minLength: 0)
                }
            }
        } footer: {
            ForgePrimaryButton(
                tr("Continue"),
                icon: "arrow.right",
                isEnabled: canContinue
            ) {
                orchestrator.confirmBrandAndContinue()
            }
        }
    }

    @ViewBuilder
    private var nameSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(tr("MODEL NAME"))
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Text(displayName)
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Brand.typeHi)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 0)
                    Text(tr("Ready"))
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(Brand.success)
                }
                Text(localPathPreview)
                    .font(.caption2)
                    .foregroundStyle(Brand.typeTertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Brand.bgInner.opacity(0.55))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .stroke(Brand.separator, lineWidth: 0.5)
                    )
            )
        }
    }

    @ViewBuilder
    private var runtimeMetadataPreview: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionHeader(tr("MTPLX_RUNTIME.JSON PREVIEW"))
            Group {
                if let metadata = orchestrator.brandedRuntimeMetadata {
                    RuntimeMetadataTable(json: metadata.rawJSON)
                } else {
                    Text(tr("Forge will register the model only after the verified MTPLX runtime metadata is available."))
                        .font(.caption2)
                        .foregroundStyle(Brand.typeTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(10)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Brand.bgInner.opacity(0.5))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .stroke(Brand.separator, lineWidth: 0.5)
                    )
            )
        }
    }

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .heavy, design: .monospaced))
            .tracking(1.5)
            .foregroundStyle(Brand.typeTertiary)
    }

    private var displayName: String {
        let raw = orchestrator.state.brand.brandedName.trimmingCharacters(in: .whitespacesAndNewlines)
        if !raw.isEmpty { return raw }
        if let probe = orchestrator.state.sourceProbe {
            return ForgeBrandInfo.derivedBrandedName(sourceRepo: probe.hfRepo, role: .speed)
        }
        return ForgeBrandInfo.resolvedBrandedName(userName: "Model")
    }

    private var localPathPreview: String {
        if let path = orchestrator.completedLocalPath, !path.isEmpty {
            return tr("Saved to %@", path.replacingOccurrences(of: NSHomeDirectory(), with: "~"))
        }
        return tr("Saved as %@", displayName)
    }

    private var canContinue: Bool {
        orchestrator.state.hasSpeedWinningVerification
            && !displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}
