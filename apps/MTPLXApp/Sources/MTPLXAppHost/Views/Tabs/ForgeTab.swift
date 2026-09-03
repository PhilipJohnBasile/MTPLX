import SwiftUI
import MTPLXAppCore

// MARK: - ForgeTab
//
// Shell for the MTP Forge — the dashboard surface that turns MTPLX
// from a runtime into the hub for MTP-on-MLX. Composes the segmented
// header (Create / Discover / My Models) on top of whichever sub-mode
// body is currently active. The wizard, the local browser, and the
// HF discover wall all live inside one tab so the user never leaves
// "I'm in the Forge" while switching between making, browsing, and
// publishing.
//
// When the `mtplx forge` backend subcommand isn't installed yet the
// orchestrator surfaces `backendUnavailable = true` and we render
// an explicit empty state rather than letting the wizard sit at the
// Source stage with a broken Next button.

struct ForgeTab: View {
    @EnvironmentObject private var orchestrator: ForgeOrchestrator
    @State private var orphanExpanded: Bool = false

    var body: some View {
        Group {
            if orchestrator.backendUnavailable {
                EmptyStateView(
                    symbol: "hammer.fill",
                    title: "Forge backend not available",
                    message: "Install or update MTPLX 1.x to use the model-creation surface. The rest of the app still works while Forge is offline."
                )
            } else {
                VStack(spacing: 0) {
                    ForgeTabHeader()
                    if !orchestrator.orphanRuns.isEmpty {
                        OrphanResumeBanner(
                            runs: orchestrator.orphanRuns,
                            expanded: $orphanExpanded,
                            onReveal: { orchestrator.revealOrphanRun($0) },
                            onDiscard: { orchestrator.discardOrphanRun($0) },
                            onDiscardAll: {
                                orchestrator.discardOrphanRuns()
                                orphanExpanded = false
                            }
                        )
                    }
                    Divider().overlay(Brand.separator)
                    Group {
                        switch orchestrator.state.subMode {
                        case .create:
                            ForgeCreateView()
                                .transition(.opacity)
                        case .discover:
                            ForgeDiscoverView()
                                .transition(.opacity)
                        case .mine:
                            ForgeMineView()
                                .transition(.opacity)
                        }
                    }
                    .animation(
                        orchestrator.isBuilding ? nil : .smooth(duration: 0.22),
                        value: orchestrator.state.subMode
                    )
                }
                .onAppear { orchestrator.scanForOrphanRuns() }
            }
        }
    }
}

// MARK: - OrphanResumeBanner
//
// Collapsed: single-line summary chip + Diagnostics + "Discard all".
// Expanded: per-orphan rows with Diagnostics + Discard. The
// expand-on-tap pattern keeps the wizard's vertical real estate
// untouched until the user opts into the recovery surface.
//
// Resume-from-orphan isn't implemented at the CLI yet (`mtplx forge
// build` doesn't take a `--resume` flag), so the per-row actions are
// honest: open the partial artifact in Finder to inspect, or drop it.
// When resume lands, an additional "Resume build" button goes here.

private struct OrphanResumeBanner: View {
    let runs: [ForgeOrphanRun]
    @Binding var expanded: Bool
    let onReveal: (ForgeOrphanRun) -> Void
    let onDiscard: (ForgeOrphanRun) -> Void
    let onDiscardAll: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            summaryRow
            if expanded {
                expandedList
                    .transition(.asymmetric(
                        insertion: .opacity.combined(with: .move(edge: .top)),
                        removal: .opacity
                    ))
            }
        }
        .background(Brand.warning.opacity(expanded ? 0.08 : 0.06))
        .overlay(
            Rectangle()
                .fill(Brand.warning.opacity(0.25))
                .frame(width: 2),
            alignment: .leading
        )
        .animation(.smooth(duration: 0.22), value: expanded)
    }

    private var summaryRow: some View {
        let count = runs.count
        let phase = runs.first?.lastWrittenPhase?.rawValue.capitalized ?? "in progress"
        return HStack(spacing: 10) {
            Button {
                expanded.toggle()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "arrow.uturn.backward.circle.fill")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Brand.warning)
                    Text(count == 1 ? "1 unfinished forge" : "\(count) unfinished forges")
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .foregroundStyle(Brand.typeHi)
                    Text("·")
                        .font(.system(size: 11))
                        .foregroundStyle(Brand.typeTertiary)
                    Text("last step \(phase.lowercased())")
                        .font(.system(size: 11))
                        .foregroundStyle(Brand.typeSecondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Image(systemName: "chevron.right")
                        .font(.system(size: 9, weight: .heavy))
                        .foregroundStyle(Brand.typeTertiary)
                        .rotationEffect(.degrees(expanded ? 90 : 0))
                        .animation(.smooth(duration: 0.18), value: expanded)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(expanded ? "Hide the unfinished forges" : "Show the unfinished forges with per-build actions")

            Spacer(minLength: 8)

            if let first = runs.first {
                Button {
                    onReveal(first)
                } label: {
                    Text("Diagnostics")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Brand.typeBody)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Open diagnostics for the latest unfinished build")
            }

            Button {
                onDiscardAll()
            } label: {
                Text(count == 1 ? "Discard" : "Discard all")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Brand.danger)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Remove every leftover file from interrupted builds")
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 7)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var expandedList: some View {
        VStack(spacing: 6) {
            ForEach(runs) { run in
                OrphanRunRow(
                    run: run,
                    onReveal: { onReveal(run) },
                    onDiscard: { onDiscard(run) }
                )
            }
            HStack(spacing: 8) {
                Text("Resume from a partial build isn't supported yet — Forge starts each build from scratch.")
                    .font(.system(size: 10.5))
                    .foregroundStyle(Brand.typeTertiary)
                Spacer(minLength: 0)
            }
            .padding(.top, 2)
        }
        .padding(.horizontal, 22)
        .padding(.top, 4)
        .padding(.bottom, 10)
    }
}

private struct OrphanRunRow: View {
    let run: ForgeOrphanRun
    let onReveal: () -> Void
    let onDiscard: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "doc.fill")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(Brand.typeTertiary)
                .frame(width: 14)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(run.runID)
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Brand.typeHi)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let phase = run.lastWrittenPhase {
                        Text("· \(phase.rawValue)")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Brand.typeSecondary)
                    }
                }
                Text(relativeTime(run.lastModified))
                    .font(.system(size: 10))
                    .foregroundStyle(Brand.typeTertiary)
            }
            Spacer(minLength: 8)
            Button(action: onReveal) {
                HStack(spacing: 4) {
                    Image(systemName: "folder")
                        .font(.system(size: 9, weight: .semibold))
                    Text("Diagnostics")
                        .font(.system(size: 10.5, weight: .semibold))
                }
                .foregroundStyle(Brand.typeBody)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(
                    Capsule(style: .continuous)
                        .strokeBorder(Brand.separatorStrong, lineWidth: 0.5)
                )
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .help("Open diagnostics for this partial build")
            .accessibilityLabel("Open orphan diagnostics")
            Button(action: onDiscard) {
                Image(systemName: "trash")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Brand.danger)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Delete this partial build")
            .accessibilityLabel("Delete orphan build")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(Brand.wash.opacity(0.025))
                .overlay(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    private func relativeTime(_ date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}
