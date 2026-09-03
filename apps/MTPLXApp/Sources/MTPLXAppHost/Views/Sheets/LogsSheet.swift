import SwiftUI
import MTPLXAppCore

// MARK: - LogsSheet
//
// V1 doesn't dedicate a tab to logs. Cmd-Shift-L (or the Logs link in
// Settings) presents this sheet. Same content as the old `LogsTab`,
// repainted onto the piano-black surface and given a Close button.

struct LogsSheet: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @Environment(\.dismiss) private var dismiss

    @State private var filter: LogFilter = .all
    @State private var search: String = ""
    @State private var autoScroll: Bool = true

    enum LogFilter: String, CaseIterable, Identifiable {
        case all, stdout, stderr, system
        var id: String { rawValue }
        var label: String {
            switch self {
            case .all: return tr("All")
            case .stdout: return tr("Stdout")
            case .stderr: return tr("Stderr")
            case .system: return tr("System")
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider().overlay(Brand.separator)
            controlsBar
            Divider().overlay(Brand.separator)
            content
        }
        .frame(minWidth: 720, minHeight: 540)
        .background(Brand.bgOuter.ignoresSafeArea())
        .appliesAppearance()
        .tint(Brand.accent)
        .task { await backend.refreshLogs() }
    }

    // MARK: - Sections

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "doc.text")
                .font(.title3)
                .foregroundStyle(Brand.accent)
            Text(tr("LOGS"))
                .font(.system(size: 14, weight: .heavy, design: .monospaced))
                .tracking(4)
                .foregroundStyle(Brand.textHighlight)
            Spacer()
            Button(tr("Close")) { dismiss() }
                .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Brand.bgInner)
    }

    @ViewBuilder
    private var controlsBar: some View {
        HStack(spacing: 12) {
            Picker(tr("Stream"), selection: $filter) {
                ForEach(LogFilter.allCases) { f in
                    Text(f.label).tag(f)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 320)
            TextField(tr("Search"), text: $search)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 240)
            Spacer()
            Toggle(tr("Auto-scroll"), isOn: $autoScroll)
                .toggleStyle(.switch)
                .controlSize(.mini)
            Button {
                Task { await backend.refreshLogs() }
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.borderless)
            .help(tr("Refresh log buffer"))
            .accessibilityLabel(tr("Refresh logs"))
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Brand.bgInner.opacity(0.6))
    }

    @ViewBuilder
    private var content: some View {
        let filtered = filteredLogs()
        if filtered.isEmpty {
            VStack(spacing: 6) {
                Text(emptyMessage)
                    .font(.callout)
                    .foregroundStyle(Brand.textHighlight.opacity(0.7))
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(filtered) { entry in
                            row(entry).id(entry.id)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                }
                .onChange(of: filtered.last?.id) { _, newId in
                    guard autoScroll, let newId else { return }
                    withAnimation(.linear(duration: 0.1)) {
                        proxy.scrollTo(newId, anchor: .bottom)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func row(_ entry: LogEntry) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(timestamp(entry.date))
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Brand.textHighlight.opacity(0.5))
            PillBadge(text: entry.stream.rawValue, tint: tint(for: entry.stream))
            Text(entry.message)
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(Brand.textHighlight)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 1)
    }

    // MARK: - Helpers

    private func filteredLogs() -> [LogEntry] {
        let needle = search.trimmingCharacters(in: .whitespaces).lowercased()
        return backend.logs.filter { entry in
            guard filter == .all || entry.stream.rawValue == filter.rawValue else { return false }
            guard !needle.isEmpty else { return true }
            return entry.message.lowercased().contains(needle)
        }
    }

    private var emptyMessage: String {
        backend.logs.isEmpty
            ? "No output yet."
            : tr("No log lines match the current filter.")
    }

    private func tint(for stream: LogEntry.Stream) -> Color {
        switch stream {
        case .stdout: return Brand.textHighlight.opacity(0.7)
        case .stderr: return Brand.danger
        case .system: return Brand.accent
        }
    }

    private func timestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter.string(from: date)
    }
}
