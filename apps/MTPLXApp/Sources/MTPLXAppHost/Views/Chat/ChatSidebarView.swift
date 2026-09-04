import SwiftUI
import MTPLXAppCore

// MARK: - ChatSidebarView
//
// Left rail listing past conversations. "+ New Chat" button at the
// top, searchable list of conversations sorted by `updatedAt` desc,
// right-click → Delete on each row. Collapses to a 0pt rail when
// hidden so the chat surface gets the full window width.

struct ChatSidebarView: View {
    @ObservedObject var viewModel: ChatViewModel
    @Binding var collapsed: Bool
    /// Launch-time chat store recovery (kept-aside store or memory-only
    /// session), injected at the app root; shown here until dismissed.
    @EnvironmentObject private var chatStoreRecovery: ChatStoreRecoveryState
    @State private var searchQuery: String = ""
    @State private var confirmingDelete: ChatConversation?

    var body: some View {
        Group {
            if collapsed {
                EmptyView()
            } else {
                content
                    .frame(width: 240)
                    .transition(.move(edge: .leading).combined(with: .opacity))
            }
        }
    }

    private var content: some View {
        VStack(spacing: 0) {
            header
            if let notice = chatStoreRecovery.notice {
                ChatStoreRecoveryBanner(notice: notice) {
                    chatStoreRecovery.notice = nil
                }
            }
            searchField
            Divider()
                .frame(height: 0.5)
                .overlay(Brand.separator)
            list
        }
        .background(
            Brand.bgInner
                .overlay(
                    Rectangle()
                        .fill(Brand.separator)
                        .frame(width: 0.5),
                    alignment: .trailing
                )
        )
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text(tr("Chats"))
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1.5)
                .foregroundStyle(Brand.typeTertiary)
            Spacer()
            Button {
                _ = viewModel.createNewConversation()
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Brand.typeSecondary)
                    .frame(width: 24, height: 24)
                    .background(
                        Circle()
                            .fill(Brand.wash.opacity(0.04))
                            .overlay(Circle().stroke(Brand.separator, lineWidth: 0.5))
                    )
            }
            .buttonStyle(.plain)
            .help(tr("New chat (⌘N)"))
            .accessibilityLabel(tr("New chat"))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 12)
    }

    private var searchField: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 11))
                .foregroundStyle(Brand.typeTertiary)
            TextField(tr("Search"), text: $searchQuery)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .foregroundStyle(Brand.typeHi)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Brand.wash.opacity(0.04))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
        .padding(.horizontal, 10)
        .padding(.bottom, 10)
    }

    private var list: some View {
        ScrollView {
            LazyVStack(spacing: 2) {
                ForEach(filteredConversations, id: \.id) { conversation in
                    row(for: conversation)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
        }
    }

    private func row(for conversation: ChatConversation) -> some View {
        let isSelected = viewModel.current?.id == conversation.id
        return Button {
            viewModel.select(conversation)
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(conversation.title)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(isSelected ? Brand.typeHi : Brand.typeSecondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Text(Self.relativeDate(conversation.updatedAt))
                    .font(.system(size: 10))
                    .foregroundStyle(Brand.typeTertiary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 8)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(isSelected ? Brand.wash.opacity(0.06) : Color.clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button(role: .destructive) {
                confirmingDelete = conversation
            } label: {
                Label(tr("Delete"), systemImage: "trash")
            }
        }
        .alert(
            tr("Delete this chat?"),
            isPresented: Binding(
                get: { confirmingDelete?.id == conversation.id },
                set: { newValue in
                    if !newValue { confirmingDelete = nil }
                }
            )
        ) {
            Button(tr("Cancel"), role: .cancel) {}
            Button(tr("Delete"), role: .destructive) {
                Task { await viewModel.delete(conversation) }
            }
        } message: {
            Text(tr("This will remove \"%@\" and its messages.", conversation.title))
        }
    }

    private var filteredConversations: [ChatConversation] {
        let trimmed = searchQuery.trimmingCharacters(in: .whitespaces).lowercased()
        guard !trimmed.isEmpty else { return viewModel.conversations }
        return viewModel.conversations.filter { convo in
            convo.title.lowercased().contains(trimmed)
        }
    }

    @MainActor
    private static let relativeDateFormatter: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        return formatter
    }()

    @MainActor
    private static func relativeDate(_ date: Date) -> String {
        relativeDateFormatter.localizedString(for: date, relativeTo: Date())
    }
}

// MARK: - ChatStoreRecoveryBanner

/// Persistent, dismissable card at the top of the rail: the chat store
/// on disk would not open at launch. Either the old files were kept
/// beside the new store (Reveal in Finder shows them) or, when no store
/// could be created at all, chats are living in memory for this session.
private struct ChatStoreRecoveryBanner: View {
    let notice: ChatStoreRecoveryNotice
    let onDismiss: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Color.mtplxWarning)
                Text(message)
                    .font(.system(size: 11))
                    .foregroundStyle(Brand.typeHi)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 8) {
                if case .recoveredFromUnreadableStore(let preservedAt) = notice {
                    Button(tr("Reveal in Finder")) {
                        NSWorkspace.shared.activateFileViewerSelecting([preservedAt])
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                Spacer(minLength: 0)
                Button(tr("Dismiss"), action: onDismiss)
                    .buttonStyle(.borderless)
                    .controlSize(.small)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.mtplxWarning.opacity(0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(Color.mtplxWarning.opacity(0.28), lineWidth: 0.5)
                )
        )
        .padding(.horizontal, 10)
        .padding(.bottom, 10)
    }

    private var message: String {
        switch notice {
        case .recoveredFromUnreadableStore(let preservedAt):
            return tr(
                "Your earlier chats could not be opened. They were kept as %@ and a new chat history was started.",
                preservedAt.lastPathComponent
            )
        case .inMemoryOnly:
            return tr("Chats cannot be saved right now, so this session's chats will not be kept.")
        }
    }
}
