import SwiftUI
import PDFKit
import MTPLXAppCore

// MARK: - AttachmentCard
//
// Composer-strip chip showing a pending or sent file attachment.
// Inspired by Aphanes V2's `AttachmentCard` but redrawn against Brand
// tokens — compact 136pt × 56pt card with type badge on the left
// (PDFKit thumbnail for .pdf, monochrome glyph for others), filename
// + size to the right, and an × remove affordance on hover when an
// `onRemove` handler is provided. A red dot replaces the badge when
// `errorMessage` is non-nil (failed extraction).

struct AttachmentCard: View {
    let filename: String
    let fileExtension: String
    let sizeBytes: Int
    var fileURL: URL? = nil
    /// Encoded image bytes for vision attachments; renders the actual
    /// picture as the badge thumbnail.
    var imageData: Data? = nil
    var errorMessage: String? = nil
    var onTap: (() -> Void)? = nil
    var onRemove: (() -> Void)? = nil

    @State private var hovering = false

    var body: some View {
        ZStack(alignment: .topTrailing) {
            HStack(spacing: 8) {
                badge
                VStack(alignment: .leading, spacing: 2) {
                    Text(filename)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Brand.typeHi)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    HStack(spacing: 6) {
                        Text(fileExtension.uppercased())
                            .font(.system(size: 9, weight: .heavy, design: .monospaced))
                            .tracking(1)
                            .foregroundStyle(Brand.typeTertiary)
                        if sizeBytes > 0 {
                            Text(Self.formatBytes(sizeBytes))
                                .font(.system(size: 10))
                                .foregroundStyle(Brand.typeTertiary)
                        }
                        if let errorMessage {
                            Text(errorMessage)
                                .font(.system(size: 10))
                                .foregroundStyle(Brand.warning)
                                .lineLimit(1)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(width: 200, height: 56, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Brand.wash.opacity(0.05))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .stroke(Brand.separator, lineWidth: 0.5)
                    )
            )
            .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .onTapGesture {
                onTap?()
            }
            .onHover { hovering = $0 }

            if let onRemove, hovering {
                Button {
                    onRemove()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 14))
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(Brand.onInk, Brand.ink.opacity(0.5))
                        .padding(4)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .offset(x: 6, y: -6)
                .transition(.opacity.combined(with: .scale(scale: 0.6)))
                .accessibilityLabel("Remove attachment")
            }
        }
        .animation(.smooth(duration: 0.16), value: hovering)
    }

    @ViewBuilder
    private var badge: some View {
        if let imageData, let thumb = NSImage(data: imageData) {
            Image(nsImage: thumb)
                .resizable()
                .scaledToFill()
                .frame(width: 32, height: 32)
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        } else if errorMessage != nil {
            ZStack {
                Circle()
                    .fill(Brand.wash.opacity(0.06))
                Circle()
                    .fill(Brand.warning)
                    .frame(width: 8, height: 8)
            }
            .frame(width: 32, height: 32)
        } else if fileExtension.lowercased() == "pdf",
            let fileURL,
            let thumb = Self.pdfThumbnail(for: fileURL)
        {
            Image(nsImage: thumb)
                .resizable()
                .scaledToFill()
                .frame(width: 32, height: 32)
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        } else {
            ZStack {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Self.badgeColor(for: fileExtension))
                Image(systemName: Self.iconName(for: fileExtension))
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(Brand.ink.opacity(0.7))
            }
            .frame(width: 32, height: 32)
        }
    }

    private static func badgeColor(for ext: String) -> Color {
        switch ext.lowercased() {
        case "pdf": return Color(red: 0.95, green: 0.45, blue: 0.40)
        case "docx": return Color(red: 0.40, green: 0.65, blue: 0.95)
        case "md": return Color(red: 0.55, green: 0.85, blue: 0.55)
        case "txt": return Color(red: 0.80, green: 0.80, blue: 0.80)
        default: return Color(red: 0.70, green: 0.70, blue: 0.70)
        }
    }

    private static func iconName(for ext: String) -> String {
        switch ext.lowercased() {
        case "pdf": return "doc.richtext"
        case "docx": return "doc.text"
        case "md": return "text.alignleft"
        case "txt": return "doc.plaintext"
        default: return "doc"
        }
    }

    static func pdfThumbnail(for url: URL) -> NSImage? {
        guard let document = PDFDocument(url: url),
            let page = document.page(at: 0)
        else { return nil }
        let pageRect = page.bounds(for: .mediaBox)
        let scale = 64.0 / max(pageRect.width, pageRect.height)
        let size = NSSize(width: pageRect.width * scale, height: pageRect.height * scale)
        let image = NSImage(size: size)
        image.lockFocus()
        if let context = NSGraphicsContext.current?.cgContext {
            context.saveGState()
            context.scaleBy(x: scale, y: scale)
            page.draw(with: .mediaBox, to: context)
            context.restoreGState()
        }
        image.unlockFocus()
        return image
    }

    static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.includesUnit = true
        return formatter.string(fromByteCount: Int64(bytes))
    }
}
