import SwiftUI
import MTPLXAppCore

// MARK: - ReasoningStreamView
//
// Math-aware panel for live AIME reasoning. The product path renders a bounded
// bottom-pinned LaTeX tail with no live scroll work; diagnostics can still
// force full scrollback modes to prove cost.

struct ReasoningStreamView: View {
    @ObservedObject var document: StreamingDocumentStore
    let active: Bool
    @State private var lastScrollAt: Date = .distantPast

    var body: some View {
        let mode = AIMEDiagnostics.renderMode
        let blocks = visibleBlocks(for: mode)
        let renderStyle = renderStyle(for: mode)

        if mode == .hidden {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .onChange(of: document.revision) { _, _ in
                    recordRenderPublication(mode: mode, visibleBlockCount: 0, bottomID: nil, scrolled: false)
                }
        } else if mode == .tailLatex || mode == .plainTail {
            tailBody(mode: mode, blocks: blocks, renderStyle: renderStyle)
        } else if mode == .fullScroll {
            autoScrollingBody(mode: mode, blocks: blocks, renderStyle: renderStyle)
        } else {
            passiveBody(mode: mode, blocks: blocks, renderStyle: renderStyle)
        }
    }

    private func tailBody(
        mode: AIMERenderMode,
        blocks: [StreamingDocumentBlock],
        renderStyle: MathReasoningRenderStyle
    ) -> some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottomLeading) {
                if blocks.isEmpty {
                    Text(tr("Awaiting reasoning..."))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Brand.typeTertiary)
                        .lineSpacing(2)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                } else {
                    MathReasoningRender(blocks: blocks, style: renderStyle)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .frame(width: geometry.size.width, alignment: .leading)
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height, alignment: .bottomLeading)
            .clipped()
            .mask {
                LinearGradient(
                    stops: [
                        .init(color: .clear, location: 0.0),
                        .init(color: .white, location: 0.06),
                        .init(color: .white, location: 1.0)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            }
        }
        .onChange(of: document.revision) { _, _ in
            recordRenderPublication(
                mode: mode,
                visibleBlockCount: blocks.count,
                bottomID: blocks.last?.id,
                scrolled: false
            )
        }
    }

    private func autoScrollingBody(
        mode: AIMERenderMode,
        blocks: [StreamingDocumentBlock],
        renderStyle: MathReasoningRenderStyle
    ) -> some View {
        ScrollViewReader { proxy in
            scrollBody(blocks: blocks, renderStyle: renderStyle)
            .onChange(of: document.revision) { _, _ in
                let bottomID = blocks.last?.id
                recordRenderPublication(mode: mode, visibleBlockCount: blocks.count, bottomID: bottomID, scrolled: false)
                guard active, let bottomID else { return }
                let now = Date()
                guard now.timeIntervalSince(lastScrollAt) >= 0.16 else { return }
                lastScrollAt = now
                var transaction = Transaction()
                transaction.animation = nil
                withTransaction(transaction) {
                    proxy.scrollTo(bottomID, anchor: .bottom)
                }
                AIMEDiagnostics.signpost(.scroll)
                recordRenderPublication(mode: mode, visibleBlockCount: blocks.count, bottomID: bottomID, scrolled: true)
            }
        }
    }

    private func passiveBody(
        mode: AIMERenderMode,
        blocks: [StreamingDocumentBlock],
        renderStyle: MathReasoningRenderStyle
    ) -> some View {
        scrollBody(blocks: blocks, renderStyle: renderStyle)
            .onChange(of: document.revision) { _, _ in
                recordRenderPublication(
                    mode: mode,
                    visibleBlockCount: blocks.count,
                    bottomID: blocks.last?.id,
                    scrolled: false
                )
            }
    }

    private func scrollBody(
        blocks: [StreamingDocumentBlock],
        renderStyle: MathReasoningRenderStyle
    ) -> some View {
        ScrollView(.vertical, showsIndicators: false) {
            Group {
                if blocks.isEmpty {
                    Text(tr("Awaiting reasoning..."))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Brand.typeTertiary)
                        .lineSpacing(2)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    MathReasoningRender(blocks: blocks, style: renderStyle)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .mask {
            LinearGradient(
                stops: [
                    .init(color: .clear, location: 0.0),
                    .init(color: .white, location: 0.06),
                    .init(color: .white, location: 0.94),
                    .init(color: .clear, location: 1.0)
                ],
                startPoint: .top,
                endPoint: .bottom
            )
        }
    }

    private func visibleBlocks(for mode: AIMERenderMode) -> [StreamingDocumentBlock] {
        switch mode {
        case .tailLatex, .plainTail:
            return Array(document.blocks.suffix(AIMEDiagnostics.tailBlockLimit))
        case .hidden:
            return []
        case .fullScroll, .noAutoscroll:
            return document.blocks
        }
    }

    private func renderStyle(for mode: AIMERenderMode) -> MathReasoningRenderStyle {
        mode == .plainTail ? .plain : .latex
    }

    private func recordRenderPublication(
        mode: AIMERenderMode,
        visibleBlockCount: Int,
        bottomID: Int?,
        scrolled: Bool
    ) {
        guard AIMEDiagnostics.isEnabled else { return }
        guard AIMEDiagnostics.shouldRecordCadenced(
            "render_publication",
            intervalS: 1,
            tokenCount: document.wordCount,
            identity: mode.rawValue
        ) else { return }
        AIMEDiagnostics.signpost(.renderPublication)
        AIMEDiagnostics.record(
            "render_publication",
            fields: AIMEDiagnostics.fields(
                ("mode", .string(mode.rawValue)),
                ("active", .bool(active)),
                ("revision", .int(document.revision)),
                ("total_blocks", .int(document.blocks.count)),
                ("visible_blocks", .int(visibleBlockCount)),
                ("bottom_id", bottomID.map(AIMEDiagnosticValue.int)),
                ("scrolled", .bool(scrolled))
            )
        )
    }
}
