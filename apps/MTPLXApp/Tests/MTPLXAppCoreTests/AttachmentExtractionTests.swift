import CoreGraphics
import Foundation
import ImageIO
import SwiftData
import UniformTypeIdentifiers
import XCTest

@testable import MTPLXAppCore

// MARK: - AttachmentExtractionTests
//
// Attaching a file used to run the whole extraction (PDFKit page walk,
// a docx unzip waiting on a child process, image decoding) inline on
// the main actor, freezing the app for the duration with no feedback.
// Extraction now runs detached; the card shows an extracting state and
// settles to ready (with a truncation note when the caps cut something)
// or to a visible failure that keeps the file on the strip. These pin
// where the work runs, how the card follows it, and the caps.

final class AttachmentExtractionTests: XCTestCase {
    override func setUp() {
        super.setUp()
        L10n.activate(.english)
    }

    override func tearDown() {
        L10n.activate(.english)
        super.tearDown()
    }

    private static let unreachableClient = MTPLXChatClient(
        apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:1")!)
    )

    @MainActor
    private func makeViewModel(
        extractor: @escaping ChatViewModel.AttachmentExtractor
    ) throws -> ChatViewModel {
        ChatViewModel(
            container: try ChatStore.makeInMemoryContainer(),
            chatClientProvider: { Self.unreachableClient },
            modelName: { "mtplx-test-model" },
            attachmentExtractor: extractor
        )
    }

    private func scratchDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-attachment-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Thread-safe box for what the injected extractor observed.
    private final class Observed: @unchecked Sendable {
        private let lock = NSLock()
        private var _mainThreadCalls = 0
        private var _calls = 0
        func record(onMain: Bool) {
            lock.lock(); defer { lock.unlock() }
            _calls += 1
            if onMain { _mainThreadCalls += 1 }
        }
        var calls: Int { lock.lock(); defer { lock.unlock() }; return _calls }
        var mainThreadCalls: Int { lock.lock(); defer { lock.unlock() }; return _mainThreadCalls }
    }

    // MARK: Off the main actor

    @MainActor
    func testExtractionRunsOffTheMainThreadAndAppliesItsResult() async throws {
        let observed = Observed()
        let viewModel = try makeViewModel { url in
            observed.record(onMain: Thread.isMainThread)
            return ChatViewModel.ExtractedAttachment(
                filename: url.lastPathComponent,
                mimeType: "text/markdown",
                sizeBytes: 42,
                extractedText: "extracted body of \(url.lastPathComponent)"
            )
        }

        await viewModel.attach([URL(fileURLWithPath: "/tmp/notes.md"), URL(fileURLWithPath: "/tmp/more.md")])

        XCTAssertEqual(observed.calls, 2)
        XCTAssertEqual(observed.mainThreadCalls, 0, "extraction must not run on the main thread")
        XCTAssertEqual(viewModel.pendingAttachments.map(\.filename), ["notes.md", "more.md"])
        XCTAssertEqual(viewModel.pendingAttachments.map(\.extractedText), ["extracted body of notes.md", "extracted body of more.md"])
        XCTAssertEqual(viewModel.pendingAttachments.map(\.sizeBytes), [42, 42])
        for attachment in viewModel.pendingAttachments {
            XCTAssertEqual(viewModel.extractionState(for: attachment), .ready(truncation: nil))
        }
        XCTAssertFalse(viewModel.isExtractingAttachments)
        XCTAssertTrue(viewModel.hasSendablePendingAttachments)
        XCTAssertNil(viewModel.lastError)
    }

    /// End to end with the production extractor: a text-heavy 500-page
    /// PDF (about 0.8 s of PDFKit work) must not stall the main actor.
    /// A ticker on the main actor measures its longest gap; the probe is
    /// calibrated first with a deliberate synchronous block so a broken
    /// ticker cannot pass the test. Before the fix the gap equalled the
    /// whole extraction (0.82 s); after it, a few milliseconds.
    @MainActor
    func testRealPDFExtractionDoesNotBlockTheMainActor() async throws {
        let directory = try scratchDirectory()
        let pdfURL = directory.appendingPathComponent("heavy.pdf")
        try Self.writePDF(pages: 500, linesPerPage: 45, to: pdfURL)
        let viewModel = try makeViewModel(extractor: ChatViewModel.extractAttachment)

        final class Gap: @unchecked Sendable {
            private let lock = NSLock()
            private var _max: TimeInterval = 0
            var max: TimeInterval {
                get { lock.lock(); defer { lock.unlock() }; return _max }
                set { lock.lock(); defer { lock.unlock() }; _max = newValue }
            }
        }
        let gap = Gap()
        let ticker = Task { @MainActor in
            var last = Date()
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(5))
                let now = Date()
                gap.max = Swift.max(gap.max, now.timeIntervalSince(last))
                last = now
            }
        }
        try await Task.sleep(for: .milliseconds(50))

        let calibrationOnMain = Self.blockMainActorSynchronously(seconds: 0.3)
        try await Task.sleep(for: .milliseconds(20))
        let calibrationGap = gap.max
        XCTAssertTrue(calibrationOnMain)
        XCTAssertGreaterThan(calibrationGap, 0.25, "the ticker must register a blocked main actor")
        gap.max = 0

        let started = Date()
        await viewModel.attach([pdfURL])
        let elapsed = Date().timeIntervalSince(started)
        try await Task.sleep(for: .milliseconds(20))
        ticker.cancel()

        XCTAssertEqual(viewModel.pendingAttachments.count, 1)
        XCTAssertTrue(viewModel.hasSendablePendingAttachments)
        XCTAssertGreaterThan(elapsed, 0.05, "the fixture should be heavy enough to measure")
        XCTAssertLessThan(
            gap.max, Swift.max(0.1, elapsed / 4),
            "the main actor was blocked for most of the extraction (\(gap.max)s of \(elapsed)s)"
        )
    }

    /// Synchronous on purpose: a non-async main-actor function may block
    /// the actor, which is exactly what the calibration needs.
    @MainActor
    private static func blockMainActorSynchronously(seconds: TimeInterval) -> Bool {
        Thread.sleep(forTimeInterval: seconds)
        return Thread.isMainThread
    }

    // MARK: Extracting, then ready

    @MainActor
    func testCardShowsExtractingUntilTheWorkFinishesAndSendWaits() async throws {
        let directory = try scratchDirectory()
        let releaseURL = directory.appendingPathComponent("release")
        let viewModel = try makeViewModel { url in
            // Hold until the test releases us, like a slow PDF.
            let deadline = Date().addingTimeInterval(10)
            while !FileManager.default.fileExists(atPath: releaseURL.path), Date() < deadline {
                Thread.sleep(forTimeInterval: 0.01)
            }
            return ChatViewModel.ExtractedAttachment(
                filename: url.lastPathComponent, mimeType: "application/pdf",
                sizeBytes: 1_000, extractedText: "the paper"
            )
        }

        let attaching = Task { await viewModel.attach([URL(fileURLWithPath: "/tmp/paper.pdf")]) }
        try await pollUntil("the card appears in the extracting state") {
            viewModel.pendingAttachments.count == 1
                && viewModel.attachmentStates.values.first == .extracting
        }
        XCTAssertTrue(viewModel.isExtractingAttachments)
        XCTAssertFalse(viewModel.hasSendablePendingAttachments, "nothing to send yet")
        XCTAssertEqual(viewModel.pendingAttachments.first?.filename, "paper.pdf")

        // The main actor is free while the file extracts: a send during
        // extraction is held so the file is not left behind.
        viewModel.send("about the paper")
        XCTAssertTrue(viewModel.conversations.isEmpty, "send waits for the strip to settle")
        XCTAssertTrue(viewModel.visibleMessages.isEmpty)
        XCTAssertFalse(viewModel.isStreaming)

        try Data("go".utf8).write(to: releaseURL)
        await attaching.value

        let attachment = try XCTUnwrap(viewModel.pendingAttachments.first)
        XCTAssertEqual(viewModel.extractionState(for: attachment), .ready(truncation: nil))
        XCTAssertEqual(attachment.extractedText, "the paper")
        XCTAssertFalse(viewModel.isExtractingAttachments)
        XCTAssertTrue(viewModel.hasSendablePendingAttachments)
    }

    @MainActor
    func testRemovingACardWhileItExtractsDropsTheResult() async throws {
        let directory = try scratchDirectory()
        let releaseURL = directory.appendingPathComponent("release")
        let viewModel = try makeViewModel { url in
            let deadline = Date().addingTimeInterval(10)
            while !FileManager.default.fileExists(atPath: releaseURL.path), Date() < deadline {
                Thread.sleep(forTimeInterval: 0.01)
            }
            return ChatViewModel.ExtractedAttachment(
                filename: url.lastPathComponent, mimeType: "text/plain", sizeBytes: 3, extractedText: "abc"
            )
        }

        let attaching = Task { await viewModel.attach([URL(fileURLWithPath: "/tmp/a.txt")]) }
        try await pollUntil("the card appears") { viewModel.pendingAttachments.count == 1 }
        let attachment = try XCTUnwrap(viewModel.pendingAttachments.first)
        viewModel.removeAttachment(attachment)
        XCTAssertTrue(viewModel.pendingAttachments.isEmpty)
        XCTAssertFalse(viewModel.isExtractingAttachments)

        try Data("go".utf8).write(to: releaseURL)
        await attaching.value
        XCTAssertTrue(viewModel.pendingAttachments.isEmpty, "a removed card does not come back")
        XCTAssertTrue(viewModel.attachmentStates.isEmpty)
    }

    // MARK: Failure stays visible

    @MainActor
    func testFailingExtractionReportsOnTheCardAndKeepsTheFile() async throws {
        let viewModel = try makeViewModel { url in
            throw FileExtractorError.unreadable(filename: url.lastPathComponent, reason: "the file is encrypted")
        }

        await viewModel.attach([URL(fileURLWithPath: "/tmp/locked.pdf")])

        let attachment = try XCTUnwrap(viewModel.pendingAttachments.first, "a failed file stays on the strip")
        XCTAssertEqual(attachment.filename, "locked.pdf")
        XCTAssertEqual(
            viewModel.extractionState(for: attachment),
            .failed(message: "Could not read locked.pdf: the file is encrypted")
        )
        XCTAssertFalse(viewModel.hasSendablePendingAttachments)
        XCTAssertNil(viewModel.lastError, "an attachment problem lives on its card, not on the reply error card")
        XCTAssertFalse(viewModel.isExtractingAttachments)
    }

    // MARK: The production extractor

    @MainActor
    func testDefaultExtractorReadsDocumentsAndImages() async throws {
        let directory = try scratchDirectory()
        let mdURL = directory.appendingPathComponent("notes.md")
        try "# Notes\n\nMTPLX_B07_MARKDOWN".write(to: mdURL, atomically: true, encoding: .utf8)
        let pngURL = directory.appendingPathComponent("shot.png")
        try Self.writePNG(width: 8, height: 8, to: pngURL)
        let brokenPNG = directory.appendingPathComponent("broken.png")
        try Data("not an image".utf8).write(to: brokenPNG)

        let viewModel = try makeViewModel(extractor: ChatViewModel.extractAttachment)
        await viewModel.attach([mdURL, pngURL, brokenPNG])

        XCTAssertEqual(viewModel.pendingAttachments.map(\.filename), ["notes.md", "shot.png", "broken.png"])
        let notes = viewModel.pendingAttachments[0]
        XCTAssertEqual(notes.extractedText, "# Notes\n\nMTPLX_B07_MARKDOWN")
        XCTAssertEqual(viewModel.extractionState(for: notes), .ready(truncation: nil))

        let shot = viewModel.pendingAttachments[1]
        XCTAssertNotNil(shot.imageData)
        XCTAssertTrue(shot.isImage)
        XCTAssertEqual(viewModel.extractionState(for: shot), .ready(truncation: nil))

        let broken = viewModel.pendingAttachments[2]
        XCTAssertNil(broken.imageData)
        guard case .failed(let message) = viewModel.extractionState(for: broken) else {
            return XCTFail("a corrupt image must fail visibly, got \(String(describing: viewModel.extractionState(for: broken)))")
        }
        XCTAssertTrue(message.contains("broken.png"), message)
    }

    // MARK: Caps

    func testCharacterCapTruncatesWithANoteForTheUserAndAMarkerForTheModel() throws {
        let directory = try scratchDirectory()
        let bigURL = directory.appendingPathComponent("big.txt")
        let total = FileExtractor.maxCharacters + 5_000
        try String(repeating: "x", count: total).write(to: bigURL, atomically: true, encoding: .utf8)

        let extracted = try FileExtractor.extract(from: bigURL)
        let truncation = try XCTUnwrap(extracted.truncation)
        XCTAssertNil(truncation.pages)
        XCTAssertEqual(truncation.characters, .init(included: FileExtractor.maxCharacters, total: total))
        XCTAssertTrue(extracted.combinedText.hasPrefix(String(repeating: "x", count: 100)))
        XCTAssertTrue(
            extracted.combinedText.hasSuffix(
                "[Attachment truncated: first \(FileExtractor.maxCharacters) of \(total) characters included]"
            ),
            "the model is told the document continues"
        )
        XCTAssertLessThan(extracted.combinedText.count, total)
        XCTAssertEqual(
            truncation.summary,
            "Truncated to 200,000 of 205,000 characters"
        )
    }

    func testSmallFilesAreNotTruncated() throws {
        let directory = try scratchDirectory()
        let url = directory.appendingPathComponent("small.txt")
        try "just a few words".write(to: url, atomically: true, encoding: .utf8)
        let extracted = try FileExtractor.extract(from: url)
        XCTAssertNil(extracted.truncation)
        XCTAssertEqual(extracted.combinedText, "just a few words")
    }

    func testPDFPageCapReadsOnlyTheFirstPagesAndSaysSo() throws {
        let directory = try scratchDirectory()
        let pdfURL = directory.appendingPathComponent("book.pdf")
        let pageCount = FileExtractor.maxPDFPages + 3
        try Self.writePDF(pages: pageCount, to: pdfURL)

        let extracted = try FileExtractor.extract(from: pdfURL)
        XCTAssertEqual(extracted.pageCount, pageCount)
        let truncation = try XCTUnwrap(extracted.truncation)
        XCTAssertEqual(truncation.pages, .init(included: FileExtractor.maxPDFPages, total: pageCount))
        XCTAssertTrue(extracted.combinedText.contains("PAGE 1 "))
        XCTAssertTrue(extracted.combinedText.contains("PAGE \(FileExtractor.maxPDFPages) "))
        XCTAssertFalse(extracted.combinedText.contains("PAGE \(FileExtractor.maxPDFPages + 1) "))
        XCTAssertTrue(
            extracted.combinedText.hasSuffix(
                "[Attachment truncated: first \(FileExtractor.maxPDFPages) of \(pageCount) pages included]"
            )
        )
        XCTAssertEqual(truncation.summary, "Truncated to \(FileExtractor.maxPDFPages) of \(pageCount) pages")
    }

    func testTruncationSummaryIsLocalised() {
        L10n.activate(.german)
        let truncation = ExtractionTruncation(
            pages: .init(included: 500, total: 812),
            characters: .init(included: 200_000, total: 1_234_567)
        )
        XCTAssertEqual(
            truncation.summary,
            "Gekürzt auf 500 von 812 Seiten · Gekürzt auf 200.000 von 1.234.567 Zeichen"
        )
    }

    // MARK: Fixtures

    private static func writePNG(width: Int, height: Int, to url: URL) throws {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw NSError(domain: "AttachmentExtractionTests", code: 1) }
        context.setFillColor(CGColor(red: 0.2, green: 0.4, blue: 0.8, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let image = context.makeImage(),
            let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw NSError(domain: "AttachmentExtractionTests", code: 2) }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw NSError(domain: "AttachmentExtractionTests", code: 3)
        }
    }

    private static func writePDF(pages: Int, linesPerPage: Int = 1, to url: URL) throws {
        let data = NSMutableData()
        var mediaBox = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let consumer = CGDataConsumer(data: data),
            let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil)
        else { throw NSError(domain: "AttachmentExtractionTests", code: 4) }
        let font = CTFontCreateWithName("Helvetica" as CFString, 10, nil)
        for page in 1...pages {
            context.beginPDFPage(nil)
            for row in 0..<linesPerPage {
                let text = row == 0
                    ? "PAGE \(page) of the book"
                    : "Page \(page) line \(row): lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor"
                let line = CTLineCreateWithAttributedString(
                    NSAttributedString(string: text, attributes: [.font: font])
                )
                context.textPosition = CGPoint(x: 36, y: 760 - CGFloat(row) * 16)
                CTLineDraw(line, context)
            }
            context.endPDFPage()
        }
        context.closePDF()
        try data.write(to: url)
    }
}
