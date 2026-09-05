import AppKit
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest

@testable import MTPLXAppCore

// MARK: - PasteAttachmentTests
//
// ⌘V in the composer used to do nothing for a screenshot or a Finder
// copy: NSTextView (plain text, no graphics) had no readable type and
// swallowed the paste. Now the pasteboard is classified — files, then
// an image, then plain text as before — and an image becomes a pending
// card through the same cap, validity check and downscale a dropped
// image file gets. A model that cannot see refuses the image on the
// card instead of quietly dropping it. These pin the precedence, the
// PNG normalization, the card lifecycle and the vision gate.

final class PasteAttachmentTests: XCTestCase {
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
        extractor: @escaping ChatViewModel.AttachmentExtractor = ChatViewModel.extractAttachment
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
            .appendingPathComponent("mtplx-paste-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    // MARK: Classifier precedence

    func testFilesCopiedInFinderWinOverAnImageOnTheSamePasteboard() throws {
        let directory = try scratchDirectory()
        let notes = directory.appendingPathComponent("notes.md")
        try "# notes".write(to: notes, atomically: true, encoding: .utf8)
        let png = try Self.pngData(width: 4, height: 4)

        let action = ComposerPasteClassifier.classify(.init(urls: [notes], imageData: png))

        XCTAssertEqual(action, .files([notes]))
    }

    func testNonFileURLsAreIgnoredSoABrowserImageCopyPastesTheImage() throws {
        // Browsers put the image's address on the pasteboard as a URL and
        // as text next to the image itself.
        let png = try Self.pngData(width: 4, height: 4)
        let address = URL(string: "https://example.com/picture.png")!

        XCTAssertEqual(
            ComposerPasteClassifier.classify(.init(urls: [address], imageData: png)),
            .image(png)
        )
        XCTAssertEqual(
            ComposerPasteClassifier.classify(.init(urls: [address], imageData: nil)),
            .passthrough,
            "a bare web address is text"
        )
    }

    func testURLsToMissingFilesAreIgnored() throws {
        let directory = try scratchDirectory()
        let present = directory.appendingPathComponent("present.txt")
        try "here".write(to: present, atomically: true, encoding: .utf8)
        let missing = directory.appendingPathComponent("gone.txt")

        XCTAssertEqual(
            ComposerPasteClassifier.classify(.init(urls: [missing, present])),
            .files([present])
        )
        XCTAssertEqual(
            ComposerPasteClassifier.classify(.init(urls: [missing])),
            .passthrough
        )
    }

    func testAPasteboardWithoutFilesOrAnImagePassesThroughToAppKit() {
        XCTAssertEqual(ComposerPasteClassifier.classify(.init()), .passthrough)
    }

    func testPNGBytesAreKeptUntouched() throws {
        // A screenshot stays lossless and byte-identical.
        let png = try Self.pngData(width: 6, height: 3)
        XCTAssertEqual(ComposerPasteClassifier.classify(.init(imageData: png)), .image(png))
    }

    func testTIFFNormalizesToPNG() throws {
        let tiff = try Self.tiffData(width: 7, height: 5)
        guard case .image(let data) = ComposerPasteClassifier.classify(.init(imageData: tiff)) else {
            return XCTFail("TIFF on the pasteboard must classify as an image")
        }
        XCTAssertEqual(Array(data.prefix(8)), [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], "not PNG")
        let source = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
        XCTAssertEqual(CGImageSourceGetType(source) as String?, UTType.png.identifier)
        let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        XCTAssertEqual(properties[kCGImagePropertyPixelWidth] as? Int, 7)
        XCTAssertEqual(properties[kCGImagePropertyPixelHeight] as? Int, 5)
    }

    func testUnreadableImageBytesFallThroughToText() {
        let garbage = Data("this is not a picture".utf8)
        XCTAssertEqual(ComposerPasteClassifier.classify(.init(imageData: garbage)), .passthrough)
    }

    func testPastedImageFilenameIsLocalizedAndStampedWithTheWallClock() throws {
        let calendar = Calendar.current
        let lateEvening = try XCTUnwrap(calendar.date(bySettingHour: 23, minute: 58, second: 12, of: Date()))
        XCTAssertEqual(ComposerPasteClassifier.pastedImageFilename(at: lateEvening), "Pasted image 23-58-12.png")
        let earlyMorning = try XCTUnwrap(calendar.date(bySettingHour: 7, minute: 5, second: 3, of: Date()))
        XCTAssertEqual(ComposerPasteClassifier.pastedImageFilename(at: earlyMorning), "Pasted image 07-05-03.png")
    }

    // MARK: The pasted-image card

    @MainActor
    func testPastedImageBecomesAReadyPNGCard() async throws {
        let png = try Self.pngData(width: 12, height: 9)
        let viewModel = try makeViewModel()

        await viewModel.attachPastedImage(png, filename: "Pasted image 12-00-00.png", visionEnabled: true)

        let card = try XCTUnwrap(viewModel.pendingAttachments.first)
        XCTAssertEqual(viewModel.pendingAttachments.count, 1)
        XCTAssertEqual(card.filename, "Pasted image 12-00-00.png")
        XCTAssertEqual(card.mimeType, "image/png")
        XCTAssertEqual(card.imageData, png, "an image that already fits keeps its exact bytes")
        XCTAssertEqual(card.sizeBytes, png.count)
        XCTAssertTrue(card.isImage)
        XCTAssertEqual(viewModel.extractionState(for: card), .ready(truncation: nil))
        XCTAssertFalse(viewModel.isExtractingAttachments)
        XCTAssertTrue(viewModel.hasSendablePendingAttachments)
        XCTAssertNil(viewModel.lastError)
    }

    @MainActor
    func testPastedImageIsDownscaledLikeADroppedFile() async throws {
        let wide = try Self.pngData(width: 3000, height: 30)
        let viewModel = try makeViewModel()

        await viewModel.attachPastedImage(wide, filename: "Pasted image 12-00-01.png", visionEnabled: true)

        let card = try XCTUnwrap(viewModel.pendingAttachments.first)
        XCTAssertEqual(viewModel.extractionState(for: card), .ready(truncation: nil))
        XCTAssertEqual(card.mimeType, "image/png")
        let data = try XCTUnwrap(card.imageData)
        let source = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
        let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        XCTAssertEqual(properties[kCGImagePropertyPixelWidth] as? Int, 2048)
        XCTAssertEqual(card.sizeBytes, data.count)
    }

    @MainActor
    func testOversizedPastedImageFailsOnItsCard() async throws {
        let viewModel = try makeViewModel()
        let tooBig = Data(count: 20 * 1024 * 1024 + 1)

        await viewModel.attachPastedImage(tooBig, filename: "Pasted image 12-00-02.png", visionEnabled: true)

        let card = try XCTUnwrap(viewModel.pendingAttachments.first, "a failed paste stays on the strip")
        XCTAssertEqual(
            viewModel.extractionState(for: card),
            .failed(message: "Could not read Pasted image 12-00-02.png: image exceeds the 20MB attachment limit")
        )
        XCTAssertNil(card.imageData)
        XCTAssertFalse(viewModel.hasSendablePendingAttachments)
        XCTAssertFalse(viewModel.isExtractingAttachments)
    }

    @MainActor
    func testNonImagePastedBytesFailOnTheirCard() async throws {
        let viewModel = try makeViewModel()

        await viewModel.attachPastedImage(Data("hello".utf8), filename: "Pasted image 12-00-03.png", visionEnabled: true)

        let card = try XCTUnwrap(viewModel.pendingAttachments.first)
        XCTAssertEqual(
            viewModel.extractionState(for: card),
            .failed(message: "Could not read Pasted image 12-00-03.png: not a readable image")
        )
        XCTAssertNil(card.imageData)
        XCTAssertFalse(viewModel.hasSendablePendingAttachments)
    }

    // MARK: Vision gate

    @MainActor
    func testPastedImageIsRefusedVisiblyWhenTheModelCannotSee() async throws {
        let png = try Self.pngData(width: 4, height: 4)
        let viewModel = try makeViewModel()

        await viewModel.attachPastedImage(png, filename: "Pasted image 12-00-04.png", visionEnabled: false)

        let card = try XCTUnwrap(viewModel.pendingAttachments.first, "the refusal is a card, not a silent no-op")
        XCTAssertEqual(card.filename, "Pasted image 12-00-04.png")
        XCTAssertEqual(viewModel.extractionState(for: card), .failed(message: "This model can't see images."))
        XCTAssertNil(card.imageData, "nothing the server would have to ignore")
        XCTAssertFalse(viewModel.hasSendablePendingAttachments)
        XCTAssertFalse(viewModel.isExtractingAttachments)
        XCTAssertNil(viewModel.lastError)

        viewModel.removeAttachment(card)
        XCTAssertTrue(viewModel.pendingAttachments.isEmpty)
        XCTAssertTrue(viewModel.attachmentStates.isEmpty)
    }

    /// The same gate closes the existing gap for image FILES (drop, Finder
    /// paste): documents still attach, the image is refused on its card,
    /// and the extractor never sees it.
    @MainActor
    func testImageFileIsRefusedWhenTheModelCannotSeeWhileDocumentsStillAttach() async throws {
        let directory = try scratchDirectory()
        let mdURL = directory.appendingPathComponent("notes.md")
        try "# Notes".write(to: mdURL, atomically: true, encoding: .utf8)
        let pngURL = directory.appendingPathComponent("shot.png")
        try Self.pngData(width: 4, height: 4).write(to: pngURL)

        final class Seen: @unchecked Sendable {
            private let lock = NSLock()
            private var _urls: [URL] = []
            func record(_ url: URL) { lock.lock(); defer { lock.unlock() }; _urls.append(url) }
            var urls: [URL] { lock.lock(); defer { lock.unlock() }; return _urls }
        }
        let seen = Seen()
        let viewModel = try makeViewModel { url in
            seen.record(url)
            return try ChatViewModel.extractAttachment(from: url)
        }

        await viewModel.attach([mdURL, pngURL], visionEnabled: false)

        XCTAssertEqual(viewModel.pendingAttachments.map(\.filename), ["notes.md", "shot.png"])
        let notes = viewModel.pendingAttachments[0]
        XCTAssertEqual(notes.extractedText, "# Notes")
        XCTAssertEqual(viewModel.extractionState(for: notes), .ready(truncation: nil))

        let shot = viewModel.pendingAttachments[1]
        XCTAssertEqual(viewModel.extractionState(for: shot), .failed(message: "This model can't see images."))
        XCTAssertNil(shot.imageData)
        XCTAssertEqual(seen.urls, [mdURL], "the image is refused before any decoding work")
        XCTAssertTrue(viewModel.hasSendablePendingAttachments, "the document is still sendable")
        XCTAssertFalse(viewModel.isExtractingAttachments)
    }

    @MainActor
    func testImageFileStillAttachesWhenTheModelCanSee() async throws {
        let directory = try scratchDirectory()
        let pngURL = directory.appendingPathComponent("shot.png")
        try Self.pngData(width: 4, height: 4).write(to: pngURL)
        let viewModel = try makeViewModel()

        await viewModel.attach([pngURL], visionEnabled: true)

        let shot = try XCTUnwrap(viewModel.pendingAttachments.first)
        XCTAssertNotNil(shot.imageData)
        XCTAssertEqual(viewModel.extractionState(for: shot), .ready(truncation: nil))
    }

    // MARK: Fixtures

    private static func solidImage(width: Int, height: Int) throws -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw NSError(domain: "PasteAttachmentTests", code: 1) }
        context.setFillColor(CGColor(red: 0.8, green: 0.3, blue: 0.2, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let image = context.makeImage() else { throw NSError(domain: "PasteAttachmentTests", code: 2) }
        return image
    }

    private static func encoded(_ image: CGImage, as type: UTType) throws -> Data {
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, type.identifier as CFString, 1, nil) else {
            throw NSError(domain: "PasteAttachmentTests", code: 3)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw NSError(domain: "PasteAttachmentTests", code: 4) }
        return output as Data
    }

    static func pngData(width: Int, height: Int) throws -> Data {
        try encoded(try solidImage(width: width, height: height), as: .png)
    }

    static func tiffData(width: Int, height: Int) throws -> Data {
        try encoded(try solidImage(width: width, height: height), as: .tiff)
    }
}
