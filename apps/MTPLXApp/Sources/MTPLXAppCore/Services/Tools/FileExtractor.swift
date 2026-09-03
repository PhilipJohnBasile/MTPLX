import Foundation
#if canImport(PDFKit)
import PDFKit
#endif

// MARK: - File extraction
//
// Port of Aphanes V2's `DocumentTextExtractor`. The whole concern is
// turning a user-attached file into plain text the model can consume
// inside the next user message — server-side multimodal is out of
// scope for v1.
//
// Per format:
//   - pdf:  PDFKit page-by-page, joined with "\n\n"
//   - txt/md:  UTF-8 string, trimmed
//   - docx: spawn `/usr/bin/unzip` to extract the archive in a temp
//           directory, then regex-strip `word/document.xml` paragraph
//           tags. Aphanes does the exact same thing and ships it; not
//           pretty but it works without a third-party docx parser.
//   - unknown: best-effort UTF-8 fallback

public struct ExtractedFile: Sendable, Hashable {
    public var filename: String
    public var mimeType: String
    public var combinedText: String
    public var sizeBytes: Int
    public var pageCount: Int?
    /// What the caps cut, when they cut anything. The text itself ends
    /// with a plain marker saying the same, so the model knows the
    /// document continues past what it was given.
    public var truncation: ExtractionTruncation?

    public init(
        filename: String,
        mimeType: String,
        combinedText: String,
        sizeBytes: Int,
        pageCount: Int? = nil,
        truncation: ExtractionTruncation? = nil
    ) {
        self.filename = filename
        self.mimeType = mimeType
        self.combinedText = combinedText
        self.sizeBytes = sizeBytes
        self.pageCount = pageCount
        self.truncation = truncation
    }

    public var isEmpty: Bool {
        combinedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

/// How much of an attachment was kept when it exceeded the extraction
/// caps. Either span is nil when that cap did not apply.
public struct ExtractionTruncation: Sendable, Hashable {
    public struct Span: Sendable, Hashable {
        public var included: Int
        public var total: Int

        public init(included: Int, total: Int) {
            self.included = included
            self.total = total
        }
    }

    public var pages: Span?
    public var characters: Span?

    public init(pages: Span? = nil, characters: Span? = nil) {
        self.pages = pages
        self.characters = characters
    }

    /// One line for the attachment card, in the active language.
    public var summary: String {
        var parts: [String] = []
        if let pages {
            parts.append(tr("Truncated to %lld of %lld pages", pages.included, pages.total))
        }
        if let characters {
            parts.append(
                tr(
                    "Truncated to %@ of %@ characters",
                    Self.grouped(characters.included),
                    Self.grouped(characters.total)
                )
            )
        }
        return parts.joined(separator: " · ")
    }

    /// The marker appended to the extracted text so the model knows the
    /// document continues past what it was given.
    var modelMarker: String {
        var parts: [String] = []
        if let pages {
            parts.append("first \(pages.included) of \(pages.total) pages")
        }
        if let characters {
            parts.append("first \(characters.included) of \(characters.total) characters")
        }
        return "[Attachment truncated: \(parts.joined(separator: ", ")) included]"
    }

    private static func grouped(_ value: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.locale = L10n.language.locale
        return formatter.string(from: NSNumber(value: value)) ?? String(value)
    }
}

public enum FileExtractorError: LocalizedError {
    case unreadable(filename: String, reason: String)
    case unsupported(filename: String, ext: String)

    public var errorDescription: String? {
        switch self {
        case .unreadable(let name, let reason):
            return tr("Could not read %@: %@", name, reason)
        case .unsupported(let name, let ext):
            return tr("Unsupported file type for %@: .%@", name, ext)
        }
    }
}

public enum FileExtractor {
    /// Supported file extensions the composer's NSOpenPanel should
    /// allow. Kept in one place so the UI and extractor stay in sync.
    public static let supportedExtensions: Set<String> = [
        "pdf", "txt", "md", "docx",
    ]

    // Extraction caps. The text is inlined into the user message
    // verbatim on every turn, so an unbounded attachment is an unbounded
    // prefill for a local model. Generous on purpose: a long paper or a
    // large source file fits; a whole book does not, and the user is
    // told exactly what was kept (card note) as is the model (marker).
    /// Most PDF pages read; the rest of the document is not extracted.
    public static let maxPDFPages = 500
    /// Most characters kept from any attachment.
    public static let maxCharacters = 200_000

    public static func mimeType(for pathExtension: String) -> String {
        switch pathExtension.lowercased() {
        case "md": return "text/markdown"
        case "txt": return "text/plain"
        case "pdf": return "application/pdf"
        case "docx":
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        default: return "application/octet-stream"
        }
    }

    /// Extracts plain text from a local file URL. Throws on completely
    /// unreadable content; returns an `ExtractedFile` with empty
    /// `combinedText` for files that read but contain no extractable
    /// text (e.g. an empty PDF).
    public static func extract(from url: URL) throws -> ExtractedFile {
        let filename = url.lastPathComponent
        let ext = url.pathExtension.lowercased()
        let mime = mimeType(for: ext)

        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw FileExtractorError.unreadable(
                filename: filename,
                reason: error.localizedDescription
            )
        }

        let rawText: String
        var pageCount: Int?
        var pageTruncation: ExtractionTruncation.Span?
        switch ext {
        case "pdf":
            #if canImport(PDFKit)
            let pdf = extractPDF(from: url)
            rawText = pdf.text
            pageCount = pdf.pageCount
            pageTruncation = pdf.truncatedPages
            #else
            throw FileExtractorError.unsupported(filename: filename, ext: ext)
            #endif

        case "docx":
            rawText = (extractDocxText(from: data) ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)

        default:
            // txt, md, and a best-effort UTF-8 read of anything else.
            rawText = (String(data: data, encoding: .utf8) ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let capped = applyCharacterCap(to: rawText, pageTruncation: pageTruncation)
        return ExtractedFile(
            filename: filename,
            mimeType: mime,
            combinedText: capped.text,
            sizeBytes: data.count,
            pageCount: pageCount,
            truncation: capped.truncation
        )
    }

    /// Cuts `text` to `maxCharacters` and, when anything was cut (pages
    /// or characters), appends the marker the model reads.
    static func applyCharacterCap(
        to text: String,
        pageTruncation: ExtractionTruncation.Span?
    ) -> (text: String, truncation: ExtractionTruncation?) {
        var truncation = ExtractionTruncation(pages: pageTruncation)
        var kept = text
        if text.count > maxCharacters {
            kept = String(text.prefix(maxCharacters))
            truncation.characters = .init(included: maxCharacters, total: text.count)
        }
        guard truncation.pages != nil || truncation.characters != nil else {
            return (kept, nil)
        }
        return (kept + "\n\n" + truncation.modelMarker, truncation)
    }

    // MARK: - Private helpers

    #if canImport(PDFKit)
    private static func extractPDF(
        from url: URL
    ) -> (text: String, pageCount: Int, truncatedPages: ExtractionTruncation.Span?) {
        guard let document = PDFDocument(url: url) else { return ("", 0, nil) }
        let readPages = min(document.pageCount, maxPDFPages)
        var parts: [String] = []
        for index in 0..<readPages {
            guard let page = document.page(at: index),
                let pageText = page.string?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                !pageText.isEmpty
            else { continue }
            parts.append(pageText)
        }
        let truncated: ExtractionTruncation.Span? = readPages < document.pageCount
            ? .init(included: readPages, total: document.pageCount)
            : nil
        return (parts.joined(separator: "\n\n"), document.pageCount, truncated)
    }
    #endif

    /// Spawns `/usr/bin/unzip` to extract a `.docx` archive and parses
    /// `word/document.xml` with regex. Verbatim port of Aphanes V2's
    /// implementation — not elegant, but ships without a third-party
    /// dependency and survives almost any well-formed docx.
    private static func extractDocxText(from data: Data) -> String? {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        do {
            try FileManager.default.createDirectory(
                at: tempDir,
                withIntermediateDirectories: true
            )
            let zipURL = tempDir.appendingPathComponent("doc.zip")
            try data.write(to: zipURL)

            let unzipDir = tempDir.appendingPathComponent("unzipped")
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
            process.arguments = ["-o", "-q", zipURL.path, "-d", unzipDir.path]
            // Bounded wait (#158): a hostile or degenerate archive
            // must not park the attachment flow forever — a timeout
            // reads as "not extractable", same as a malformed docx.
            let watchdog = SubprocessWatchdog(process)
            try process.run()
            guard watchdog.wait(for: process, timeout: 30) else {
                return nil
            }

            let xmlURL = unzipDir.appendingPathComponent("word/document.xml")
            guard let xmlData = try? Data(contentsOf: xmlURL) else { return nil }
            let xmlString = String(data: xmlData, encoding: .utf8) ?? ""
            let stripped = xmlString
                .replacingOccurrences(of: "<w:p[^>]*>", with: "\n", options: .regularExpression)
                .replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
                .replacingOccurrences(of: "&amp;", with: "&")
                .replacingOccurrences(of: "&lt;", with: "<")
                .replacingOccurrences(of: "&gt;", with: ">")
                .replacingOccurrences(of: "&quot;", with: "\"")
                .replacingOccurrences(of: "&apos;", with: "'")
            return stripped
        } catch {
            return nil
        }
    }
}
