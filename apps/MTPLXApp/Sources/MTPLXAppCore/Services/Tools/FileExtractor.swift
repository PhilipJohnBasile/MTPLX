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

    public init(
        filename: String,
        mimeType: String,
        combinedText: String,
        sizeBytes: Int,
        pageCount: Int? = nil
    ) {
        self.filename = filename
        self.mimeType = mimeType
        self.combinedText = combinedText
        self.sizeBytes = sizeBytes
        self.pageCount = pageCount
    }

    public var isEmpty: Bool {
        combinedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

public enum FileExtractorError: LocalizedError {
    case unreadable(filename: String, reason: String)
    case unsupported(filename: String, ext: String)

    public var errorDescription: String? {
        switch self {
        case .unreadable(let name, let reason):
            return "Could not read \(name): \(reason)"
        case .unsupported(let name, let ext):
            return "Unsupported file type for \(name): .\(ext)"
        }
    }
}

public enum FileExtractor {
    /// Supported file extensions the composer's NSOpenPanel should
    /// allow. Kept in one place so the UI and extractor stay in sync.
    public static let supportedExtensions: Set<String> = [
        "pdf", "txt", "md", "docx",
    ]

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

        switch ext {
        case "pdf":
            #if canImport(PDFKit)
            let (text, pages) = extractPDF(from: url)
            return ExtractedFile(
                filename: filename,
                mimeType: mime,
                combinedText: text,
                sizeBytes: data.count,
                pageCount: pages
            )
            #else
            throw FileExtractorError.unsupported(filename: filename, ext: ext)
            #endif

        case "txt", "md":
            let text = (String(data: data, encoding: .utf8) ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ExtractedFile(
                filename: filename,
                mimeType: mime,
                combinedText: text,
                sizeBytes: data.count
            )

        case "docx":
            let text = (extractDocxText(from: data) ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ExtractedFile(
                filename: filename,
                mimeType: mime,
                combinedText: text,
                sizeBytes: data.count
            )

        default:
            let text = (String(data: data, encoding: .utf8) ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ExtractedFile(
                filename: filename,
                mimeType: mime,
                combinedText: text,
                sizeBytes: data.count
            )
        }
    }

    // MARK: - Private helpers

    #if canImport(PDFKit)
    private static func extractPDF(from url: URL) -> (text: String, pageCount: Int) {
        guard let document = PDFDocument(url: url) else { return ("", 0) }
        var parts: [String] = []
        for index in 0..<document.pageCount {
            guard let page = document.page(at: index),
                let pageText = page.string?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                !pageText.isEmpty
            else { continue }
            parts.append(pageText)
        }
        return (parts.joined(separator: "\n\n"), document.pageCount)
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
