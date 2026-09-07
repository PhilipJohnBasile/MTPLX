import AppKit
import Foundation

// MARK: - ComposerPasteClassifier

/// Pure decision logic for ⌘V in the chat composer.
///
/// The AppKit bridge (`ComposerNSTextView.paste(_:)`) reads the general
/// pasteboard into a `Contents` value and asks here what the paste means:
/// attachments — files copied in Finder, an image copied from a screenshot,
/// a browser or Preview — or "let NSTextView paste text as it always has".
/// Keeping the decision out of the text view makes the precedence testable
/// without a live pasteboard, the same split `ComposerTextSync` uses for
/// the IME gate.
///
/// Precedence:
///   1. file URLs that exist on disk → `.files` (the drag-drop path)
///   2. image bytes                  → `.image` (always PNG)
///   3. anything else                → `.passthrough`
///
/// An image wins over a string on purpose: browsers put the image's URL on
/// the pasteboard as text next to the image, and a user who copied a
/// picture expects the picture. Paste and Match Style (⌥⇧⌘V) is not
/// intercepted, so the text rendition stays one keystroke away.
public enum ComposerPasteClassifier {
    public enum Action: Equatable {
        case files([URL])
        case image(Data)
        case passthrough
    }

    /// What the bridge read off the pasteboard. `urls` is every `NSURL` the
    /// pasteboard offered, file or not; `imageData` is the best image
    /// rendition found (PNG preferred, then TIFF, then anything `NSImage`
    /// can read), or nil.
    public struct Contents: Equatable {
        public var urls: [URL]
        public var imageData: Data?

        public init(urls: [URL] = [], imageData: Data? = nil) {
            self.urls = urls
            self.imageData = imageData
        }
    }

    public static func classify(_ contents: Contents) -> Action {
        let files = contents.urls.filter {
            $0.isFileURL && FileManager.default.fileExists(atPath: $0.path)
        }
        if !files.isEmpty {
            return .files(files)
        }
        if let data = contents.imageData, let png = pngData(from: data) {
            return .image(png)
        }
        return .passthrough
    }

    /// PNG bytes for any bitmap the pasteboard can carry. PNG is kept
    /// byte-for-byte (a screenshot stays lossless and untouched); TIFF and
    /// everything else re-encode through `NSBitmapImageRep`. Nil when the
    /// bytes are not an image AppKit can decode, so the paste falls
    /// through to text.
    static func pngData(from data: Data) -> Data? {
        if data.starts(with: pngSignature) {
            return data
        }
        let bitmap = NSBitmapImageRep(data: data)
            ?? NSImage(data: data)?.tiffRepresentation.flatMap { NSBitmapImageRep(data: $0) }
        return bitmap?.representation(using: .png, properties: [:])
    }

    private static let pngSignature: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]

    /// "Pasted image HH-mm-ss.png". A pasted image has no file name, so the
    /// card gets a localized one stamped with the local wall-clock time,
    /// which also keeps two pastes in one message apart. `String(format:)`
    /// without a locale keeps the digits ASCII in every language.
    public static func pastedImageFilename(at date: Date = Date()) -> String {
        let parts = Calendar.current.dateComponents([.hour, .minute, .second], from: date)
        let stamp = String(
            format: "%02d-%02d-%02d", parts.hour ?? 0, parts.minute ?? 0, parts.second ?? 0
        )
        return "\(tr("Pasted image")) \(stamp).png"
    }
}
