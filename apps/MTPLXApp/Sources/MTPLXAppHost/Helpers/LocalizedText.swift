import SwiftUI

// MARK: - Text(markdown:)
//
// A translated string that carries inline Markdown (code spans, bold).
// `Text(String)` renders verbatim and `Text(LocalizedStringKey)` would
// re-resolve the translated text against Bundle.main in the process
// language, so neither works for localized Markdown. This parses the
// already-translated string directly and falls back to plain text.

extension Text {
    init(markdown source: String) {
        if let attributed = try? AttributedString(
            markdown: source,
            options: AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            self.init(attributed)
        } else {
            self.init(source)
        }
    }
}
