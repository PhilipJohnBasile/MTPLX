import AppKit
import SwiftUI

// MARK: - WindowSizingTuner
//
// Kills the per-display-cycle window min-size storm (2026-07-31 perf
// hunt, deferred item 1; receipts in outputs/app-frontend-hunt-*.md).
//
// The mechanism being removed: with default `sizingOptions`
// (.standardBounds), AppKit re-derives the window's content-size
// extrema from the SwiftUI graph on constraint invalidation —
// `NSHostingView.updateConstraints →
// updateWindowContentSizeExtremaIfNecessary → minSize()` — which is a
// FULL `sizeThatFits` walk of every realized view. Streaming
// invalidates constraints continuously, so that walk ran every display
// cycle and was the length-independent ~50 ms per-flush floor (and
// ~2.7 s of a 14 s scroll sample on a 20k-token transcript).
//
// Our root view's size is determined by the window (it fills it), so
// deriving window extrema from content buys nothing: set
// `sizingOptions = []` on the window's hosting view and pin an
// explicit `contentMinSize` equal to the floor ContentView used to
// express via `.frame(minWidth: 420, minHeight: 540)`. Sheets and
// overlays own separate hosting views and keep default behavior.
//
// `MTPLX_APP_SIZING_TUNER=0` disables (diagnostic escape hatch).

/// Unconstrained protocol conformance so the generic
/// `NSHostingView<Content>` can be recognized and configured without
/// knowing `Content` at the call site.
@MainActor
private protocol MTPLXHostingSizingConfigurable: AnyObject {
    var mtplxSizingOptions: NSHostingSizingOptions { get set }
}

extension NSHostingView: MTPLXHostingSizingConfigurable {
    var mtplxSizingOptions: NSHostingSizingOptions {
        get { sizingOptions }
        set { sizingOptions = newValue }
    }
}

struct WindowSizingTuner: NSViewRepresentable {
    static let contentMinSize = NSSize(width: 420, height: 540)

    static var isEnabled: Bool {
        switch ProcessInfo.processInfo.environment["MTPLX_APP_SIZING_TUNER"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "0", "false", "off", "no": return false
        default: return true
        }
    }

    func makeNSView(context: Context) -> TunerView {
        TunerView()
    }

    func updateNSView(_ nsView: TunerView, context: Context) {
        nsView.applyIfNeeded()
    }

    final class TunerView: NSView {
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            DispatchQueue.main.async { [weak self] in
                self?.applyIfNeeded()
            }
        }

        func applyIfNeeded() {
            // 2026-08-17 field regression: the original cast
            // `window.contentView as? MTPLXHostingSizingConfigurable`
            // silently failed — the hosting view is a DESCENDANT of the
            // content view on this window shape — so the tuner never
            // applied and the min-size storm it documents shipped in
            // 2.8.x (34% of the idle main thread; worse while
            // streaming). Search the subtree, and re-assert on every
            // update instead of latching: SwiftUI scene updates can
            // re-arm `sizingOptions`, and both writes are idempotent.
            guard WindowSizingTuner.isEnabled, let window else { return }
            guard let hosting = Self.hostingView(in: window.contentView, depth: 4)
            else { return }
            if !hosting.mtplxSizingOptions.isEmpty {
                hosting.mtplxSizingOptions = []
            }
            if window.contentMinSize != WindowSizingTuner.contentMinSize {
                window.contentMinSize = WindowSizingTuner.contentMinSize
            }
        }

        private static func hostingView(
            in view: NSView?,
            depth: Int
        ) -> MTPLXHostingSizingConfigurable? {
            guard let view else { return nil }
            if let hosting = view as? MTPLXHostingSizingConfigurable {
                return hosting
            }
            guard depth > 0 else { return nil }
            for subview in view.subviews {
                if let found = hostingView(in: subview, depth: depth - 1) {
                    return found
                }
            }
            return nil
        }
    }
}
