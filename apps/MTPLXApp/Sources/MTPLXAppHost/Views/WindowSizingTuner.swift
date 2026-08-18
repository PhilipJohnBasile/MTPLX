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

    // Cached: this gate is consulted on every constraints pass, and
    // ProcessInfo.environment rebuilds its whole dictionary per access
    // (uncached, it alone was ~26% of the idle main thread on
    // 2026-08-17 — the same bug class as the AIMEDiagnostics gate).
    static let isEnabled: Bool = {
        switch ProcessInfo.processInfo.environment["MTPLX_APP_SIZING_TUNER"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "0", "false", "off", "no": return false
        default: return true
        }
    }()

    func makeNSView(context: Context) -> TunerView {
        TunerView()
    }

    func updateNSView(_ nsView: TunerView, context: Context) {
        nsView.applyIfNeeded()
    }

    final class TunerView: NSView {
        // Participate in EVERY constraints pass. Constraint updates run
        // bottom-up, and this view is a descendant of the hosting view,
        // so `updateConstraints` below neutralizes `sizingOptions`
        // BEFORE NSHostingView.updateConstraints derives window extrema
        // in the same pass — a SwiftUI scene update re-arming the
        // options between cycles can never buy another transcript walk.
        // (2026-08-17: the one-shot apply verifiably cleared the
        // options, yet the A/B profile still showed ~2,300 walk samples
        // — something re-arms; per-pass neutralization is race-free.)
        override class var requiresConstraintBasedLayout: Bool { true }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            DispatchQueue.main.async { [weak self] in
                self?.applyIfNeeded()
            }
        }

        override func updateConstraints() {
            applyIfNeeded()
            super.updateConstraints()
        }

        // Activity-proportional re-arm: layout() runs whenever our
        // superview lays out (every streaming flush; never at rest), so
        // the NEXT constraints pass revisits us and neutralizes any
        // scene re-arm first. The earlier async-per-cycle re-arm kept
        // the runloop spinning at display cadence even at idle — the
        // app burned ~95% CPU at rest doing nothing (2026-08-17).
        override func layout() {
            super.layout()
            if !needsUpdateConstraints {
                needsUpdateConstraints = true
            }
        }

        // Cached: applyIfNeeded now runs per constraints pass, and
        // ProcessInfo.environment rebuilds its dictionary per access.
        static let debugEnabled: Bool =
            ProcessInfo.processInfo.environment["MTPLX_SIZING_TUNER_DEBUG"] == "1"

        private var rearmObservations = 0

        func applyIfNeeded() {
            if Self.debugEnabled,
               let window,
               let hosting = Self.hostingView(in: window.contentView, depth: 4),
               !hosting.mtplxSizingOptions.isEmpty {
                rearmObservations += 1
                if rearmObservations <= 8 || rearmObservations % 100 == 0 {
                    FileHandle.standardError.write(Data(
                        "[sizing-tuner] non-empty options seen (n=\(rearmObservations)) value=\(hosting.mtplxSizingOptions.rawValue)\n".utf8
                    ))
                }
            }
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
