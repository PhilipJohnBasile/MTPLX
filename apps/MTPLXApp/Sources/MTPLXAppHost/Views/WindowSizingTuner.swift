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
            maintenanceTimer?.invalidate()
            maintenanceTimer = nil
            guard window != nil else { return }
            DispatchQueue.main.async { [weak self] in
                _ = self?.applyIfNeeded()
            }
            // 1 Hz backstop sweep. The constraint-pass chain below only
            // sustains itself while re-arms are being OBSERVED — if a
            // scene update re-arms sizingOptions in a period where
            // nothing lays out this view and no pass is chained (found
            // 2026-08-18: post-stream, an ambient animation re-armed per
            // frame and the walk burned ~53% CPU at "rest" on an
            // 11k-token transcript), this timer notices within a second
            // and re-seeds the chain. Costs one options read per second;
            // fires no constraint work when options are already empty.
            maintenanceTimer = Timer.scheduledTimer(
                withTimeInterval: 1.0, repeats: true
            ) { [weak self] _ in
                MainActor.assumeIsolated {
                    guard let self else { return }
                    if self.applyIfNeeded(), !self.needsUpdateConstraints {
                        self.needsUpdateConstraints = true
                    }
                }
            }
            maintenanceTimer?.tolerance = 0.3
        }

        override func updateConstraints() {
            let observedRearm = applyIfNeeded()
            super.updateConstraints()
            // Chain another pass ONLY while someone is actively
            // re-arming the options. At true rest the options stay
            // empty, no re-arm is observed, and the chain dies — the
            // unconditional per-cycle re-arm tried on 2026-08-17 kept
            // the runloop at display cadence and burned ~95% CPU idle.
            if observedRearm {
                DispatchQueue.main.async { [weak self] in
                    guard let self, !self.needsUpdateConstraints else { return }
                    self.needsUpdateConstraints = true
                }
            }
        }

        // Activity-proportional re-arm: layout() runs whenever our
        // superview lays out (every streaming flush), so the NEXT
        // constraints pass revisits us and neutralizes any scene
        // re-arm first.
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
        private var maintenanceTimer: Timer?

        /// Neutralizes the hosting view's sizing options and pins the
        /// window minimum. Returns true when it OBSERVED non-empty
        /// options (i.e. something re-armed since the last clear) —
        /// the signal the caller uses to decide whether to chain
        /// another constraints pass.
        @discardableResult
        func applyIfNeeded() -> Bool {
            // 2026-08-17 field regression: the original cast
            // `window.contentView as? MTPLXHostingSizingConfigurable`
            // silently failed — the hosting view is a DESCENDANT of the
            // content view on this window shape — so the tuner never
            // applied and the min-size storm it documents shipped in
            // 2.8.x (34% of the idle main thread; worse while
            // streaming). Search the subtree, and re-assert on every
            // update instead of latching: SwiftUI scene updates can
            // re-arm `sizingOptions`, and both writes are idempotent.
            guard WindowSizingTuner.isEnabled, let window else { return false }
            guard let hosting = Self.hostingView(in: window.contentView, depth: 4)
            else { return false }
            let observedRearm = !hosting.mtplxSizingOptions.isEmpty
            if observedRearm {
                if Self.debugEnabled {
                    rearmObservations += 1
                    if rearmObservations <= 8 || rearmObservations % 100 == 0 {
                        FileHandle.standardError.write(Data(
                            "[sizing-tuner] non-empty options seen (n=\(rearmObservations)) value=\(hosting.mtplxSizingOptions.rawValue)\n".utf8
                        ))
                    }
                }
                hosting.mtplxSizingOptions = []
            }
            if window.contentMinSize != WindowSizingTuner.contentMinSize {
                window.contentMinSize = WindowSizingTuner.contentMinSize
            }
            return observedRearm
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
