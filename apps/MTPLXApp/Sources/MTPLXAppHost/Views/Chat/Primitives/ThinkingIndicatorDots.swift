import SwiftUI
import MTPLXAppCore

// MARK: - ThinkingIndicatorDots
//
// Three-dot pulse used next to "Thinking" / "Searching the web" /
// "Solving" titles while a tool call, reasoning stream, or benchmark
// problem is in flight.
//
// The cadence used to be driven by a `Timer.publish(...)` stored as a
// plain `let`. Any parent that re-rendered faster than the 0.18s tick
// (the benchmark live card flushes streamed reasoning every 80ms)
// recreated the publisher and re-subscribed before it ever fired, so
// the dots froze on screen. The second version self-animated via a
// `.repeatForever` kicked off in `.onAppear` — and leaked: SwiftUI
// cannot reliably cancel repeatForever when the view unmounts (the
// documented GaugeView pathology), so every settled turn left an
// invisible 60 Hz animation driving display cycles forever. Post-
// stream the app burned 25-50% CPU at "rest", compounding per
// completed turn (2026-08-18 sampling). This version derives the
// pulse phase from `TimelineView(.animation)` — frame scheduling is
// owned by the mounted view, so unmounting is a hard stop by
// construction. Reduce Motion pauses the schedule outright.

struct ThinkingIndicatorDots: View {
    var color: Color = Brand.typeSecondary
    var size: CGFloat = 4
    var spacing: CGFloat = 3

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private static let period: Double = 1.0
    private static let dotDelay: Double = 0.16

    var body: some View {
        TimelineView(
            .animation(minimumInterval: 1.0 / 30.0, paused: reduceMotion)
        ) { context in
            let now = context.date.timeIntervalSinceReferenceDate
            HStack(spacing: spacing) {
                ForEach(0..<3, id: \.self) { index in
                    let phase = Self.phase(at: now, index: index)
                    Circle()
                        .fill(color)
                        .frame(width: size, height: size)
                        .opacity(0.35 + 0.60 * phase)
                        .scaleEffect(0.78 + 0.22 * phase)
                }
            }
        }
        .accessibilityLabel("Working")
    }

    /// 0→1→0 ease-in-out pulse, staggered per dot — same visual as the
    /// old autoreversing 0.5 s repeatForever.
    private static func phase(at time: Double, index: Int) -> Double {
        let shifted = time - Double(index) * dotDelay
        let cycle = shifted.truncatingRemainder(dividingBy: period) / period
        let triangle = cycle < 0.5 ? cycle * 2 : (1 - cycle) * 2
        return triangle * triangle * (3 - 2 * triangle)
    }
}
