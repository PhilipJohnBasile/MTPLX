import SwiftUI
import Charts
import MTPLXAppCore

// MARK: - DecodeChart
//
// 5-minute decode TPS time series. Apple Charts AreaMark + LineMark with
// chrome-silver stops. RuleMark for the rolling mean. Animation key uses
// `rolling.count` so we only re-animate on point insertion, not on every
// state mutation.

struct DecodeChart: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore
    private static let maxRenderedPoints = 240

    /// Continuation of the LiveTab lift stagger started by `TileRow`
    /// (indices 0…4) and `AcceptanceSection` (index 5). Decode picks
    /// up at index 6 so the lift wave keeps reading as one beat as
    /// it travels down the page.
    static let liftStaggerIndex: Int = 6

    var body: some View {
        let rolling = backend.observedCompletionCount > 0 ? backend.rolling : nil
        let motion = !backend.configuration.performanceLock
        let isRunning = backend.daemonState.kind == .running
        let liftAnimation: Animation? = themeStore.reduceMotionPreference
            ? nil
            : Motion.surfaceLiftSpring
        let liftDelay: TimeInterval = themeStore.reduceMotionPreference
            ? 0
            : Double(Self.liftStaggerIndex) * Motion.surfaceLiftStaggerStep
        let points = rolling.map(seriesPoints) ?? []
        let mean = rolling?.mean

        VStack(alignment: .leading, spacing: 10) {
            header(rolling: rolling)

            if points.count >= 2 {
                Chart {
                    ForEach(Array(points.enumerated()), id: \.offset) { idx, tps in
                        AreaMark(
                            x: .value("sample", idx),
                            y: .value("TPS", tps)
                        )
                        .foregroundStyle(
                            LinearGradient(
                                colors: [
                                    Brand.accentChrome.opacity(0.40),
                                    Brand.accentChrome.opacity(0.02),
                                ],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        .interpolationMethod(.monotone)

                        LineMark(
                            x: .value("sample", idx),
                            y: .value("TPS", tps)
                        )
                        .foregroundStyle(Brand.chromeAccent)
                        .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round))
                        .interpolationMethod(.monotone)
                    }

                    if let mean {
                        RuleMark(y: .value("average", mean))
                            .foregroundStyle(Brand.textHighlight.opacity(0.45))
                            .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                            .annotation(position: .top, alignment: .leading) {
                                Text(tr("average %@ TPS", Format.tps(mean)))
                                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                                    .foregroundStyle(Brand.textHighlight.opacity(0.65))
                                    .padding(.leading, 4)
                            }
                    }
                }
                .chartXScale(domain: 0...Double(points.count - 1))
                .chartYScale(domain: 0...yDomainTop(points: points, mean: mean))
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .leading) { _ in
                        AxisGridLine().foregroundStyle(Brand.separator)
                        AxisValueLabel()
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(Brand.textHighlight.opacity(0.55))
                    }
                }
                .frame(height: 200)
                .animation(motion ? .easeOut(duration: 0.30) : nil, value: points.count)
            } else {
                emptyState
            }
        }
        .padding(Brand.Spacing.s4)
        .background {
            LiftedSurface(
                lifted: isRunning,
                cornerRadius: Brand.Radii.l,
                animation: liftAnimation,
                delay: liftDelay
            )
        }
    }

    @ViewBuilder
    private func header(rolling: RollingMetrics?) -> some View {
        HStack(spacing: 8) {
            Text(tr("DECODE TPS · LAST 5 MIN"))
                .font(.system(size: 11, weight: .heavy, design: .monospaced))
                .tracking(3)
                .foregroundStyle(Brand.textHighlight)
            Spacer()
            if let r = rolling {
                Text(tr("%lld samples", r.count))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(Brand.textHighlight.opacity(0.55))
            }
        }
    }

    /// Plotted decode series. Prefers the denser `liveHistory`
    /// (per-tick samples) over the sparse per-request `history` so the
    /// curve reads as a real trace, and falls back when one is empty.
    /// The chart plots by sample *index*, not the raw `t` timestamp —
    /// the old `t`-keyed plot collapsed every point onto one x (and
    /// rendered as a single vertical bar) whenever the daemon emitted
    /// identical or zeroed timestamps. Index plotting always spans the
    /// full width.
    private func seriesPoints(_ rolling: RollingMetrics) -> [Double] {
        let live = rolling.liveHistory.map(\.tokS).filter { $0.isFinite && $0 >= 0 }
        let hist = rolling.history.map(\.tokS).filter { $0.isFinite && $0 >= 0 }
        let points = live.count > hist.count ? live : hist
        guard points.count > Self.maxRenderedPoints else { return points }
        return downsample(points, maxCount: Self.maxRenderedPoints)
    }

    private func downsample(_ points: [Double], maxCount: Int) -> [Double] {
        guard maxCount > 1, points.count > maxCount else { return points }
        let stride = Double(points.count - 1) / Double(maxCount - 1)
        return (0..<maxCount).map { idx in
            points[min(points.count - 1, Int((Double(idx) * stride).rounded()))]
        }
    }

    /// Top of the Y axis — headroom above the peak (and the average
    /// rule) with a sane floor so a near-flat series still reads.
    private func yDomainTop(points: [Double], mean: Double?) -> Double {
        let top = max(points.max() ?? 0, mean ?? 0)
        return max(10, top * 1.15)
    }

    @ViewBuilder
    private var emptyState: some View {
        Text(tr("Your decode speed history will show here."))
            .font(.callout)
            .foregroundStyle(Brand.textHighlight.opacity(0.6))
            .frame(maxWidth: .infinity, minHeight: 160, alignment: .leading)
    }
}
