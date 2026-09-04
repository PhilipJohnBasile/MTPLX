import SwiftUI
import MTPLXAppCore

// MARK: - GaugeMode
//
// The hero gauge is *one* surface that morphs between five states:
//
// - `.tps`       — decode rate, headline TPS on a 0-100 log arc.
// - `.prefill`   — prefill rate, headline TPS on a 0-1500 log arc, with
//                  token-progress + ETA shown in the caption.
// - `.dim`       — daemon stopped or pre-warmup, dim arc, power glyph.
// - `.loading`   — daemon mid-startup, spinning snake arc.
// - `.degraded`  — daemon degraded, danger-red tint.
//
// Modeled as a single enum so SwiftUI's `.animation(_, value: mode)` does
// all the cross-fade interpolation for us.
//
// Arc geometry: 270° track, piecewise-log value mapping
// (`log10(1 + value) / log10(1 + axisMax)`) so low TPS values move the
// needle meaningfully while still leaving room for the headline number.
// Tick stops are baked into each mode to read like a speedtest.com dial.

enum GaugeMode: Equatable {
    case tps(decode: Double?, max: Double)
    case prefill(progress: Double, livePrefillTPS: Double?, eta: TimeInterval?, requestId: String?)
    case dim
    case loading(phase: LoadingPhase)
    case degraded

    // MARK: Axis

    /// Upper bound of the speedometer scale. `.tps` is clamped at 100
    /// tok/s (the project decode target); `.prefill` at 1500 tok/s
    /// (above the historical M5 Max ceiling). Other modes have no axis.
    var axisMax: Double {
        switch self {
        case .tps: return 100
        case .prefill: return 1500
        case .dim, .loading, .degraded: return 1
        }
    }

    /// Speedtest-style tick labels rendered along the inside of the
    /// arc. The fill needle and these labels share the same log mapping
    /// so they always line up exactly.
    var tickStops: [Double] {
        switch self {
        case .tps:
            return [0, 5, 10, 25, 50, 75, 100]
        case .prefill:
            return [0, 10, 50, 100, 250, 500, 1000, 1500]
        case .dim, .loading, .degraded:
            return []
        }
    }

    // MARK: Geometry

    /// The current speedometer value (the number under the needle), if
    /// known. `nil` while the daemon is warming up or the value source
    /// has been sanity-rejected.
    var speedValue: Double? {
        switch self {
        case let .tps(decode, _):
            return decode
        case let .prefill(_, livePrefillTPS, _, _):
            return livePrefillTPS
        case .dim, .loading, .degraded:
            return nil
        }
    }

    /// 0...1 fill amount around the 270° track using the mode's own
    /// tick stops. Tick labels are laid out at equal angular intervals
    /// around the arc (so "0 to 5" takes the same angular distance as
    /// "75 to 100") and the needle interpolates piecewise-linear
    /// between adjacent stops. Falls back to prefill token-progress
    /// when the rate value is unknown so the arc still climbs while
    /// the daemon stabilises a sane TPS reading.
    var scaledValue: Double {
        if let speed = speedValue {
            return Self.arcFraction(value: speed, tickStops: tickStops)
        }
        if case let .prefill(progress, _, _, _) = self {
            return min(1, max(0, progress))
        }
        return 0
    }

    /// Piecewise-linear mapping from a value to a 0...1 arc fraction.
    /// Tick labels are evenly spaced around the arc; the needle
    /// linearly interpolates between the two adjacent labels'
    /// fractions based on where the value sits inside their value
    /// range. This is how mechanical speedometers with non-uniform
    /// numeric gradations work (e.g. 0, 20, 40, 80, 160 km/h is
    /// non-linear but every label is the same angular distance from
    /// its neighbours).
    static func arcFraction(value: Double, tickStops: [Double]) -> Double {
        guard tickStops.count >= 2 else { return value > 0 ? 1 : 0 }
        let topStop = tickStops.last ?? 0
        let clamped = Swift.max(0, Swift.min(value, topStop))
        let segments = Double(tickStops.count - 1)
        for idx in 0..<(tickStops.count - 1) {
            let lower = tickStops[idx]
            let upper = tickStops[idx + 1]
            if clamped <= upper {
                let segmentLen = upper - lower
                let segmentProgress = segmentLen > 0
                    ? (clamped - lower) / segmentLen
                    : 0
                return (Double(idx) + segmentProgress) / segments
            }
        }
        return 1
    }

    /// Tick label position around the arc (0 = start, 1 = end). Equal
    /// angular spacing regardless of value.
    static func tickFraction(index: Int, tickStops: [Double]) -> Double {
        guard tickStops.count >= 2 else { return 0 }
        return Double(index) / Double(tickStops.count - 1)
    }

    // MARK: Labels

    /// Big center number. Running modes never emit "—" — the gauge is
    /// a speedometer and should always read as a numeric value
    /// (mirrors the web dashboard which uses `Math.max(0, liveTokS ?? 0)`
    /// + `toFixed(1)`). Inactive modes (dim / loading / degraded) do not
    /// render this label at all so the placeholder there is purely
    /// defensive.
    var centerLabel: String {
        switch self {
        case let .tps(decode, _):
            return Format.tps(decode ?? 0)
        case let .prefill(progress, livePrefillTPS, _, _):
            if let livePrefillTPS {
                return Format.tps(livePrefillTPS)
            }
            return "\(Int((progress * 100).rounded()))%"
        case .dim, .loading, .degraded:
            return "—"
        }
    }

    /// Subtitle text under the big number. Empty for `.dim` and
    /// `.loading` — the visual (power glyph / spinning arc) is
    /// self-explanatory and the text reads as clutter at hero scale.
    var subtitle: String {
        switch self {
        case .tps: return tr("TPS")
        case let .prefill(_, livePrefillTPS, _, _):
            return livePrefillTPS == nil ? tr("PREFILL") : tr("PREFILL TPS")
        case .dim: return ""
        case .loading: return ""
        case .degraded: return tr("DEGRADED")
        }
    }

    /// Caption — tiny line under the subtitle. Holds the prefill ETA
    /// when applicable. `.dim` deliberately has no caption (the power
    /// glyph is self-explanatory).
    var caption: String? {
        switch self {
        case .tps: return "decode"
        case let .prefill(progress, _, eta, _):
            let percent = "\(Int((progress * 100).rounded()))%"
            if let eta { return tr("ETA %@ · %@", Format.duration(eta), percent) }
            return tr("prefilling · %@", percent)
        case .dim: return nil
        case .loading: return nil
        case .degraded: return tr("startup failed")
        }
    }

    var preservesCaptionCase: Bool {
        if case .prefill = self { return true }
        return false
    }

    // MARK: Tint

    /// Whether the arc should render warm-chrome instead of cool chrome.
    var isWarm: Bool {
        if case .prefill = self { return true }
        return false
    }

    /// True when the daemon is unreachable; suppresses animations.
    var isDim: Bool {
        if case .dim = self { return true }
        return false
    }

    /// True for the degraded state; the arc renders dimmed with a red tint.
    var isDegraded: Bool {
        if case .degraded = self { return true }
        return false
    }

    /// True while the daemon is mid-startup; gauge shows a spinner in
    /// the center and keeps the circle full-360°.
    var isLoading: Bool {
        if case .loading = self { return true }
        return false
    }

    /// Whether the outer track should be a complete 360° circle (power-
    /// button + loading look) or the 270° speedometer arc.
    var wantsFullCircle: Bool {
        switch self {
        case .dim, .loading: return true
        case .tps, .prefill, .degraded: return false
        }
    }

    /// Whether arc tick labels should be drawn for this mode.
    var showsTicks: Bool {
        switch self {
        case .tps, .prefill: return true
        case .dim, .loading, .degraded: return false
        }
    }
}

// MARK: - LoadingPhase

/// The intermediate daemon states between "stopped" and "running" map
/// to this enum so the gauge spinner can label itself appropriately.
public enum LoadingPhase: String, Equatable, Sendable {
    case starting
    case waitingForServer
    case rampingFans
    case warming
    case stopping

    public var label: String {
        switch self {
        case .starting: return tr("STARTING")
        case .waitingForServer: return tr("WAITING FOR MODEL")
        case .rampingFans: return tr("RAMPING FANS")
        case .warming: return tr("LOADING MODEL")
        case .stopping: return tr("STOPPING")
        }
    }
}
