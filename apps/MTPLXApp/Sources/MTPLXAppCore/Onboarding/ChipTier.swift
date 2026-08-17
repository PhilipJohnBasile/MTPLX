import Foundation

// MARK: - ChipTier
//
// Coarse classification of the user's Mac for onboarding decisions:
// which model variant to recommend and which memory ceilings to enforce.
// Deliberately matches the daemon's own M-series
// generation bucket in `mtplx/default_models.py` so the auto-routing
// stays consistent between the daemon's `select_default_model` and
// the Swift app's first-launch picker.
//
// Generation buckets (verified against daemon):
//   - M1, M2 (incl. Pro/Max/Ultra) → `.legacyApple` → FP16 variant
//   - M3, M4, M5 (incl. Pro/Max/Ultra) → `.modernApple` → Q4 variant
//   - Intel x86 → `.intel` (unsupported but not blocked)
//   - anything we can't classify → `.unknown` (treat as modernApple)

public enum ChipTier: String, Codable, Equatable, Sendable {
    /// Apple Silicon M1/M2 families and their Pro/Max/Ultra variants.
    /// These chips' MLX kernels handle FP16 floats more predictably
    /// than the Q4 trunk path on the same model — the daemon's
    /// `select_default_model` routes them to the FP16 artifact.
    case legacyApple = "legacy_apple"

    /// Apple Silicon M3/M4/M5 families and their Pro/Max/Ultra
    /// variants. Default routing target — Q4 trunk + Q4 MTP sidecar
    /// is the proven fast path on these.
    case modernApple = "modern_apple"

    /// Intel Mac. The daemon technically still loads MLX-CPU but
    /// performance is degraded enough that no model is "recommended" —
    /// the picker still lets the user try, but every card shows a
    /// `tightFit` verdict at best.
    case intel = "intel"

    /// We couldn't detect the chip family at all. Treated identically
    /// to `.modernApple` for recommendations but rendered as a more
    /// hedged tier label on the scan screen.
    case unknown
}

// MARK: - DetectedHardware
//
// Snapshot of what we know about the user's Mac at the moment of the
// hardware scan. All fields are nullable except the two that are
// reliably present from `ProcessInfo` even when the daemon CLI is
// missing — chip name (via sysctl) and unified memory bytes. The
// optional GPU / CPU / model-identifier fields are populated only when
// `mtplx hardware inspect --json` succeeds; the sysctl fallback path
// fills the minimum needed for tier classification and feasibility
// gating.

public struct DetectedHardware: Equatable, Sendable {
    public var chipName: String
    public var appleSiliconGeneration: String?
    public var modelIdentifier: String?
    public var unifiedMemoryBytes: Int64
    public var gpuCoreCount: Int?
    public var cpuCoreCount: Int?

    public init(
        chipName: String,
        appleSiliconGeneration: String? = nil,
        modelIdentifier: String? = nil,
        unifiedMemoryBytes: Int64,
        gpuCoreCount: Int? = nil,
        cpuCoreCount: Int? = nil
    ) {
        self.chipName = chipName
        self.appleSiliconGeneration = appleSiliconGeneration
        self.modelIdentifier = modelIdentifier
        self.unifiedMemoryBytes = unifiedMemoryBytes
        self.gpuCoreCount = gpuCoreCount
        self.cpuCoreCount = cpuCoreCount
    }

    /// Bucket the chip into the tier the onboarding model picker uses.
    /// Mirrors the daemon's `_LEGACY_APPLE_FP16_GENERATIONS` /
    /// `_NEWER_APPLE_SPEED_GENERATIONS` sets exactly.
    public var tier: ChipTier {
        if let gen = appleSiliconGeneration?.lowercased() {
            switch gen {
            case "m1", "m2": return .legacyApple
            case "m3", "m4", "m5": return .modernApple
            // `mtplx hardware inspect` reports Intel Macs as the literal
            // generation string "intel" (classify_apple_silicon_generation,
            // mtplx/hardware.py). Without this case they landed in
            // `.unknown` — treated as modernApple — and were recommended
            // Q4 models the tier gating exists to warn about.
            case "intel": return .intel
            default: return .unknown
            }
        }
        let lower = chipName.lowercased()
        if lower.contains("intel") { return .intel }
        if lower.range(of: #"\bm[12]\b"#, options: .regularExpression) != nil {
            return .legacyApple
        }
        if lower.range(of: #"\bm[3-5]\b"#, options: .regularExpression) != nil {
            return .modernApple
        }
        return .unknown
    }

    /// Human-friendly tier label for non-onboarding surfaces that need
    /// a neutral chip bucket. The onboarding hardware reveal shows the
    /// chip and specs directly instead of a slogan.
    public var tierDisplayName: String {
        switch tier {
        case .legacyApple: return "Apple Silicon"
        case .modernApple: return "Apple Silicon"
        case .intel: return "Intel Mac"
        case .unknown: return "Apple Silicon"
        }
    }

    public var unifiedMemoryGiB: Double {
        Double(unifiedMemoryBytes) / 1_073_741_824.0
    }
}
