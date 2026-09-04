import SwiftUI
import MTPLXAppCore

// MARK: - RuntimeMetadataTable
//
// Generic key/value table that renders any `[String: Any]` from
// JSONSerialization. Handles strings, numbers, booleans, nested
// dicts (collapsible), and arrays (one row per element). Used by
// the Brand stage's runtime-metadata preview and the My Models
// detail panel.
//
// Keys are rendered as monospaced caps for the visual rhythm the
// rest of the app uses (mirrors StatTile / MicroHeader). Values
// honour their underlying type so a dict shows as a chevron-
// disclosed sub-table, an array shows as a numbered list, and a
// scalar shows inline.

struct RuntimeMetadataTable: View {
    let json: [String: Any]
    var depth: Int = 0

    init(json: [String: Any], depth: Int = 0) {
        self.json = json
        self.depth = depth
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(sortedKeys, id: \.self) { key in
                RuntimeMetadataRow(key: key, value: json[key], depth: depth)
            }
        }
    }

    /// Stable ordering: known spine keys first, then everything else
    /// alphabetically. Keeps the preview readable across renders.
    private var sortedKeys: [String] {
        let priority = [
            "mtplx_version",
            "arch_id",
            "mtp_depth_max",
            "recommended_profile",
            "verified_on",
            "sampler",
            "mtp_sidecar",
            "base_trunk",
            "artifact_role",
            "exactness_baseline",
            "speed_evidence",
            "forge_provenance"
        ]
        let known = priority.filter { json[$0] != nil }
        let rest = json.keys.filter { !priority.contains($0) }.sorted()
        return known + rest
    }
}

private struct RuntimeMetadataRow: View {
    let key: String
    let value: Any?
    let depth: Int

    @State private var expanded: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .top, spacing: 8) {
                if isCollapsibleValue {
                    Button {
                        withAnimation(.spring(response: 0.28, dampingFraction: 0.86)) {
                            expanded.toggle()
                        }
                    } label: {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                            .rotationEffect(.degrees(expanded ? 90 : 0))
                            .foregroundStyle(Brand.typeTertiary)
                            .frame(width: 12)
                    }
                    .buttonStyle(.plain)
                } else {
                    Color.clear.frame(width: 12)
                }

                Text(key)
                    .font(.system(size: 10, weight: .heavy, design: .monospaced))
                    .tracking(0.8)
                    .foregroundStyle(Brand.typeTertiary)
                    .frame(width: 180, alignment: .leading)

                if !isCollapsibleValue {
                    scalarValue
                } else {
                    Text(collapsibleHint)
                        .font(.caption2)
                        .foregroundStyle(Brand.typeTertiary)
                }

                Spacer(minLength: 0)
            }
            if isCollapsibleValue && expanded {
                expandedValue
                    .padding(.leading, 12 + 8 + 12)
            }
        }
    }

    @ViewBuilder
    private var scalarValue: some View {
        Text(displayValue)
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(Brand.typeHi)
            .lineLimit(3)
            .truncationMode(.tail)
            .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder
    private var expandedValue: some View {
        if let dict = value as? [String: Any] {
            RuntimeMetadataTable(json: dict, depth: depth + 1)
                .padding(.leading, 8)
        } else if let array = value as? [Any] {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(Array(array.enumerated()), id: \.offset) { idx, element in
                    HStack(alignment: .top, spacing: 6) {
                        Text("[\(idx)]")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundStyle(Brand.typeTertiary)
                        if let dict = element as? [String: Any] {
                            RuntimeMetadataTable(json: dict, depth: depth + 1)
                        } else {
                            Text(displayScalar(element))
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(Brand.typeBody)
                        }
                    }
                }
            }
        }
    }

    private var isCollapsibleValue: Bool {
        value is [String: Any] || value is [Any]
    }

    private var collapsibleHint: String {
        if let dict = value as? [String: Any] {
            return dict.count == 1 ? tr("1 field") : tr("%lld fields", dict.count)
        }
        if let array = value as? [Any] {
            return array.count == 1 ? tr("1 item") : tr("%lld items", array.count)
        }
        return ""
    }

    private var displayValue: String {
        guard let value else { return "—" }
        return displayScalar(value)
    }

    private func displayScalar(_ value: Any) -> String {
        if let s = value as? String { return s }
        if let b = value as? Bool { return b ? "true" : "false" }
        if let i = value as? Int { return String(i) }
        if let i64 = value as? Int64 { return String(i64) }
        if let d = value as? Double {
            if d.truncatingRemainder(dividingBy: 1) == 0 {
                return String(Int(d))
            }
            return String(format: "%.4g", d)
        }
        if value is NSNull { return "null" }
        return String(describing: value)
    }
}
