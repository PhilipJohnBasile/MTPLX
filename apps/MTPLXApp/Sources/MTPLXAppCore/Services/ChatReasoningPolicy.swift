import Foundation

public enum ChatReasoningPolicy {
    public static func enableThinking(
        explicitMode: String?,
        liveMode: String? = nil,
        modelControls: ModelControls? = nil,
        modelFamily: String? = nil,
        fallbackMode: String? = nil
    ) -> Bool? {
        switch normalizedMode(explicitMode) ?? normalizedMode(liveMode) {
        case "on":
            return true
        case "off":
            return false
        case "auto":
            return enableThinkingForDefault(
                modelControls: modelControls,
                modelFamily: modelFamily,
                fallbackMode: fallbackMode
            )
        default:
            return enableThinkingForDefault(
                modelControls: modelControls,
                modelFamily: modelFamily,
                fallbackMode: fallbackMode
            )
        }
    }

    public static func normalizedMode(_ raw: String?) -> String? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "auto", "on", "off":
            return raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        default:
            return nil
        }
    }

    private static func enableThinkingForDefault(
        modelControls: ModelControls?,
        modelFamily: String?,
        fallbackMode: String?
    ) -> Bool? {
        switch normalizedMode(modelControls?.reasoning?.defaultMode)
            ?? normalizedMode(fallbackMode)
            ?? defaultMode(forFamily: modelControls?.modelFamily ?? modelFamily)
        {
        case "on":
            return true
        case "off":
            return false
        default:
            return nil
        }
    }

    private static func defaultMode(forFamily rawFamily: String?) -> String? {
        switch rawFamily?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "qwen3_5", "qwen3_6", "qwen3_8", "qwen", "step", "gemma4":
            return "auto"
        default:
            return nil
        }
    }
}
