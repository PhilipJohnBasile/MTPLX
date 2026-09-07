import SwiftUI

// MARK: - Direction-aware offset
//
// SwiftUI mirrors stacks, alignments and edge padding for right-to-left
// languages, but `.offset(x:)` is a raw translation and keeps pointing
// right. Small hover shifts and badge nudges that mean "towards the
// trailing edge" go through this modifier so Arabic gets the mirror
// image instead of an arrow that slides away from its text.

private struct TrailingOffset: ViewModifier {
    @Environment(\.layoutDirection) private var layoutDirection
    let x: CGFloat
    let y: CGFloat

    func body(content: Content) -> some View {
        content.offset(x: layoutDirection == .rightToLeft ? -x : x, y: y)
    }
}

extension View {
    /// Offsets by `x` points towards the trailing edge (right in LTR,
    /// left in RTL) and by `y` points down.
    func offset(towardsTrailing x: CGFloat, y: CGFloat = 0) -> some View {
        modifier(TrailingOffset(x: x, y: y))
    }
}
