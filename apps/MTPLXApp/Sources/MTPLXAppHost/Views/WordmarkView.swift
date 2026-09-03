import SwiftUI
import AppKit
import MTPLXAppCore

// MARK: - WordmarkView
//
// Jet Chrome wordmark. Single source of truth across the top strip, the
// menubar popover, the About sheet, the chat empty state, and the
// onboarding hero. Renders the bundled PNG (`Resources/Wordmark/Wordmark.png`)
// at any height >= 14pt so the polished steel glyphs ship as the
// designer drew them; below that — and any caller that explicitly
// opts in via `fallbackTracking` — it falls back to live `Text("MTPLX")`
// painted with the `MTPLXChromeText` modifier so the wordmark stays
// vector-crisp at menu-bar-mini sizes where a raster would blur.
//
// The PNG is loaded via the cached `wordmarkNSImage()` lookup below
// rather than `Image("Wordmark", bundle: .module)` so the release app can
// ship a normal signed bundle with assets in `Contents/Resources`.
// The cached lookup probes `Bundle.main`, then scans every loaded
// `Bundle.allBundles` for any `Wordmark.png`, so the raster ships as long
// as the file is bundled anywhere reachable.
// If every probe fails, we fall back to the chrome-text rendering
// rather than rendering nothing.

struct WordmarkView: View {
    var height: CGFloat
    var fallbackTracking: Bool

    init(height: CGFloat, fallbackTracking: Bool = false) {
        self.height = height
        self.fallbackTracking = fallbackTracking
    }

    var body: some View {
        Group {
            if useRaster, let nsImage = wordmarkNSImage() {
                Image(nsImage: nsImage)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .frame(height: height)
            } else {
                Text("MTPLX")
                    .font(.system(size: height * 0.9, weight: .heavy, design: .rounded))
                    .tracking(height * 0.05)
                    .chromeText()
                    .fixedSize()
            }
        }
        .accessibilityLabel("MTPLX")
        .accessibilityAddTraits(.isHeader)
    }

    private var useRaster: Bool {
        !fallbackTracking && height >= 14
    }
}

// MARK: - Wordmark NSImage lookup
//
// Resolves Wordmark.png from any reachable bundle. The result is
// cached so we don't re-scan every paint — once located, subsequent
// calls return the same NSImage instance.

private let cachedWordmarkImage: NSImage? = {
    let resourceName = "Wordmark"
    let resourceExt  = "png"

    // 1) The main bundle (used when the release app bundles the PNG flat
    //    into `Contents/Resources` rather than into a sub-bundle).
    if let url = Bundle.main.url(forResource: resourceName, withExtension: resourceExt),
       let img = NSImage(contentsOf: url) {
        return img
    }

    // 2) Every loaded bundle. Catches development layouts and any other
    //    plausible location before falling back to text.
    for bundle in Bundle.allBundles {
        if let url = bundle.url(forResource: resourceName, withExtension: resourceExt),
           let img = NSImage(contentsOf: url) {
            return img
        }
    }

    // 3) Sibling/resource scan. Walks the executable directory, app
    //    resources, app root, and parent looking for any `Wordmark.png`
    //    before we drop to the text fallback.
    let probeRoots: [URL] = {
        var roots: [URL] = []
        let exec = Bundle.main.executableURL?.deletingLastPathComponent()
        if let exec { roots.append(exec) }
        if let resources = Bundle.main.resourceURL { roots.append(resources) }
        roots.append(Bundle.main.bundleURL)
        roots.append(Bundle.main.bundleURL.deletingLastPathComponent())
        return roots
    }()
    let fm = FileManager.default
    for root in probeRoots {
        guard let entries = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else {
            continue
        }
        for entry in entries where entry.pathExtension == "bundle" {
            let candidate = entry.appendingPathComponent("\(resourceName).\(resourceExt)")
            if let img = NSImage(contentsOf: candidate) {
                return img
            }
        }
        let direct = root.appendingPathComponent("\(resourceName).\(resourceExt)")
        if let img = NSImage(contentsOf: direct) {
            return img
        }
    }

    return nil
}()

func wordmarkNSImage() -> NSImage? { cachedWordmarkImage }

// MARK: - WordmarkSubtitle

struct WordmarkSubtitle: View {
    var dividerWidth: CGFloat = 240

    var body: some View {
        VStack(spacing: 12) {
            Rectangle()
                .fill(Brand.separator)
                .frame(width: dividerWidth, height: 1)
            Text(tr("native MTP · Apple Silicon"))
                .font(BrandFont.subtitle())
                .foregroundStyle(Brand.typeSecondary)
        }
        .accessibilityElement(children: .combine)
    }
}
