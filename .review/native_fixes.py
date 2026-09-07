"""SDK-compatible fixes without unchecked Sendable or actor suppression."""
from pathlib import Path
import sys
root=Path(sys.argv[1])

def replace(name,old,new):
    p=root/name;s=p.read_text();assert s.count(old)==1,(name,s.count(old));p.write_text(s.replace(old,new))

replace('apps/MTPLXApp/Sources/MTPLXAppCore/Persistence/ChatModels.swift',
    '    public static let versionIdentifier = Schema.Version(1, 0, 0)',
    '''    // Older supported SDKs do not mark Schema.Version Sendable. Construct
    // the immutable value per access instead of sharing non-Sendable storage.
    public static var versionIdentifier: Schema.Version { Schema.Version(1, 0, 0) }''')
replace('apps/MTPLXApp/Sources/MTPLXAppCore/Models/DashboardModels.swift',
    '''        switch values["session_cache_hit"]?.boolValue {
        case true: return .hit
        case false: return .miss
        case nil: return .unknown
        }''',
    '''        switch values["session_cache_hit"]?.boolValue {
        case .some(true): return .hit
        case .some(false): return .miss
        case .none: return .unknown
        }''')
replace('apps/MTPLXApp/Sources/MTPLXAppCore/Onboarding/OnboardingOrchestrator.swift',
    '''                await MainActor.run {
                    self?.handleDownloadEvent(event)
                }''',
    '''                // Hop directly to the owning actor. Do not capture the
                // task's weak reference inside another actor-isolated closure.
                await self?.handleDownloadEvent(event)''')
wordmark='apps/MTPLXApp/Sources/MTPLXAppHost/Views/WordmarkView.swift'
# NSImage is reference-backed and not Sendable in the supported SDK. Every
# cache reader is a SwiftUI View; retain caching with explicit UI-actor ownership.
for declaration in ('private let cachedWordmarkImage:', 'private let cachedWordmarkInkImage:',
                    'func wordmarkNSImage() -> NSImage?', 'func wordmarkNSImage(for scheme: ColorScheme) -> NSImage?'):
    replace(wordmark, declaration, '@MainActor\n'+declaration)
