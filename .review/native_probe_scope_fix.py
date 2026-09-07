from pathlib import Path
import sys
p=Path(sys.argv[1])/'apps/MTPLXApp/Sources/MTPLXAppCore/Services/DaemonSupervisor.swift'
s=p.read_text()
old='        let existingHealth = probeHealth ? await initialHealthProbe(healthBaseURL, apiKey) : nil\n'
new='''        // The large payload crosses a small, fixed-size async result boundary.
        // Keep probe completion separate from the launch's later suspension points.
        let initialSnapshot: InitialHealthSnapshot?
        if probeHealth {
            initialSnapshot = await readInitialHealthSnapshot(healthBaseURL, apiKey: apiKey)
        } else {
            initialSnapshot = nil
        }
        let existingHealth = initialSnapshot?.health
'''
assert s.count(old)==1
s=s.replace(old,new)
anchor='    private func startOwned(\n'
helper='''    private final class InitialHealthSnapshot: Sendable {
        let health: HealthPayload?

        init(health: HealthPayload?) {
            self.health = health
        }
    }

    private func readInitialHealthSnapshot(_ baseURL: URL, apiKey: String?) async -> InitialHealthSnapshot {
        let health = await initialHealthProbe(baseURL, apiKey)
        return InitialHealthSnapshot(health: health)
    }

'''
assert s.count(anchor)==1
p.write_text(s.replace(anchor,helper+anchor))
