from pathlib import Path
import sys
p=Path(sys.argv[1])/'apps/MTPLXApp/Sources/MTPLXAppCore/Services/DaemonSupervisor.swift'
s=p.read_text()
old='        let existingHealth = probeHealth ? await initialHealthProbe(healthBaseURL, apiKey) : nil\n'
new='''        // Keep the suspended probe result in a named, explicitly typed scope.
        // A conditional await here triggers task-storage teardown failures on
        // the supported Swift 6.1 toolchain during a cold/cancelled launch.
        let existingHealth: HealthPayload?
        if probeHealth {
            existingHealth = await initialHealthProbe(healthBaseURL, apiKey)
        } else {
            existingHealth = nil
        }
'''
assert s.count(old)==1
p.write_text(s.replace(old,new))
