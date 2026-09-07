from pathlib import Path
import sys
root=Path(sys.argv[1])
p=root/'apps/MTPLXApp/Sources/MTPLXAppCore/Services/MetricsStreamClient.swift'
s=p.read_text()
old='''            } catch {
                if Task.isCancelled { return }
'''
new='''            } catch {
                print("FIXTURE_STREAM_ERROR", String(reflecting: error))
                if Task.isCancelled { return }
'''
assert s.count(old)==1
p.write_text(s.replace(old,new))
p=root/'apps/MTPLXApp/Tests/MTPLXAppCoreTests/MTPLXAppCoreTests.swift'
s=p.read_text()
# Capture only already-published state; do not add an awaited operation to a timing test.
s=s.replace('        await backend.refreshLogs()\n        print("START_DIAGNOSTIC",','        print("START_DIAGNOSTIC",')
p.write_text(s)
