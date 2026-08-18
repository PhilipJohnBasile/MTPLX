# piped build output masked a failed swift build and a stale product got installed and A/B tested as the fix — always check build exit status directly and verify installed binary provenance by a new symbol before any verdict

**Symptom →** `swift build -c release 2>&1 | tail -5` "completed exit 0"
(the pipeline exit is `tail`'s), the build had actually FAILED on a
leftover identifier, `cp .build/release/App` then installed the
previous night's stale product, and a full interactive A/B was run
believing the "fixed" leg was fixed. The legs also differed in
workload (rich settled chat vs empty chat), so the numbers looked like
a win and nearly shipped as verification.

**Cause →** (1) exit status read from a pipeline instead of the build
command; (2) install step trusted the product path without provenance;
(3) A/B legs varied the workload (transcript shape) along with the
binary.

**Fix / rule →**
- Run builds with the exit code captured directly (`swift build …;
  echo $?` or redirect to a log file), never behind a pipe.
- Before ANY verdict on an installed binary, prove provenance with
  something only the new code contains — a new Swift type name greps
  from the release binary (`strings App | grep -c NewTypeName`);
  comments and function names do not survive, type metadata does.
- An A/B leg is invalid unless the workload is pinned: same chat
  shape, same transcript size, same interaction script. Transcript
  size alone flipped 161 stalls to 1 on identical code.
- Same disease as the LaunchServices duplicate-bundle and the
  site-packages shadow roulettes — build-products edition. Verify the
  artifact, not the intention.
