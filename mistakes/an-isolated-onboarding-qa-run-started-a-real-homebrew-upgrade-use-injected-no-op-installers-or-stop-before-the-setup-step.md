# An isolated onboarding QA run started a real Homebrew upgrade — use injected no-op installers or stop before the setup step

**Symptom:** A uniquely bundled app with scratch settings and chat-store paths
still started `brew upgrade youssofal/mtplx/mtplx` when UI QA advanced from
model selection into runtime setup.

**Cause:** Isolating app persistence does not isolate `RuntimeSetupService`.
That onboarding step intentionally inspects and updates the user's real
terminal CLI, fan-control helper, and app runtime.

**Fix / rule:** Never enter runtime setup in an app QA run unless every
installer/updater is explicitly injected or disabled. For localization-only
QA, stop at model selection. After any accidental entry, terminate the full
subprocess tree and verify the active Homebrew keg, `opt` link, and both CLI
links are exactly restored before continuing.
