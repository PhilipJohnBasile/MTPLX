# Two A/B rounds measured a stale 2.7 binary because LaunchServices picks any registered duplicate bundle id — verify the resolved binary path before trusting any app measurement

**Symptom (2026-08-18):** app fixes that verifiably worked in
direct-binary debug runs "failed" two full A/B validation rounds — the
min-size walk persisted, the table smush rendered pixel-identically
through two different implementations. Impossible results.

**Cause:** `launch_app` (and anything LaunchServices-routed, incl.
`open -b`) resolves `com.youssofal.mtplx` against EVERY registered
copy: 80+ existed — worktree `dist/` builds, `build-artifacts/` QA
archives, `~/.mtplx/app-backups/`, and 57 under `~/.mtplx/releases/`.
Different launches got a 2.7.1 worktree build and a 2.7.0 release
archive. The founder's original report WAS the real /Applications
binary (verified in its sample header), so the diagnosis stood — but
my "after" instances were roulette.

**Fixes/rules:**
1. After EVERY app launch used for measurement or QA:
   `ps -p <pid> -o comm=` and assert the path is the bundle you
   installed, plus `CFBundleVersion`. No verification, no verdict.
2. `lsregister -u` stale copies; `-f` the canonical one. Backup app
   copies keep their bundle id — park them unregistered.
3. Two more ops scars from the same night: `build_and_run.sh` KILLS a
   running MTPLXApp (never rebuild while a driven test instance is
   live — it beheaded a test turn mid-generation and mimicked a
   persistence bug), and the toolbar Start/Stop toggle can double-fire
   under AX press (use the MTPLX menu items for automation).
4. Corollary for "impossible" debug results: when two different code
   changes produce IDENTICAL wrong pixels, stop debugging the code and
   verify WHICH code is running.
5. **The Python twin (2026-08-18):** the app daemon launches
   `python -m mtplx.server.openai` with cwd = the runtime venv's
   site-packages — and a REAL `mtplx/` directory from an old wheel
   install sat there, shadowing the editable checkout. An engine A/B
   round "showed no improvement" because the fixes never loaded. Worse,
   `venv/bin/python -c "import mtplx; print(mtplx.__file__)"` LIED when
   run from the checkout directory (cwd precedes site-packages on
   sys.path). Rules: resolve-check imports from a NEUTRAL cwd
   (`cd /tmp`), and after every daemon restart verify the running code
   by PROBING for a marker only the new code has (e.g. a freshly added
   telemetry field in the request record). No probe, no verdict —
   binary and package alike.
