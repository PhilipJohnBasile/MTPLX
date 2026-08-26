# Current main now conflicts in model-compatibility.md as well. The materializer
# already rewrites the first stop to expect both Swift paths plus hf_loader;
# extend that expectation and resolve docs by keeping current main then inserting
# the DeepSeek external-AR contract before the next heading.
old_expected = (
    "  expected=$'apps/MTPLXApp/Sources/MTPLXAppCore/Services/DaemonSupervisor.swift\\n"
    "apps/MTPLXApp/Sources/MTPLXAppCore/Stores/MTPLXBackendStore.swift\\n"
    "mtplx/hf_loader.py'\n"
    '  test "$conflicts" = "$expected"'
)
new_expected = (
    "  expected=$'apps/MTPLXApp/Sources/MTPLXAppCore/Services/DaemonSupervisor.swift\\n"
    "apps/MTPLXApp/Sources/MTPLXAppCore/Stores/MTPLXBackendStore.swift\\n"
    "docs/model-compatibility.md\\n"
    "mtplx/hf_loader.py'\n"
    '  test "$conflicts" = "$expected"'
)
if rebase.count(old_expected) != 1:
    raise SystemExit("current-main conflict expectation anchor changed")
rebase = rebase.replace(old_expected, new_expected, 1)

old_continue = (
    "  git add mtplx/hf_loader.py\n"
    "  # The two Swift paths remain unmerged in this same rebase stop.\n"
    "  rc=1"
)
new_continue = (
    "  git add mtplx/hf_loader.py\n"
    "  git checkout --ours docs/model-compatibility.md\n"
    "  python - <<'PYDOC'\n"
    "from pathlib import Path\n"
    "path = Path('docs/model-compatibility.md')\n"
    "text = path.read_text(encoding='utf-8')\n"
    "paragraph = '''The external AR route is reserved for\n"
    "`philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly` at immutable\n"
    "revision `ac33e4f3ca3546e6cec104558d42161e15814e33`. Admission requires its\n"
    "DeepSeek V4 target-only configuration, all 44 exact weight shards, required\n"
    "sidecars, and the closed safetensors index; cached content is hash-checked and\n"
    "a same-size corrupt file is repaired through an atomic re-download. MTPLX then\n"
    "executes the separately installed `mlx-serve` binary. The required zero\n"
    "`num_nextn_predict_layers` / zero `dspark_block_size` contract means an MTP or\n"
    "DSpark artifact is not silently routed here. The external runtime's memory\n"
    "preflight remains enabled. Its streaming and throughput are unapproved.\n\n'''\n"
    "anchor = '## Embedded MTP heads and third-party loaders (#306)\\n'\n"
    "if paragraph.strip() not in text:\n"
    "    if text.count(anchor) != 1:\n"
    "        raise SystemExit('model-compatibility insertion anchor changed')\n"
    "    text = text.replace(anchor, paragraph + anchor, 1)\n"
    "path.write_text(text, encoding='utf-8')\n"
    "PYDOC\n"
    "  git add docs/model-compatibility.md\n"
    "  # The two Swift paths remain unmerged in this same rebase stop.\n"
    "  rc=1"
)
if rebase.count(old_continue) != 1:
    raise SystemExit("current-main docs continuation anchor changed")
rebase = rebase.replace(old_continue, new_continue, 1)

post_start_begin = (
    '    if (\n'
    '        "private func refreshPostStartState(" in ours\n'
)
post_start_end_marker = '        kinds.add("post_start")\n'
post_start_start = rebase.find(post_start_begin)
if post_start_start < 0:
    raise SystemExit("post-start resolver start anchor changed")
post_start_end = rebase.find(post_start_end_marker, post_start_start)
if post_start_end < 0:
    raise SystemExit("post-start resolver end anchor changed")
post_start_end += len(post_start_end_marker)
post_start_replacement = '''    if (
        "configuration: MTPLXAppConfiguration," in ours
        and "lifecycleEpoch: Int" in ours
        and "recoveryGeneration: Int?" in ours
        and ") async -> Bool" in ours
        and "configuration: MTPLXAppConfiguration" in theirs
        and "daemonBackendKind(for: configuration) == .mtplx" in theirs
        and "external mlx-serve ready" in theirs
        and "startExternalMlxServeHealthWatchdog" in theirs
    ):
        text = ours
        if not text.endswith("\\n"):
            text += "\\n"
        text += (
            "        guard daemonBackendKind(for: configuration) == .mtplx else {\\n"
            "            health = nil\\n"
            "            capabilities = nil\\n"
            "            sessions = nil\\n"
            "            sessionBank = nil\\n"
            "            settings = nil\\n"
            "            pendingLiveSettings = nil\\n"
            "            pendingLiveSettingsModel = nil\\n"
            "            connectionState = .idle\\n"
            "            await supervisor.logs.append(\\n"
            "                \\\"external mlx-serve ready; MTPLX live controls and metrics are unavailable\\\",\\n"
            "                stream: .system\\n"
            "            )\\n"
            "            guard daemonSessionIsCurrent(\\n"
            "                lifecycleEpoch: lifecycleEpoch,\\n"
            "                launchID: nil,\\n"
            "                recoveryGeneration: recoveryGeneration\\n"
            "            ) else { return false }\\n"
            "            startExternalMlxServeHealthWatchdog()\\n"
            "            return daemonSessionIsCurrent(\\n"
            "                lifecycleEpoch: lifecycleEpoch,\\n"
            "                launchID: nil,\\n"
            "                recoveryGeneration: recoveryGeneration\\n"
            "            )\\n"
            "        }\\n"
        )
        kinds.add("post_start")
'''
rebase = rebase[:post_start_start] + post_start_replacement + rebase[post_start_end:]
