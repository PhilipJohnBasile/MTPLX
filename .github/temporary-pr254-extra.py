# Current main added richer source-marker provenance in hf_loader.py. Preserve
# that behavior alongside the DeepSeek pinned-source rules.
hf_start = rebase.find(
    '    elif count == 1:\n'
    '        if (\n'
    '            "if target.exists():" not in ours_text\n'
)
if hf_start < 0:
    raise SystemExit("hf_loader second-conflict resolver anchor changed")
hf_end_marker = '    else:\n        raise SystemExit("unexpected additional hf_loader conflict")\n'
hf_end = rebase.find(hf_end_marker, hf_start)
if hf_end < 0:
    raise SystemExit("hf_loader resolver end anchor changed")
hf_end += len(hf_end_marker)
hf_replacement = '''    elif count == 1:
        if (
            "def read_source_marker(" not in ours_text
            or "def _pinned_source_identity(" not in theirs_text
        ):
            raise SystemExit(
                "unexpected source-marker helper conflict\\nOURS:\\n"
                + ours_text
                + "\\nTHEIRS:\\n"
                + theirs_text
            )
        output.extend(ours)
        if ours and not ours[-1].endswith("\\n"):
            output.append("\\n")
        output.extend(theirs)
    elif count == 2:
        if (
            "Subset compare" not in ours_text
            or "canonical_repo_id" not in theirs_text
            or "canonical_revision" not in theirs_text
        ):
            raise SystemExit(
                "unexpected source-marker comparison conflict\\nOURS:\\n"
                + ours_text
                + "\\nTHEIRS:\\n"
                + theirs_text
            )
        output.extend(
            [
                "    # Preserve current-main provenance fields while enforcing the\\n",
                "    # canonical identity of pinned model artifacts.\\n",
                "    canonical_repo_id, canonical_revision = pinned\\n",
                "    return (\\n",
                "        payload.get(\\\"repo_id\\\") == canonical_repo_id\\n",
                "        and payload.get(\\\"revision\\\") == canonical_revision\\n",
                "        and revision == canonical_revision\\n",
                "    )\\n",
            ]
        )
    elif count == 3:
        if (
            "if target.exists():" not in ours_text
            or "target.unlink()" not in ours_text
            or "if force and partial.exists():" not in theirs_text
        ):
            raise SystemExit(
                "unexpected repair conflict\\nOURS:\\n"
                + ours_text
                + "\\nTHEIRS:\\n"
                + theirs_text
            )
        output.extend(
            [
                "    if force:\\n",
                "        # Retain a known-bad target until the replacement is complete\\n",
                "        # and hashed; a failed repair must not discard the only local copy.\\n",
                "        if partial.exists():\\n",
                "            partial.unlink()\\n",
                "    elif target.exists():\\n",
                "        # A size-mismatched final file is stale, not an interrupted\\n",
                "        # partial. Do not append a remote tail to old final content.\\n",
                "        target.unlink()\\n",
            ]
        )
    else:
        raise SystemExit("unexpected additional hf_loader conflict")
'''
rebase = rebase[:hf_start] + hf_replacement + rebase[hf_end:]
old_count_check = (
    'if count != 2:\n'
    '    raise SystemExit(f"expected two hf_loader conflicts, resolved {count}")'
)
new_count_check = (
    'if count != 4:\n'
    '    raise SystemExit(f"expected four hf_loader conflicts, resolved {count}")'
)
if rebase.count(old_count_check) != 1:
    raise SystemExit("hf_loader conflict-count anchor changed")
rebase = rebase.replace(old_count_check, new_count_check, 1)

# Current main now conflicts in model-compatibility.md as well. Keep current
# documentation and insert the DeepSeek external-AR contract before the next
# heading.
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
