# Adapt the guarded PR254 rebase resolver to current main.
# The feature and current main now overlap in richer pull provenance and one
# documentation section. Merge the semantics rather than choosing a side.

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
                + ours_text + "\\nTHEIRS:\\n" + theirs_text
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
                + ours_text + "\\nTHEIRS:\\n" + theirs_text
            )
        output.extend([
            "    # Preserve current-main provenance fields while enforcing the\\n",
            "    # canonical identity of pinned model artifacts.\\n",
            "    canonical_repo_id, canonical_revision = pinned\\n",
            "    return (\\n",
            "        payload.get(\\\"repo_id\\\") == canonical_repo_id\\n",
            "        and payload.get(\\\"revision\\\") == canonical_revision\\n",
            "        and revision == canonical_revision\\n",
            "    )\\n",
        ])
    elif count == 3:
        if (
            "payload: dict[str, Any]" not in ours_text
            or "resolved_sha" not in ours_text
            or "pinned = _pinned_source_identity(repo_id)" not in theirs_text
        ):
            raise SystemExit(
                "unexpected source-marker write conflict\\nOURS:\\n"
                + ours_text + "\\nTHEIRS:\\n" + theirs_text
            )
        output.extend(theirs)
        output.extend(ours)
    elif count == 4:
        if (
            "if target.exists():" not in ours_text
            or "target.unlink()" not in ours_text
            or "if force and partial.exists():" not in theirs_text
        ):
            raise SystemExit(
                "unexpected repair conflict\\nOURS:\\n"
                + ours_text + "\\nTHEIRS:\\n" + theirs_text
            )
        output.extend([
            "    if force:\\n",
            "        # Retain a known-bad target until the replacement is complete\\n",
            "        # and hashed; a failed repair must not discard the only local copy.\\n",
            "        if partial.exists():\\n",
            "            partial.unlink()\\n",
            "    elif target.exists():\\n",
            "        # A size-mismatched final file is stale, not an interrupted\\n",
            "        # partial. Do not append a remote tail to old final content.\\n",
            "        target.unlink()\\n",
        ])
    elif count == 5:
        if (
            "marker = read_source_marker(destination)" not in ours_text
            or "def _fresh_against_remote()" not in ours_text
            or "pinned_integrity_errors" not in theirs_text
            or "repair_paths" not in theirs_text
        ):
            raise SystemExit(
                "unexpected pull setup conflict\\nOURS:\\n"
                + ours_text + "\\nTHEIRS:\\n" + theirs_text
            )
        text = ours_text
        setup_anchor = "    if (\\n        not force_sync\\n"
        setup = (
            "    pinned_integrity_errors = (\\n"
            "        _pinned_artifact_integrity_errors(destination, repo_id)\\n"
            "        if destination.is_dir() and _pinned_source_identity(repo_id) is not None\\n"
            "        else None\\n"
            "    )\\n"
            "    repair_paths = (\\n"
            "        frozenset(\\n"
            "            error.partition(\\\":\\\")[0]\\n"
            "            for error in pinned_integrity_errors\\n"
            "            if error.partition(\\\":\\\")[0]\\n"
            "        )\\n"
            "        if pinned_integrity_errors\\n"
            "        else frozenset()\\n"
            "    )\\n"
            "    pinned_integrity_checked = pinned_integrity_errors is not None\\n"
        )
        if text.count(setup_anchor) != 1:
            raise SystemExit("pull setup if anchor changed")
        text = text.replace(setup_anchor, setup + setup_anchor, 1)
        ready_old = "        and _cached_model_ready_for_repo(destination, repo_id)\\n"
        ready_new = (
            "        and _cached_model_ready_for_repo(\\n"
            "            destination,\\n"
            "            repo_id,\\n"
            "            pinned_integrity_errors=pinned_integrity_errors,\\n"
            "        )\\n"
        )
        if text.count(ready_old) != 1:
            raise SystemExit("cached-model readiness anchor changed")
        text = text.replace(ready_old, ready_new, 1)
        output.append(text)
    elif count == 6:
        if (
            "download_revision" not in ours_text
            or "remote_files" not in ours_text
            or "DEEPSEEK_V4_TARGET_ONLY_REPO_BYTES" not in theirs_text
        ):
            raise SystemExit(
                "unexpected total-bytes conflict\\nOURS:\\n"
                + ours_text + "\\nTHEIRS:\\n" + theirs_text
            )
        text = ours_text
        laguna = (
            "        if repo_id.casefold() == LAGUNA_S_2_1_REPO_ID.casefold():\\n"
            "            total_bytes: int | None = LAGUNA_S_2_1_REPO_BYTES\\n"
        )
        deepseek = (
            "        elif repo_id.casefold() == DEEPSEEK_V4_TARGET_ONLY_REPO_ID.casefold():\\n"
            "            total_bytes = DEEPSEEK_V4_TARGET_ONLY_REPO_BYTES\\n"
        )
        if text.count(laguna) != 1:
            raise SystemExit("total-bytes Laguna anchor changed")
        text = text.replace(laguna, laguna + deepseek, 1)
        output.append(text)
    elif count == 7:
        if (
            "_validate_pinned_laguna_files" not in ours_text
            or "resolved_sha=remote_sha" not in ours_text
            or "_validate_pinned_model_files" not in theirs_text
        ):
            raise SystemExit(
                "unexpected final provenance conflict\\nOURS:\\n"
                + ours_text + "\\nTHEIRS:\\n" + theirs_text
            )
        output.extend([
            "        _validate_pinned_model_files(resolved, repo_id)\\n",
            "        # Preserve current-main provenance for every successful pull;\\n",
            "        # _write_source_marker canonicalizes pinned identities first.\\n",
            "        _write_source_marker(\\n",
            "            resolved,\\n",
            "            repo_id=repo_id,\\n",
            "            revision=revision,\\n",
            "            resolved_sha=remote_sha,\\n",
            "            files=remote_files,\\n",
            "        )\\n",
        ])
    else:
        raise SystemExit(
            f"unexpected additional hf_loader conflict #{count}\\n"
            + "OURS:\\n" + ours_text + "\\nTHEIRS:\\n" + theirs_text
        )
'''
rebase = rebase[:hf_start] + hf_replacement + rebase[hf_end:]

old_count_check = (
    'if count != 2:\n'
    '    raise SystemExit(f"expected two hf_loader conflicts, resolved {count}")'
)
new_count_check = (
    'if count != 8:\n'
    '    raise SystemExit(f"expected eight hf_loader conflicts, resolved {count}")'
)
if rebase.count(old_count_check) != 1:
    raise SystemExit("hf_loader conflict-count anchor changed")
rebase = rebase.replace(old_count_check, new_count_check, 1)

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

store_start = rebase.find("def merge_store(ours: str, theirs: str)")
if store_start < 0:
    raise SystemExit("merge_store anchor changed")
store_return = rebase.find("    return text, kinds\n", store_start)
if store_return < 0:
    raise SystemExit("merge_store return anchor changed")
late_health = '''    if (
        "Never guard this on supervisor.isRunning()" in ours
        and "guard supportsMTPLXLiveControls else { return }" in theirs
        and "guard supervisor.isRunning() else { return }" in theirs
    ):
        text = "        guard supportsMTPLXLiveControls else { return }\\n" + ours
        kinds.add("late_health_recovery")

'''
rebase = rebase[:store_return] + late_health + rebase[store_return:]

expected_store_old = '''expected_store = {
    "restart_runtime",
    "start_runtime",
    "refresh_static",
    "flush_fresh",
    "fan_mode",
    "thermal",
    "post_start",
    "prefill_history",
    "models",
}'''
expected_store_new = '''expected_store = {
    "restart_runtime",
    "start_runtime",
    "refresh_static",
    "flush_fresh",
    "fan_mode",
    "thermal",
    "post_start",
    "prefill_history",
    "models",
    "late_health_recovery",
}

# Current main moved launch parameters into an OwnedLaunch restart recipe.
# Carry backendKind through that recipe so external launches and automatic
# restarts do not silently fall back to MTPLX lifecycle semantics.
supervisor_path = Path("apps/MTPLXApp/Sources/MTPLXAppCore/Services/DaemonSupervisor.swift")
supervisor_text = supervisor_path.read_text(encoding="utf-8")
supervisor_text = replace_once(
    supervisor_text,
    "        let apiKey: String?\\n        let probeHealth: Bool\\n",
    "        let apiKey: String?\\n        let backendKind: DaemonBackendKind\\n        let probeHealth: Bool\\n",
    "OwnedLaunch backend kind field",
)
supervisor_text = replace_once(
    supervisor_text,
    "                apiKey: apiKey,\\n                probeHealth: probeHealth,\\n",
    "                apiKey: apiKey,\\n                backendKind: backendKind,\\n                probeHealth: probeHealth,\\n",
    "OwnedLaunch backend kind construction",
)
supervisor_text = replace_once(
    supervisor_text,
    "        let apiKey = launch.apiKey\\n        let probeHealth = launch.probeHealth\\n",
    "        let apiKey = launch.apiKey\\n        let backendKind = launch.backendKind\\n        let probeHealth = launch.probeHealth\\n",
    "OwnedLaunch backend kind unpacking",
)
supervisor_path.write_text(supervisor_text, encoding="utf-8")'''
if rebase.count(expected_store_old) != 1:
    raise SystemExit("expected_store whitelist anchor changed")
rebase = rebase.replace(expected_store_old, expected_store_new, 1)

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
