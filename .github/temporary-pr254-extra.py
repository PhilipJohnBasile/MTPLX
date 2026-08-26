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

# retrigger current-main materialization on 2026-08-26
