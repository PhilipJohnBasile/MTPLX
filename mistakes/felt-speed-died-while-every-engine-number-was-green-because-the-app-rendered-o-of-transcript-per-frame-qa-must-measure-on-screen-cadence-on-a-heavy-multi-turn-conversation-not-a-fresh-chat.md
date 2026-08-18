# Felt speed died while every engine number was green because the app rendered O(transcript) per frame — QA must measure on-screen cadence on a heavy multi-turn conversation, not a fresh chat

**Symptom (2026-08-17, founder field report on 2.8.2/2.8.3-rc):** long
temp-1.0 chats froze seconds then pasted bursts ("freeze and vomit"),
felt ~25 tok/s while the TPS chip said 45-50; settings popover opened
slow and scrolled mushy even AFTER generation; table rows drew on top
of each other. Engine fully exonerated by receipts: gate replay over
the founder's exact 37k tokens = zero holdback engagements, uistream
drain 60 Hz with max 0.92 s gap, live wire probe max 0.51 s gap.

**Cause — the app burned O(entire transcript) per frame, several ways
at once:**
- `.windowResizability(.contentMinSize)` kept NSHostingView `minSize`
  derivation armed → full-window `sizeThatFits` walk of the transcript
  on EVERY constraint invalidation (34% of idle main thread; up to
  62×/s streaming). The WindowSizingTuner written to kill this silently
  never applied (cast `window.contentView` but the hosting view is a
  descendant; one-shot latch blocked re-assert).
- `fenceCount` re-walked every character of every block per rendered
  frame; open-fence lexer re-keyed every interior line per frame;
  `streamingContent` concatenated the whole answer per delta;
  coalescer re-scanned all blocks per 16 ms flush.
- Scroll bookkeeping lived in `@State` (incl. a non-Equatable Task) →
  every revision tick invalidated the whole conversation view.
- The typewriter's >4 KB backlog path whole-drained in ONE frame — the
  literal "vomit" paste that made stalls visible.
- Severity scaled with CONVERSATION LENGTH (min-size walk measures all
  realized turns), which is why the founder's all-day heavy chat
  screamed while a fresh-conversation validation saw zero stalls.

**Fix (2.8.3):** tuner subtree-search + re-assert + `.automatic`
resizability; fence counts stamped on blocks at construction; lex-state
chain cache; has-flag emptiness checks + pending-buffer accessors;
scroll-state box; 256-char/tick reveal ceiling; SSE byte state machine
for metrics; performanceLock as an Environment value; table
`idealWidth` (measure/place mismatch under horizontal ScrollView).

**Rules:**
1. App QA runs on a conversation with ≥4 prior 10k+-token turns — a
   fresh chat hides every O(transcript)-per-frame term.
2. "Delivered cadence" has THREE layers: engine wire (read1 probe),
   app ingest (uistream `drained_bytes`/`gap_ms`), and ON-SCREEN text.
   uistream's `apply_ms` ends at the document store — it CANNOT see
   render cost; a clean uistream file does not clear the app. Measure
   screen-side (AX text growth, or a render-span probe) before calling
   streaming smooth.
3. AX/interaction latency IS a stall meter: `get_window_state` against
   the busy app took 7-20 s in the bad build. Any interactive probe
   that slow = main thread starved, whatever the counters say.
4. Never let SwiftUI derive window extrema from a transcript
   (`sizingOptions` must stay empty on the chat window's hosting view;
   pin `contentMinSize` explicitly).
5. `AsyncLineSequence` SKIPS blank lines — never build SSE framing on
   `.lines` (the blank line is the message boundary; verified
   2026-08-17).
6. Catch-up after any stall must be rate-limited (bounded reveal per
   tick), so hiccups read as fast typing, never a paste.
