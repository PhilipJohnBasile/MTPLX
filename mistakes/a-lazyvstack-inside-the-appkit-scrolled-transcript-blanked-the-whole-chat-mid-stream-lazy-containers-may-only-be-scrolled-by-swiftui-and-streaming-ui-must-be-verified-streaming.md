# A LazyVStack inside the AppKit-scrolled transcript blanked the whole chat mid-stream — lazy containers may only be scrolled by SwiftUI, and streaming UI must be verified streaming

**Symptom (2026-08-18):** during generation the transcript flickered
after every few line additions and intermittently went COMPLETELY
BLANK (founder screenshot: empty black scroll area, live tok/s chip
still ticking). Shipped in the 2.8.3 fix branch despite a full A/B
validation round.

**Cause:** the 2.8.3 perf work converted the transcript VStack to a
LazyVStack (to bound a window min-size walk). A lazy stack only
ESTIMATES off-screen subview heights and coordinates content offset
with SwiftUI's own scroll bookkeeping — but this transcript is
scrolled by AppKit (`ChatConversationScrollDriver` writes the clip
view origin directly, synchronously inside `frameDidChange` during
layout). Two writers on one offset, one of them computing targets from
the other's unstable estimates: SwiftUI's realization window diverges
from the real visible rect and culls rows that are on screen. Apple
says all three parts outright (WWDC26 session 321: estimated heights;
offset compensation; "avoid using the absolute content size or content
offset with lazy stacks"); Apple forum 741406 reports the identical
blank-view failure and the confirmed workaround is VStack.

**Why the A/B missed it:** validation measured CPU, walk samples, and
probe latency — and the table fix was verified on a STATIC restored
transcript. Nothing watched pixels during a live stream.

**Fixes/rules:**
1. Transcript + streaming-card stacks are plain VStacks (row count is
   already bounded by the heavy-transcript tail slicing; every row is
   Equatable-cached). Comments in `ChatConversationView` and
   `StreamingAssistantMarkdownView` carry the ban.
2. NEVER put a lazy container inside a scroll surface whose offset any
   AppKit code writes. Lazy is fine where SwiftUI owns the scrolling
   (sidebar, logs sheet).
3. A rendering claim about STREAMING requires evidence FROM a live
   stream (stall census + screenshots/recording during generation), not
   a settled or restored transcript.
4. Related follow-ups: end-of-turn settle hitch (~8 stalls of
   88-154 ms as a 10k-token turn folds into the persisted bubble), and
   eager rows make full AX tree walks O(transcript) (20 s timeouts on
   an 11k-token turn) — both collapse under the planned NSTextView
   transcript virtualization (2.8.4).
