# hands-off automation cannot see interaction bugs — beforeWaiting runloop observers starve under input storms so streaming UI must be verified with synthesized wheel and mouse events and a rich transcript

**Symptom →** Founder: "when I was watching you debug it looked
flawless… but now I tried the app and there's stuttering all over the
place. My theory is the debugging app and the release app function
differently." Same binary, same engine, same prompt — the only
difference was his hand on the mouse.

**Cause →** CFRunLoop skips the `.beforeWaiting` phase on any
iteration that handled a source; sustained human input (mouse-moved /
wheel events) keeps the loop polling. A `.beforeWaiting` observer used
as a per-turn guard (the sizing-tuner clear) therefore starves exactly
and only under interaction, and Core Animation's own commit (same
phase) coalesces behind the storm — freeze-then-burst invisible to any
hands-off run. Phase-aligned proof: hands-off = 1 stall; wheel 40 s =
70 stalls / 18.7 s frozen; wiggle 30 s = 91 stalls / 26.7 s frozen;
fixed observer mask (`.beforeTimers | .beforeWaiting`) = 1 / 0 / 0.

**Fix / rule →**
- Never rely on `.beforeWaiting` alone for work that must happen every
  runloop turn; add `.beforeTimers` (fires on every iteration, still
  ordered after the previous iteration's render).
- Streaming-UI verification MUST include an interaction pass:
  synthesized wheel + mouse-move storms (Quartz `CGEventPost` at HID
  level) over a transcript with at least one rich settled answer,
  phase-aligned against the UI probe's stall census. Hands-off passes
  prove nothing about interaction regimes.
- Corollary of round two's "streaming UI must be verified streaming":
  verified streaming UNDER INTERACTION, or it isn't verified.
