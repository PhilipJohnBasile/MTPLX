# When the user forbids app launches, stop UI QA immediately and use static build and test checks only

**Symptom:** A localization QA build opened a visible app when the user wanted
the work completed without launching any app.

**Cause:** Live UI validation was treated as part of the implementation flow
without preserving the user's stronger preference against app launches.

**Fix / rule:** On any no-launch instruction, terminate the QA process,
confirm it is gone, end the automation session, and perform only non-launching
source review, lint, build, and test checks for the remainder of the task.
