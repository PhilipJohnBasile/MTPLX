# Historical receipt provenance

`research_paroquant_q35_fixed_work_v4_producer_1578f6f8.py` is an exact,
byte-hashed copy of the source recorded by the 16 v4 fixed-token performance
receipts. It is retained for reproducibility only.

That historical producer includes an interpretation error: fixed token IDs and
forward count do not fix MoE routing, because quantized artifacts can produce
different hidden states. The later benchmark, raw-receipt annotations, summary,
and research report supersede that claim. No timing or model output was rerun
or altered during the correction.
