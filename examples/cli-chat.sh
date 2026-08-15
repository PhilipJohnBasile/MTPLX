#!/usr/bin/env bash
set -euo pipefail

mtplx chat --model "${MTPLX_MODEL:-Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed}"
