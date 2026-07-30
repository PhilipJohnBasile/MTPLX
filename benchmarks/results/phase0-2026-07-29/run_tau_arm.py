"""Phase-0 chunk 1: tau measurement for one drafter arm (dflash official MLX backend)."""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mtplx.benchmarks.runners.competitor_baselines import run_dflash_mlx_baseline

TARGETS = {
    "8bit": "/Users/pjb/.mtplx/models/mlx-community--Qwen3.6-27B-8bit",
    "4bit": "/Users/pjb/.mtplx/models/mlx-community--Qwen3.6-27B-4bit",
    "fable4": "/Users/pjb/.mtplx/models/Fable-711-4bit",
}
SUITE = str(Path(__file__).resolve().parents[3] / "mtplx/benchmarks/prompts/calibration_coding.jsonl")
DFLASH_SRC = str(Path(__file__).resolve().parents[3] / "tools/refs/dflash")

def main():
    draft = sys.argv[1] if len(sys.argv) > 1 else "z-lab/Qwen3.6-27B-DFlash"
    temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    block = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tag = sys.argv[4] if len(sys.argv) > 4 else "arm"
    target = TARGETS[sys.argv[5]] if len(sys.argv) > 5 else TARGETS["8bit"]
    t0 = time.time()
    r = run_dflash_mlx_baseline(
        target, draft, SUITE,
        dflash_source=DFLASH_SRC,
        temperature=temp, top_p=0.95 if temp > 0 else 1.0, top_k=20 if temp > 0 else 0,
        max_tokens=256, block_size=(block or None), seed=0,
    )
    r["wall_s"] = round(time.time() - t0, 1)
    out = Path(__file__).parent / f"tau_{tag}.json"
    out.write_text(json.dumps(r, indent=1))
    s = r.get("summary", {})
    rows = r.get("rows", [])
    taus = [x for row in rows for x in row.get("acceptance_lengths", [])]
    mean_tau = sum(taus) / len(taus) if taus else None
    errors = [row.get("error") for row in rows if row.get("error")]
    print(json.dumps({
        "tag": tag, "target": target, "draft": draft, "temp": temp, "block": block or "default",
        "mean_tau_all_cycles": round(mean_tau, 3) if mean_tau else None,
        "summary": s, "errors": errors[:3], "error_count": len(errors),
        "wall_s": r["wall_s"], "out": str(out),
    }, indent=1))

if __name__ == "__main__":
    main()
