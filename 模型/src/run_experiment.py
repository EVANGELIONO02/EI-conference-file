from __future__ import annotations
import argparse, json
from pathlib import Path
from train_and_evaluate import run

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config.json"); p.add_argument("--quick", action="store_true"); args = p.parse_args(); path = Path(args.config)
    if not path.is_absolute(): path = Path(__file__).resolve().parents[1] / path
    result = run(path, quick=args.quick)
    expected = [path.parent / "results" / name for name in ("reconstruction_comparison.csv", "reconstruction_grouped_comparison.csv", "forecast_support_comparison.csv", "ablation_comparison.csv")]
    print(json.dumps({"device": result["device"], "completed": sum(x["status"] == "completed" for x in result["runs"]), "total": len(result["runs"]), "results_exported": all(x.exists() and x.stat().st_size > 0 for x in expected), "result_directory": str(path.parent / "results")}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
