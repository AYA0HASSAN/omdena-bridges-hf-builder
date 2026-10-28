"""Command line:

  python -m bridgebuild discover  --bbox W S E N [--out candidates.csv]
  python -m bridgebuild build     --config configs/phase1_oam_france.yaml [--only oam_osm] [--max-cells N] [--max-hours H] [--retry-failed]
  python -m bridgebuild finalize  --config ...
  python -m bridgebuild stats     --config ... [--push] [--overlays 24]
  python -m bridgebuild to-yolo   --config ... --out /kaggle/working/yolo [--neg-keep 0.5]
  python -m bridgebuild all       --config ...        (build, then finalize and stats)
"""
import argparse
import sys

from . import load_config


def main(argv=None):
    p = argparse.ArgumentParser(prog="bridgebuild")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover"); d.add_argument("--bbox", nargs=4, type=float, required=True)
    d.add_argument("--max-gsd", type=float, default=0.6); d.add_argument("--out", default="candidates.csv")
    for name in ("build", "finalize", "stats", "to-yolo", "all"):
        s = sub.add_parser(name); s.add_argument("--config", required=True)
        s.add_argument("--repo-id"); s.add_argument("--local-dir", help="write to a folder instead of the Hub (dry run)")
        if name in ("build", "all"):
            s.add_argument("--only", nargs="*"); s.add_argument("--max-cells", type=int)
            s.add_argument("--max-hours", type=float); s.add_argument("--retry-failed", action="store_true")
        if name == "stats":
            s.add_argument("--push", action="store_true"); s.add_argument("--overlays", type=int, default=24)
        if name == "to-yolo":
            s.add_argument("--out", required=True); s.add_argument("--neg-keep", type=float, default=1.0)
            s.add_argument("--splits", nargs="*", default=["train", "val", "test"]); s.add_argument("--fmt", default="png")
    a = p.parse_args(argv)

    if a.cmd == "discover":
        from .discover import discover
        df = discover(tuple(a.bbox), max_gsd_m=a.max_gsd, out_csv=a.out)
        print(df.head(20).to_string(index=False)); return
    cfg = load_config(a.config)
    if a.repo_id:
        cfg.setdefault("hub", {})["repo_id"] = a.repo_id; cfg["hub"].pop("local_dir", None)
    if a.local_dir:
        cfg.setdefault("hub", {})["local_dir"] = a.local_dir
    if a.cmd in ("build", "all"):
        from .build import build
        build(cfg, only=a.only, max_cells=a.max_cells, max_hours=a.max_hours, retry_failed=a.retry_failed)
    if a.cmd in ("finalize", "all"):
        from .finalize import finalize
        _, leak = finalize(cfg)
        if not leak["passed"].all():
            print("LEAKAGE TEST FAILED"); sys.exit(2)
    if a.cmd in ("stats", "all"):
        from .stats import stats
        from .stats import overlays
        overlays(cfg, n=getattr(a, "overlays", 24))
        stats(cfg, push=getattr(a, "push", False) or a.cmd == "all")
    if a.cmd == "to-yolo":
        from .to_yolo import to_yolo
        to_yolo(cfg, a.out, splits=tuple(a.splits), neg_keep=a.neg_keep, fmt=a.fmt)


if __name__ == "__main__":
    main()
