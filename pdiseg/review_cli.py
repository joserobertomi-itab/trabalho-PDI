from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pdiseg-review",
        description="Web UI to browse source images, overlays and crops.",
    )
    parser.add_argument(
        "--dataset",
        default="data/Train_and_Validation",
        help="Root folder with <Class>/ source images.",
    )
    parser.add_argument(
        "--calibration",
        default="calibration",
        help="Calibration bundle containing boxes.json and stats.csv.",
    )
    parser.add_argument(
        "--result",
        default="result",
        help="Optional segmentation output with *_segmentada_N.png crops.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port.")
    args = parser.parse_args(argv)

    import uvicorn

    from pdiseg.review.model import load_bundle
    from pdiseg.review.server import create_app

    bundle = load_bundle(args.dataset, args.calibration, args.result)
    app = create_app(bundle)
    print(
        f"Review viewer at http://{args.host}:{args.port}/ "
        f"(dataset={args.dataset}, calibration={args.calibration}, result={args.result})"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
