import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(cmd):
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="configs/train_config.yaml")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--config", default="configs/train_config.yaml")
    eval_parser.add_argument("--checkpoint", default=None)

    pred_parser = subparsers.add_parser("predict")
    pred_parser.add_argument("--config", default="configs/train_config.yaml")
    pred_parser.add_argument("--image", required=True)
    pred_parser.add_argument("--checkpoint", default=None)

    subparsers.add_parser("app")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        run_command([sys.executable, "src/training/train.py", "--config", args.config])
    elif args.command == "evaluate":
        cmd = [sys.executable, "src/evaluation/evaluate.py", "--config", args.config]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        run_command(cmd)
    elif args.command == "predict":
        cmd = [
            sys.executable,
            "src/inference/predict.py",
            "--config",
            args.config,
            "--image",
            args.image,
        ]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        run_command(cmd)
    elif args.command == "app":
        run_command([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])


if __name__ == "__main__":
    main()
