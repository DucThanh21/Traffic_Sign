import sys
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference.predict import get_class_metadata
from src.inference.predict import load_checkpoint_model
from src.inference.predict import predict_image


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_frame(image: Image.Image, text_lines: list[str], width: int = 960, height: int = 540) -> Image.Image:
    canvas = Image.new("RGB", (width, height), color=(16, 20, 32))

    image_panel_w = int(width * 0.46)
    image_panel_h = int(height * 0.84)
    pad = 24

    resized = image.resize((image_panel_w - 2 * pad, image_panel_h - 2 * pad), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (image_panel_w, image_panel_h), color=(30, 36, 56))
    panel.paste(resized, (pad, pad))

    canvas.paste(panel, (24, (height - image_panel_h) // 2))

    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

    x0 = image_panel_w + 48
    y = 62

    draw.text((x0, y), "Traffic Sign Recognition Demo", fill=(239, 244, 255), font=font_title)
    y += 42

    for line in text_lines:
        draw.text((x0, y), line, fill=(205, 214, 238), font=font_body)
        y += 30

    return canvas


def main():
    config_path = PROJECT_ROOT / "configs" / "train_config.yaml"
    config = load_config(config_path)

    checkpoint_path = PROJECT_ROOT / config["paths"]["model_output"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    meta_dir = PROJECT_ROOT / "data" / "Meta"
    image_paths = sorted(meta_dir.glob("*.png"))[:8]
    if not image_paths:
        raise FileNotFoundError("No sample images found in data/Meta")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    image_size = int(checkpoint.get("image_size", config["image_size"]))

    frames = []
    for image_path in image_paths:
        pred_idx, confidence = predict_image(model, image_path, image_size, device)
        meta = get_class_metadata(pred_idx)

        image = Image.open(image_path).convert("RGB")
        lines = [
            f"Image: {image_path.name}",
            f"Predicted class: {meta['name_en']}",
            f"ClassId: {pred_idx}",
            f"Description (VI): {meta['description_vi']}",
            f"Confidence: {confidence:.4f}",
        ]
        frames.append(build_frame(image, lines))

    assets_dir = PROJECT_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    gif_path = assets_dir / "demo.gif"

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1200,
        loop=0,
        optimize=False,
    )

    print(f"GIF created: {gif_path}")


if __name__ == "__main__":
    main()
