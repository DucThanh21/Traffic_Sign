import sys
from pathlib import Path

import streamlit as st
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference.predict import load_checkpoint_model
from src.inference.predict import load_class_label_map
from src.inference.predict import resolve_class_label
from src.inference.predict import get_class_metadata
from src.utils.transforms import get_eval_transforms


def main():
    st.set_page_config(page_title="Nhan dien bien bao giao thong", layout="centered")
    st.title("Traffic Sign Recognition Demo")
    st.write("Tai anh bien bao de du doan lop, mo ta tieng Viet va Top-3 kha nang.")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    config_path = PROJECT_ROOT / "configs" / "train_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoint_path = PROJECT_ROOT / config["paths"]["model_output"]

    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    image_size = checkpoint.get("image_size", config["image_size"])
    transform = get_eval_transforms(image_size=image_size)
    data_root = PROJECT_ROOT / config["paths"]["data_root"]
    class_label_map = load_class_label_map(data_root)

    uploaded_file = st.file_uploader("Chon anh bien bao", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(torch.max(probs).item())
            class_label = resolve_class_label(pred_idx, class_label_map)

            top_k = min(3, probs.shape[1])
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

        class_meta = get_class_metadata(pred_idx)

        top3_rows = []
        for rank, (cls_tensor, prob_tensor) in enumerate(
            zip(top_indices[0], top_probs[0]), start=1
        ):
            class_id = int(cls_tensor.item())
            meta = get_class_metadata(class_id)
            top3_rows.append(
                {
                    "Top": rank,
                    "ClassId": class_id,
                    "Ten bien bao": meta["name_en"],
                    "Mo ta (VI)": meta["description_vi"],
                    "Xac suat": float(prob_tensor.item()),
                }
            )

        image_col, info_col = st.columns([1.05, 1.2], gap="large")

        with image_col:
            st.image(image, caption="Anh da tai len", width="stretch")

        with info_col:
            st.success(f"Bien bao du doan: {class_meta['name_en']}")
            st.caption(f"Mo ta: {class_meta['description_vi']}")
            st.caption(f"Class index: {pred_idx}")
            st.info(f"Do tin cay: {confidence:.4f}")

            st.subheader("Top-3 probabilities")
            for row in top3_rows:
                st.write(
                    f"Top {row['Top']}: ClassId={row['ClassId']} | "
                    f"{row['Ten bien bao']} | {row['Mo ta (VI)']} | "
                    f"Xac suat={row['Xac suat']:.4f}"
                )

            with st.expander("Thong tin nhan tu Meta.csv"):
                st.write(f"Nhan tu metadata: {class_label}")


if __name__ == "__main__":
    main()
