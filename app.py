import json
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image
from fastai.vision.all import load_learner


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "banana_disease_model.pkl"

METADATA_CANDIDATES = [
    APP_DIR / "model_artifacts" / "model_metadata.json",
    APP_DIR.parent.parent / "model_artifacts" / "model_metadata.json",
]

KNOWN_CLASSES = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]


def format_confidence(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_metadata(candidates: list[Path]) -> dict:
    for metadata_path in candidates:
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as file:
                return json.load(file)
    return {"classes": []}


def select_display_classes(metadata: dict, learner_vocab: list[str]) -> list[str]:
    metadata_classes = [str(c) for c in metadata.get("classes", [])]

    # Prefer explicit metadata from training artifacts.
    if metadata_classes and len(metadata_classes) == len(learner_vocab):
        return metadata_classes

    # If learner vocab is numeric labels (0..N), map to known disease names by index.
    if all(item.isdigit() for item in learner_vocab):
        if len(learner_vocab) == len(KNOWN_CLASSES):
            return KNOWN_CLASSES

    # Otherwise keep learner vocab as-is.
    return learner_vocab


class BananaLeafDiseasePredictor:
    def __init__(self, model_path: Path, metadata: dict):
        self.model_path = model_path
        self.metadata = metadata
        self.learn = load_learner(model_path)

        self.learner_vocab = [str(c) for c in self.learn.dls.vocab]
        self.class_names = select_display_classes(self.metadata, self.learner_vocab)

        print(f"Model loaded: {self.model_path}")
        print(f"Learner vocab: {self.learner_vocab}")
        print(f"Display classes: {self.class_names}")

    def predict(self, image: Image.Image) -> dict:
        if image is None:
            raise ValueError("Please upload an image before running prediction.")
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL.Image.Image object.")

        _, pred_idx, probs = self.learn.predict(image)
        pred_idx_int = int(pred_idx)

        probabilities = {
            self.class_names[i]: float(probs[i])
            for i in range(min(len(self.class_names), len(probs)))
        }
        probabilities = dict(
            sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        )

        return {
            "disease": self.class_names[pred_idx_int],
            "confidence": float(probs[pred_idx_int]),
            "probabilities": probabilities,
        }


def build_predictor() -> BananaLeafDiseasePredictor:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Add banana_disease_model.pkl to the Space root."
        )
    metadata = load_metadata(METADATA_CANDIDATES)
    return BananaLeafDiseasePredictor(MODEL_PATH, metadata)


PREDICTOR = None
STARTUP_ERROR = ""

try:
    PREDICTOR = build_predictor()
except Exception as exc:
    STARTUP_ERROR = str(exc)
    print(f"Startup warning: {STARTUP_ERROR}")


def run_inference(image: Image.Image):
    if PREDICTOR is None:
        raise RuntimeError(
            "Model is not available. Ensure model files are present in the Space. "
            f"Details: {STARTUP_ERROR}"
        )

    result = PREDICTOR.predict(image)
    top_class = result["disease"]
    confidence_text = format_confidence(result["confidence"])
    probs_df = pd.DataFrame(
        {
            "class": list(result["probabilities"].keys()),
            "probability": list(result["probabilities"].values()),
        }
    )
    probs_df["probability_percent"] = (probs_df["probability"] * 100).round(2)
    probs_df = probs_df[["class", "probability_percent"]].reset_index(drop=True)
    return top_class, confidence_text, probs_df


def gradio_predict(image):
    if image is None:
        return "No image provided", "N/A", pd.DataFrame(
            columns=["class", "probability_percent"]
        )

    try:
        return run_inference(image)
    except Exception as exc:
        return f"Error: {exc}", "N/A", pd.DataFrame(
            columns=["class", "probability_percent"]
        )


supported_classes_text = (
    ", ".join(PREDICTOR.class_names)
    if PREDICTOR is not None
    else ", ".join(KNOWN_CLASSES)
)

description = (
    f"Upload one leaf image. Supported classes: {supported_classes_text}."
)

if STARTUP_ERROR:
    description += f"\n\nStartup note: {STARTUP_ERROR}"


demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload Banana Leaf Image"),
    outputs=[
        gr.Textbox(label="Predicted Disease"),
        gr.Textbox(label="Confidence"),
        gr.Dataframe(label="Class Probabilities (%)"),
    ],
    title="Banana Leaf Disease Classifier",
    description=description,
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)