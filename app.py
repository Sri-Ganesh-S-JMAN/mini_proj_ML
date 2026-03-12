import gradio as gr
from fastai.vision.all import load_learner
import pathlib

# --- THE FIX: Force WindowsPath to work on Linux ---
import platform
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
# --------------------------------------------------

# Load the model
learn = load_learner("banana_disease_model.pkl")

# Your mapping
CLASSES = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]

def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Banana Leaf Disease Classifier",
    description=f"Identify: {', '.join(CLASSES)}"
)

if __name__ == "__main__":
    demo.launch()