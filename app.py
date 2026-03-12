import pathlib
import gradio as gr
from fastai.vision.all import load_learner

# This is the most robust fix for Python 3.13 + Linux + Windows-saved models
posix_backup = pathlib.PosixPath
try:
    pathlib.PosixPath = pathlib.WindowsPath
    # Load the model while pretending we are on Windows (or vice versa)
    # This usually resolves the "cannot instantiate WindowsPath" issue
    learn = load_learner("banana_disease_model.pkl")
finally:
    pathlib.PosixPath = posix_backup

CLASSES = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]

def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Banana Leaf Disease Classifier"
)

if __name__ == "__main__":
    demo.launch()