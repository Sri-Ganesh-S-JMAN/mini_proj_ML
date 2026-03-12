import pathlib
import sys

# THE FIX: This must happen BEFORE importing fastai or torch
# We are forcing the system to treat WindowsPath as PosixPath globally
class SimpleWindowsPath(pathlib.PosixPath):
    def __init__(self, *args, **kwargs):
        pass

pathlib.WindowsPath = SimpleWindowsPath

# Now import the rest
import gradio as gr
from fastai.vision.all import load_learner

# Load the model
# The previous error 'res' was just a symptom of this line failing
learn = load_learner("banana_disease_model.pkl")

# Your training labels (Ensure this order matches your training folders)
CLASSES = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]

def predict(img):
    # fastai's predict returns (label, index, probabilities)
    _, _, probs = learn.predict(img)
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

# Simplified Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Banana Leaf Disease Classifier"
)

if __name__ == "__main__":
    demo.launch()