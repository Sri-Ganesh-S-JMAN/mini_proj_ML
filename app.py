import pathlib
import os
import gradio as gr
from fastai.vision.all import load_learner

# --- THE HARD FIX FOR PYTHON 3.13 ---
# We define a dummy class that looks like WindowsPath but acts like PosixPath
class WindowsPath(pathlib.PosixPath):
    pass

# We force the global pathlib to accept our dummy class
pathlib.WindowsPath = WindowsPath

# 1. LOAD THE MODEL
# Since you're on 3.13, we also ensure the model file exists to avoid a silent fail
model_file = "banana_disease_model.pkl"
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Missing {model_file} in the current directory!")

learn = load_learner(model_file)

# 2. DEFINE CLASSES (Numeric mapping)
CLASSES = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]

def predict(img):
    # Get prediction (labels are likely 0, 1, 2, 3)
    _, _, probs = learn.predict(img)
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

# 3. GRADIO INTERFACE
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Banana Leaf Disease Classifier",
    description="Identifying Cordana, Healthy, Pestalotiopsis, or Sigatoka."
)

if __name__ == "__main__":
    demo.launch()