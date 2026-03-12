import pathlib
import platform
import gradio as gr
from fastai.vision.all import load_learner

# --- CROSS-PLATFORM FIX ---
# This forces Linux to understand Windows-saved paths
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# Load the model directly
# If the file is in the same folder, this will now work.
learn = load_learner("banana_disease_model.pkl")

# Define your classes in the order they were trained (0, 1, 2, 3)
CLASSES = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]

def predict(img):
    # Fastai's predict returns (pred, pred_idx, probs)
    _, _, probs = learn.predict(img)
    
    # Return a dictionary of {ClassName: Probability}
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

# Build the Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Banana Leaf Disease Classifier",
    description=f"Identify: {', '.join(CLASSES)}"
)

if __name__ == "__main__":
    demo.launch()