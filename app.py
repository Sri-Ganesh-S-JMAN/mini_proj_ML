import gradio as gr
from fastai.vision.all import load_learner

# 1. Load the model
learn = load_learner("banana_disease_model.pkl")

# 2. Define your mapping (Order must match your training indices 0-3)
# index 0 -> cordana, 1 -> healthy, etc.
CLASSES = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]

def predict(img):
    # Get prediction from fastai
    # pred is the numeric string (e.g., '0'), probs are the confidence scores
    pred, pred_idx, probs = learn.predict(img)
    
    # Map the probabilities to your class names
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

# 3. Launch the simplified interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Banana Leaf Disease Classifier",
    description=f"Identify diseases: {', '.join(CLASSES)}"
)

if __name__ == "__main__":
    demo.launch()