import sys
import json
from pathlib import Path

print(f"Python version: {sys.version.split()[0]}")
print(f"Python executable: {sys.executable}")

PROJECT_ROOT = Path('.').resolve()
print(f"Project root: {PROJECT_ROOT}")

required_pkgs = {
    'torch': 'torch',
    'fastai': 'fastai',
    'gradio': 'gradio',
    'PIL': 'Pillow'
}

missing = []
for module_name, pip_name in required_pkgs.items():
    try:
        __import__(module_name)
    except Exception:
        missing.append(pip_name)

if missing:
    print('Missing packages detected:')
    print(', '.join(missing))
    print('Install command: pip install ' + ' '.join(missing))
else:
    print('All required packages are available.')

import torch
import pandas as pd
import gradio as gr
from PIL import Image
from fastai.vision.all import load_learner

MODEL_CANDIDATES = [
    Path('banana_disease_model.pkl'),
    Path('model_artifacts/learner_export.pkl')
]

METADATA_PATH = Path('model_artifacts/model_metadata.json')
DATA_DIR = Path('banana_data/bananaLSD/AugmentedSet')
DEFAULT_SAMPLE_IMAGE = Path('banana_data/bananaLSD/AugmentedSet/healthy').glob('*.jpg')
DEFAULT_SAMPLE_IMAGE = next(DEFAULT_SAMPLE_IMAGE, None)


def resolve_model_path(candidates):
    for model_path in candidates:
        if model_path.exists():
            return model_path
    raise FileNotFoundError(
        'No compatible learner file found. Expected one of: ' + ', '.join(str(p) for p in candidates)
    )


def load_metadata(metadata_path):
    if metadata_path.exists():
        with metadata_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    return {'classes': []}


def get_class_names_from_data(data_dir):
    """Extract class names from dataset directory structure, sorted alphabetically."""
    if data_dir.exists():
        classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        if classes:
            return classes
    return []


def format_confidence(value):
    return f"{value * 100:.2f}%"
class BananaLeafDiseasePredictor:
    def __init__(self, model_path: Path, metadata: dict):
        self.model_path = model_path
        self.metadata = metadata
        self.learn = load_learner(model_path)
        
        # Try to get class names from dataset directory (alphabetically sorted)
        self.class_names = get_class_names_from_data(DATA_DIR)
        
        # Fallback to metadata if available
        if not self.class_names:
            metadata_classes = metadata.get('classes', [])
            if metadata_classes:
                self.class_names = [str(c) for c in metadata_classes]
        
        # Final fallback to learner vocab
        if not self.class_names:
            self.class_names = [str(c) for c in self.learn.dls.vocab]
        
        print(f"Class mapping: {self.class_names}")

    def predict(self, image: Image.Image) -> dict:
        if image is None:
            raise ValueError('Please upload an image before running prediction.')
        if not isinstance(image, Image.Image):
            raise TypeError('Input must be a PIL.Image.Image object.')

        pred_class, pred_idx, probs = self.learn.predict(image)
        # Explicitly convert pred_idx to int and map to disease name
        pred_idx_int = int(pred_idx) if hasattr(pred_idx, '__int__') else pred_idx
        predicted_disease = self.class_names[pred_idx_int]
        
        print(f"DEBUG: pred_class={pred_class}, pred_idx={pred_idx}, pred_idx_int={pred_idx_int}, disease={predicted_disease}")
        
        probabilities = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }
        probabilities = dict(sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True))

        return {
            'disease': predicted_disease,
            'confidence': float(probs[pred_idx_int]),
            'probabilities': probabilities,
        }
    
MODEL_PATH = resolve_model_path(MODEL_CANDIDATES)
METADATA = load_metadata(METADATA_PATH)
PREDICTOR = BananaLeafDiseasePredictor(MODEL_PATH, METADATA)

print(f'Model loaded from: {MODEL_PATH}')
print(f'Discovered class names (from dataset): {PREDICTOR.class_names}')

learner_classes = [str(c) for c in PREDICTOR.learn.dls.vocab]
print(f'Learner vocab: {learner_classes}')

if PREDICTOR.class_names != learner_classes:
    print(f'Note: Using dataset class names instead of learner vocab.')


def run_inference(image: Image.Image):
    result = PREDICTOR.predict(image)
    top_class = result['disease']
    confidence_text = format_confidence(result['confidence'])
    probs_df = pd.DataFrame(
        {
            'class': list(result['probabilities'].keys()),
            'probability': list(result['probabilities'].values()),
        }
    )
    probs_df['probability_percent'] = (probs_df['probability'] * 100).round(2)
    probs_df = probs_df[['class', 'probability_percent']].reset_index(drop=True)
    return top_class, confidence_text, probs_df

# Expected failure path: no image provided
try:
    run_inference(None)
except Exception as exc:
    print('Validation check (expected):', exc)

# Recovery path: use a sample image if one exists
if DEFAULT_SAMPLE_IMAGE and DEFAULT_SAMPLE_IMAGE.exists():
    sample_image = Image.open(DEFAULT_SAMPLE_IMAGE)
    sample_pred, sample_conf, sample_df = run_inference(sample_image)
    print('Recovery check:')
    print('Prediction:', sample_pred)
    print('Confidence:', sample_conf)
    print(sample_df.to_string(index=False))
else:
    print('No default sample image found; upload an image in the UI section.')

SUPPORTED_CLASSES = ', '.join(PREDICTOR.class_names)


def gradio_predict(image):
    if image is None:
        return 'No image provided', 'N/A', pd.DataFrame(columns=['class', 'probability_percent'])
    top_class, confidence_text, probs_df = run_inference(image)
    return top_class, confidence_text, probs_df


demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type='pil', label='Upload Banana Leaf Image'),
    outputs=[
        gr.Textbox(label='Predicted Disease'),
        gr.Textbox(label='Confidence'),
        gr.Dataframe(label='Class Probabilities (%)')
    ],
    title='Banana Leaf Disease Classifier',
    description=f'Upload one leaf image. Supported classes: {SUPPORTED_CLASSES}.',
)

print('Launching Gradio UI...')
print('If inline preview does not render, use the local URL shown by Gradio.')
ui_handle = demo.launch(inline=True, share=False, quiet=True, prevent_thread_lock=True)