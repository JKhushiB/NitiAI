# Run this in a separate Python script or Jupyter notebook:
from sentence_transformers import SentenceTransformer

print("Downloading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded successfully!")