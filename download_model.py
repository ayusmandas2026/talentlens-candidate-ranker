import os
import sys
from transformers import AutoTokenizer, AutoModel

# Set encoding to prevent windows console errors
sys.stdout.reconfigure(encoding='utf-8')

model_dir = "e:/India Runs/model"
os.makedirs(model_dir, exist_ok=True)

try:
    print("Downloading and saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer.save_pretrained(model_dir)

    print("Downloading and saving model...")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.save_pretrained(model_dir)

    print("Done! Model saved successfully to:", model_dir)
    sys.exit(0)
except Exception as e:
    print(f"Error occurred: {e}")
    sys.exit(1)
