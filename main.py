from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertForTokenClassification, BertTokenizerFast
import pickle

# Define FastAPI app
app = FastAPI()

# Load model and tokenizer
MODEL_PATH = "data/ner_model"

def load_model():
    model = BertForTokenClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    return model, tokenizer

model, tokenizer = load_model()

# Load label mappings
with open("data/id2tag.pkl", "rb") as f:
    id2tag = pickle.load(f)

# Request body schema
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_ner(request: TextRequest):
    user_input = request.text.strip()

    if not user_input:
        return {"error": "Empty input provided."}

    # Tokenize input
    tokens = tokenizer(user_input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**tokens)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    # Convert predictions to entity labels
    predicted_labels = [id2tag[p] for p in predictions][:len(user_input.split())]

    # Return JSON response
    return {"tokens": user_input.split(), "entities": predicted_labels}