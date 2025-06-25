from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from PIL import Image
import io
import base64
import os
from typing import Optional
import huggingface_hub
from huggingface_hub import HfFolder
import time

# Configure Hugging Face Hub settings
huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 300  # 5 minutes timeout
huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = False

# Set environment variable for HF mirror (optional, uncomment if needed)
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_with_retry(model_id, max_retries=3, retry_delay=10):
    """Download model with retry logic"""
    for attempt in range(max_retries):
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Download attempt {attempt + 1} failed: {str(e)}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

app = FastAPI(title="Medical Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "medical_phi2_model")
BASE_MODEL = "microsoft/phi-2"  # Base model to load

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model not found at {MODEL_PATH}")
    print("Please ensure you have:")
    print("1. Trained the model in Google Colab")
    print("2. Downloaded the model from Colab")
    print("3. Unzipped the model and placed it in the backend directory")
    print("Expected structure:")
    print("backend/")
    print("├── main.py")
    print("├── requirements.txt")
    print("└── medical_phi2_model/")
    print("    ├── adapter_config.json")
    print("    ├── adapter_model.safetensors")
    print("    └── tokenizer.json")

class Query(BaseModel):
    query: str
    image: Optional[str] = None  # Base64 encoded image

class Response(BaseModel):
    response: str

# Initialize model and tokenizer
try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        # Load the base model first
        print("Loading base model...")
        base_model = download_with_retry(BASE_MODEL)
        
        # Load the PEFT adapter
        print("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        
        # Load tokenizer from base model to avoid compatibility issues
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        # Move model to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"Model moved to {device}")
        
        print("Model loaded successfully!")
    else:
        print("Model not found. Please follow the setup instructions.")
        model = None
        tokenizer = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

def process_image(image_data: str) -> str:
    """Process base64 image data and return a description"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Here you would typically:
        # 1. Preprocess the image
        # 2. Run it through a vision model
        # 3. Generate a description
        # For now, we'll return a placeholder
        return "Medical image detected. Please provide more context about what you'd like to know about this image."
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def generate_response(query: str, image_description: Optional[str] = None) -> str:
    """Generate response using the fine-tuned model"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please ensure the model is properly installed.")
    
    try:
        # Combine query and image description if available
        if image_description:
            prompt = f"Question: {query}\nContext: {image_description}\nAnswer:"
        else:
            prompt = f"Question: {query}\nAnswer:"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the answer part
        response = response.split("Answer:")[-1].strip()
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/query", response_model=Response)
async def process_query(query: Query):
    """Process a query and optional image"""
    try:
        image_description = None
        if query.image:
            image_description = process_image(query.image)
        
        response = generate_response(query.query, image_description)
        return Response(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 