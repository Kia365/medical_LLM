# Medical Assistant AI

A fine-tuned medical assistant based on Microsoft's Phi-2 model, capable of answering medical questions and interpreting medical images.

## Project Structure

```
.
├── train_medical_assistant.py    # Training script for Google Colab
├── medical_qa_dataset.json      # Sample dataset structure
├── requirements.txt             # Python dependencies
├── backend/                     # FastAPI backend
│   ├── main.py                 # Backend server
│   └── requirements.txt        # Backend dependencies
└── frontend/                    # React frontend
    ├── package.json
    └── src/
        └── App.js
```

## Setup Instructions

### 1. Training the Model

1. Open `train_medical_assistant.py` in Google Colab
2. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
3. Prepare your medical QA dataset in the format shown in `medical_qa_dataset.json`
4. Run the training script:
   ```bash
   !python train_medical_assistant.py
   ```
5. The fine-tuned model will be saved in the `medical_phi2_model` directory

### 2. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy your fine-tuned model to the backend directory:
   ```bash
   cp -r ../medical_phi2_model .
   ```

4. Start the backend server:
   ```bash
   python main.py
   ```
   The server will run on `http://localhost:8000`

### 3. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Usage

1. The frontend will be available at `http://localhost:3000`
2. The backend API will be available at `http://localhost:8000`
3. You can:
   - Ask medical questions in the text input
   - Upload medical images for interpretation
   - Get AI-powered responses

## API Endpoints

- `POST /query`: Process a medical query and optional image
  - Request body: `{ "query": "string", "image": "base64_string" }`
  - Response: `{ "response": "string" }`
- `GET /health`: Check API health and model status

## Model Details

- Base Model: Microsoft Phi-2 (2.7B parameters)
- Fine-tuning: LoRA (Low-Rank Adaptation)
- Training: 8-bit quantization for memory efficiency
- Dataset: Medical QA pairs and image descriptions

## Notes

- The model is fine-tuned using LoRA to reduce memory requirements
- Training is optimized for Google Colab's free GPU
- The frontend is built with React and Material-UI for a modern, responsive interface
- The backend uses FastAPI for efficient API handling

## License

This project is open source and free to use. The base model (Phi-2) is licensed under the MIT license. 