import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from torch.utils.data import Dataset as TorchDataset
import json
import pandas as pd
from huggingface_hub import HfApi
import requests
from tqdm import tqdm
import zipfile
import io
import bitsandbytes as bnb

# Configuration
MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "medical_phi2_model"
LORA_R = 4  # Further reduced
LORA_ALPHA = 8  # Further reduced
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_EPOCHS = 3
MAX_LENGTH = 128  # Further reduced

def download_medqa_data():
    """Download MedQA dataset directly from the source"""
    print("Downloading MedQA dataset...")
    url = "https://raw.githubusercontent.com/jind11/MedQA/master/data_clean/questions/US/train.jsonl"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = []
        for line in response.text.split('\n'):
            if not line.strip():
                continue
            try:
                # Clean the line
                line = line.strip()
                # Remove any BOM or special characters
                line = line.encode('utf-8').decode('utf-8-sig')
                
                # Parse JSON
                item = json.loads(line)
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    data.append({
                        'question': item['question'],
                        'answer': item['answer']
                    })
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line: {e}")
                continue
        
        if not data:
            raise ValueError("No valid examples found in MedQA dataset")
            
        print(f"Successfully loaded {len(data)} MedQA examples")
        return data
    except Exception as e:
        print(f"Error downloading MedQA dataset: {str(e)}")
        # Return some example data if download fails
        return [
            {
                'question': 'What is the most common cause of acute viral hepatitis worldwide?',
                'answer': 'Hepatitis A virus (HAV) is the most common cause of acute viral hepatitis worldwide.'
            },
            {
                'question': 'What is the first-line treatment for uncomplicated malaria?',
                'answer': 'Artemisinin-based combination therapy (ACT) is the first-line treatment for uncomplicated malaria.'
            }
        ]

def download_pubmedqa_data():
    """Download PubMedQA dataset directly from HuggingFace"""
    print("Downloading PubMedQA dataset...")
    try:
        # Use a different approach to load the dataset
        dataset = load_dataset(
            "pubmed_qa",
            "pqa_labeled",
            split="train",
            cache_dir=None,
            download_mode="force_redownload"
        )
        
        if len(dataset) == 0:
            raise ValueError("No examples found in PubMedQA dataset")
            
        return dataset
    except Exception as e:
        print(f"Error loading PubMedQA dataset: {str(e)}")
        # Return some example data if loading fails
        return Dataset.from_dict({
            'question': [
                'What are the risk factors for developing type 2 diabetes?',
                'How does aspirin work as an antiplatelet agent?'
            ],
            'long_answer': [
                'The main risk factors for type 2 diabetes include obesity, physical inactivity, family history, age, and ethnicity.',
                'Aspirin works as an antiplatelet agent by irreversibly inhibiting the cyclooxygenase-1 (COX-1) enzyme, which prevents the formation of thromboxane A2 and subsequent platelet aggregation.'
            ]
        })

class MedicalDatasetLoader:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_medqa(self):
        """Load MedQA dataset"""
        data = download_medqa_data()
        return self._process_medqa(data)
    
    def load_pubmedqa(self):
        """Load PubMedQA dataset"""
        dataset = download_pubmedqa_data()
        return self._process_pubmedqa(dataset)
    
    def _process_medqa(self, data):
        """Process MedQA data"""
        processed_data = []
        for item in data:
            processed_data.append({
                'question': item['question'],
                'answer': item['answer']
            })
        return processed_data
    
    def _process_pubmedqa(self, dataset):
        """Process PubMedQA data"""
        processed_data = []
        for item in dataset:
            processed_data.append({
                'question': item['question'],
                'answer': item['long_answer']
            })
        return processed_data

class MedicalQADataset(TorchDataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }

def main():
    try:
        # Initialize wandb
        wandb.init(project="medical-phi2-finetuning")
        
        # Create dataset directory
        os.makedirs("datasets", exist_ok=True)
        
        # Load tokenizer and model
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model with 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_8bit=True,  # Enable 8-bit quantization
            device_map="auto",
            use_cache=False,  # Disable KV cache
            torch_dtype=torch.float16
        )
        
        # Set model's padding token
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA with minimal parameters
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],  # Only target key attention modules
            inference_mode=False
        )
        
        model = get_peft_model(model, lora_config)
        
        # Load and process datasets
        print("Processing datasets...")
        dataset_loader = MedicalDatasetLoader(tokenizer, max_length=MAX_LENGTH)
        medqa_data = dataset_loader.load_medqa()
        pubmedqa_data = dataset_loader.load_pubmedqa()
        
        # Combine datasets
        combined_data = medqa_data + pubmedqa_data
        
        # Create dataset
        dataset = MedicalQADataset(combined_data, tokenizer, max_length=MAX_LENGTH)
        
        # Training arguments with aggressive memory optimizations
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=16,  # Increased for memory efficiency
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="wandb",
            run_name=f"medical-phi2-{wandb.run.id}",
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            remove_unused_columns=True,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=0,  # Disable multiprocessing
            dataloader_pin_memory=False  # Disable pinned memory
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save the model
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Close wandb
        wandb.finish()
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        wandb.finish()
        raise

if __name__ == "__main__":
    main() 