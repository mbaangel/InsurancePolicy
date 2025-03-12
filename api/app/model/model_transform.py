from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def model_transform():
    
#model_path = "./llama3_3b"  # Ruta donde descargaste el modelo
# model_path = "./microsoft_phi-2"  # Ruta donde descargaste el modelo
    model_path = "./zephyr-7b-alpha"  # Ruta donde descargaste el modelo
    model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    return model, tokenizer

