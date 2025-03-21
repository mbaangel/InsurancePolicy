import json
import torch
import os

def read_qna_data():
    qa_file = original_path()+"/model/qna/qna_polizas.json"
    with open(qa_file, "r", encoding="utf-8") as file:
        qna_data = json.load(file)
    return qna_data



def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def original_path():
    return os.getcwd()