from langchain.llms.base import LLM
from huggingface_hub import InferenceClient
from typing import Any, List, Mapping, Optional

class HuggingFaceInferenceLLM(LLM):
    model_name: str = "HuggingFaceH4/zephyr-7b-alpha"
    temperature: float = 0
    max_new_tokens: int = 512
    client: Any = None  
    
    def __init__(self, model_name=None, temperature=None, max_new_tokens=None, **kwargs):
        if model_name:
            kwargs["model_name"] = model_name
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens:
            kwargs["max_new_tokens"] = max_new_tokens
        
        
        super().__init__(**kwargs)
        
        self.client = InferenceClient() #Si se loguean por terminal no es necesario poner el token acá
        
    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.text_generation(
            model=self.model_name,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature >= 0
        )
        
        if response.startswith(prompt):
            response = response[len(prompt):]
        return response
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens
        }