from model.huggin_face_inference import HuggingFaceInferenceLLM
from langchain.memory import ConversationBufferMemory

def get_large_language_model():
    return HuggingFaceInferenceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    memory=create_memory(),
    temperature=0.01,
    max_new_tokens=512
)


def create_memory():
    memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history",
    input_key="question"
    )
    return memory
