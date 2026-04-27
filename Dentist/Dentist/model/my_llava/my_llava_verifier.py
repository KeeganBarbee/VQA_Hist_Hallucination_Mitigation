import requests
import base64
import io
import numpy as np
from PIL import Image
from Dentist.model.verifier import Verifier
import os

def call_ollama(model_name, prompt, image=None):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    if image is not None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        payload["images"] = [b64]
 
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=120
    )
    return response.json()["response"]
 
 
class LLaVA_Verifier(Verifier):
 
    def ask_model(self, image, prompt, use_image=True):
        if use_image and image is not None:
            return call_ollama("llava:7b", prompt, image)
        else:
            return call_ollama("llava:7b", prompt)
 
    def vqa_model_evaluation(self, original_image, questions):
        result = []
        for question in questions:
            answer = call_ollama("llava:7b", question, original_image)
            result.append({"question": question, "answer": answer})
        return result