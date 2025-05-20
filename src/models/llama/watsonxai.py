import os
import base64
from io import BytesIO
from transformers import AutoProcessor
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from dotenv import load_dotenv

load_dotenv()

class LLM:

    device_name = "watsonx.ai"

    def __init__(self, model_name, adapter=None):
        credentials = Credentials(
            url=os.getenv("URL"),
            api_key=os.getenv("WATSONX_APIKEY")
        )

        self.model = ModelInference(
            model_id=model_name,
            credentials=credentials,
            project_id=os.getenv("WATSONX_PROJECT_ID"),
            max_retries=1
        )

        proc_model_name = 'unsloth/Llama-3.2-90B-Vision' if '90' in model_name else 'unsloth/Llama-3.2-11B-Vision'
        self.processor = AutoProcessor.from_pretrained(proc_model_name, use_fast=True)

    @staticmethod
    def _augment_api_request_body(image):
        system_prompt = "You are a Table parser. You will be provided with an image of a table and you need to extract the table html code from the image." \
        "It is strictly forbidden to provide any other information or explanations other than writing HTML code. The first word you write is <table> and the" \
        " last word you write is </table>. You should not write any other words. "
        messages = [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": system_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{image}",
                    }
                }]
            }
        ]
        return messages
    
    @staticmethod
    def _encode_image_to_base64(pil_image, format="JPEG"):
        buffer = BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    
    def predict(self, img, max_new_tokens: int):
        base64_image = self._encode_image_to_base64(img.convert("RGB"))
        messages = self._augment_api_request_body(base64_image)
        params = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "stop": ["**Note"],
        }
        response = self.model.chat(messages=messages, params=params)
        output = response['choices'][0]['message']['content']
        return output.split('<|assistant|>')[-1].strip()
