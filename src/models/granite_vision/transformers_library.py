import torch
import warnings
from peft import PeftModel
from threading import Thread
from transformers.utils import logging
from transformers import AutoProcessor, AutoModelForVision2Seq, TextIteratorStreamer

transformers_logger = logging.get_logger("transformers")
transformers_logger.setLevel(logging.ERROR)

class LLM:

    ADAPTERS_PATH: str = "src/models/granite_vision/checkpoints/{}/checkpoint-1000"

    def __init__(self, model_name, adapter: str = None):
        if adapter:
            self.device = 'cpu'
        else:
            self.device, self.device_name = self._select_device()
            print(f"Using {self.device}: {self.device_name}")


        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map={"": 0} if self.device == 'cuda' else 'cpu',
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
            _attn_implementation="flash_attention_2"
        )
        print(f"Model loaded")

        # Load and merge LoRA adapter
        if adapter:
            adapter_path = self.ADAPTERS_PATH.format(adapter)
            self.model = PeftModel.from_pretrained(
                self.model, 
                adapter_path, 
                is_trainable=False,
                torch_dtype=torch.float32
            )
            print(f"Adapter '{adapter}' loaded")
            self.model = self.model.merge_and_unload()
            print(f"Adapter '{adapter}' merged")
            self.device, self.device_name = self._select_device()
            self.model = self.model.to(dtype=torch.bfloat16, device=self.device)
            self.model.eval()
            print(f"Using {self.device}: {self.device_name}")


    @staticmethod
    def _select_device():
        if torch.backends.mps.is_available():
            return 'mps', "Apple device with MPS support"
        elif torch.cuda.is_available():
            return 'cuda', torch.cuda.get_device_name(torch.cuda.current_device())
        return 'cpu', "CPU"
        
    
    def predict(self, img, max_new_tokens: int, query="Convert the user's image to HTML."):
        # Prepare prompt
        conversation = [
            {
            "role": "system",
            "content": [
                {"type": "text", "text": query}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"}
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        inputs = self.processor(images=[img], text=text, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = self.processor.decode(output[0], skip_special_tokens=True)
        return output.split('<|assistant|>')[-1].strip()


    def stream(self, img, max_new_tokens: int, query="Convert the user's image to HTML."):
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": query}]
            },
            {
                "role": "user",
                "content": [{"type": "image"}],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        # Set up the streamer for token generation
        streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)
        inputs = self.processor(images=[img], text=text, return_tensors="pt").to(self.device)

        # Set up generation arguments including max tokens and streamer
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
            **inputs
        }

        # Start a separate thread for model generation to allow streaming output
        thread = Thread(
            target=self.model.generate,
            kwargs=generation_args,
        )
        thread.start()

        # Yield text tokens as they are generated
        for text_token in streamer:
            yield text_token

        # Ensure the generation thread completes
        thread.join()