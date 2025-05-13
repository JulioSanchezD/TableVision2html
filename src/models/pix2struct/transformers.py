import torch, gc, time
from bs4 import BeautifulSoup
from datasets import load_dataset
from torchmetrics.text import BLEUScore
from IPython.display import display, HTML
from niteru import style_similarity, structural_similarity, similarity
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModel, AutoTokenizer, Pix2StructForConditionalGeneration


class LLM:

    def __init__(self, model_name, adapter: str = None):
        self.device, self.device_name = self._select_device()
        print(f"Using {self.device}: {self.device_name}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Load model
        self.model =Pix2StructForConditionalGeneration.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()


    @staticmethod
    def _select_device():
        if torch.backends.mps.is_available():
            return 'mps', "Apple device with MPS support"
        elif torch.cuda.is_available():
            return 'cuda', torch.cuda.get_device_name(torch.cuda.current_device())
        return 'cpu', "CPU"
        
    
    def predict(self, img, max_new_tokens: int):
        encoding = self.processor(img, return_tensors="pt", max_patches=1024, legacy=False)
        with torch.inference_mode():
            flattened_patches = encoding.pop("flattened_patches").to(self.device)
            attention_mask = encoding.pop("attention_mask").to(self.device)
            predictions = self.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        return self.processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)[0]
