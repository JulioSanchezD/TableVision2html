import os
import sys
import math
import hashlib

current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

from tqdm import tqdm
from datasets import load_dataset
from timeit import default_timer as timer
from src.models.llama.watsonxai import LLM as LlamaVisionLLM
from src.utils.helpers import append_json_line, read_jsonl, timeout
from src.models.pix2struct.transformers import LLM as Pix2StructLLM
from src.models.granite_vision.transformers_library import LLM as GraniteVisionLLM


MODELS = {
    "ibm-granite/granite-vision-3.2-2b": GraniteVisionLLM,
    "KennethTM/pix2struct-base-table2html": Pix2StructLLM,
    "meta-llama/llama-3-2-11b-vision-instruct": LlamaVisionLLM,
    "meta-llama/llama-3-2-90b-vision-instruct": LlamaVisionLLM
}

@timeout(seconds=60)
def predict(model, img, max_new_tokens):
    return model.predict(img, max_new_tokens=max_new_tokens)

def main(model_name: str, dataset: str, num_test_images: int, adapter: str = None) -> None:
    # Load dataset as a streaming iterable
    test_ds = load_dataset(dataset, split='validation', streaming=True)
    test_ds_iter = iter(test_ds)
    print(f"Streaming {num_test_images} test images from {dataset}")

    # Load model
    if model_name not in MODELS:
        raise Exception(f"Model {model_name} not supported. Supported models are: {list(MODELS.keys())}")
    model = MODELS[model_name](model_name, adapter=adapter)
    tokenizer = model.processor.tokenizer

    # Load previous results
    results = [res['inference_id'] for res in read_jsonl('data/evaluation/results.jsonl')]

    # Iterate over test images
    for _ in tqdm(range(num_test_images)):
        # Get Ground Truth
        sample = next(test_ds_iter)
        img = sample['image']
        img_id = sample['imgid']
        html = sample['html_table']

        # Calculate inference ID and skip if already exists
        inference_id = hashlib.md5(f'{dataset}_{img_id}_{model_name}_{adapter}'.encode('utf-8')).hexdigest()
        if inference_id in results:
            continue

        # Predict
        max_tokens = 2 ** math.ceil(math.log2(len(tokenizer.encode(html, truncation=False))))

        start_time = timer()
        try:
            output = predict(model, img, max_new_tokens=max_tokens)
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            continue
        end_time = timer()

        # Append to jsonl file
        append_json_line('data/evaluation/results.jsonl', {
            'inference_id': inference_id,
            'dataset': dataset,
            'imgid': img_id,
            'html': html,
            'model_name': model_name,
            'adapter': adapter,
            'model_prediction': output,
            'model_backend': model.device_name,
            'execution_time': end_time - start_time
        })
