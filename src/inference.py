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
from src.utils.helpers import append_json_line, read_jsonl
from src.models.pix2struct.transformers import LLM as Pix2StructLLM
from src.models.granite_vision.transformers_library import LLM as GraniteVisionLLM


MODELS = {
    "ibm-granite/granite-vision-3.2-2b": GraniteVisionLLM,
    "KennethTM/pix2struct-base-table2html": Pix2StructLLM
}


def main(model_name: str, dataset: str, num_test_images: int, adapter: str = None) -> None:
    # Load dataset
    test_ds = load_dataset(dataset)['validation']
    test_ds = test_ds.select(range(num_test_images))
    print(f"Loaded {len(test_ds)} test images from {dataset}")

    # Load model
    if model_name not in MODELS:
        raise Exception(f"Model {model_name} not supported. Supported models are: {list(MODELS.keys())}")
    model = MODELS[model_name](model_name)
    tokenizer = model.processor.tokenizer

    # Load previous results
    results = [res['inference_id'] for res in read_jsonl('data/evaluation/results.jsonl')]

    # Iterate over test images
    for i in tqdm(range(len(test_ds))):
        # Get Ground Truth
        img = test_ds[i]['image']
        img_id = test_ds[i]['imgid']
        html = test_ds[i]['html_table']

        # Calculate inference ID and skip if already exists
        inference_id = hashlib.md5(f'{dataset}_{img_id}_{model_name}_{adapter}'.encode('utf-8')).hexdigest()
        if inference_id in results:
            continue

        # Predict
        max_tokens = 2 ** math.ceil(math.log2(len(tokenizer.encode(html, truncation=False))))

        start_time = timer()
        output = model.predict(img, max_new_tokens=max_tokens)
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
