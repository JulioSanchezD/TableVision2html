# 📄 granite-vision-3.2-2b-table2html

## Overview

`granite-vision-3.2-2b-table2html` is a fine-tuned multimodal model based on [granite-vision-3.2-2b](https://huggingface.co/ibm-granite/granite-vision-3.2-2b). It specializes in extracting HTML `<table>` structures from images of tables.

## Intended Use

- 🧾 **Input**: An image containing a table (e.g., screenshot, scan, or photo).
- 🧪 **Output**: HTML snippet limited to the `<table>...</table>` content that structurally and semantically represents the table in the image.

### Use Cases

- OCR post-processing for tables  
- Automatic document parsing  
- AI agents generating structured markup from visual input

## Training Details

This model was fine-tuned using PEFT with LoRA (Low-Rank Adaptation) to reduce memory footprint and improve training efficiency.

- **Training Dataset**: [`apoidea/pubtabnet-html`](https://github.com/JulioSanchezD/TableVision2html/blob/main/notebooks/evaluation.ipynb)  
- **System Message**: `"Convert table to HTML (<table> ... </table>)"`
- **Number of Training Images**: 10,000  
- **Number of Test Images**: 250  
- **Max Sequence Length**: 1024  
- **Gradient Accumulation Steps**: 8  
- **Epochs**: 1  
- **Batch Size**: 1 (per device)  
- **Learning Rate**: 3e-4  
- **Warmup Steps**: 10  
- **Weight Decay**: 0.01  
- **Optimizer**: `adamw_torch_fused`  
- **Precision**: bf16

### LoRA Configuration (PEFT)

```python
target_modules = []
for layer_type in layers_to_tune:
    target_modules.extend(
        name for name, _ in model.named_modules()
        if (layer_type in name) 
        and '_proj' in name
    )
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules,
    use_dora=True,
    init_lora_weights="gaussian"
)
```

## Evaluation

- 🧪 **Eval Loss**: `0.0118`  
- 🧮 [**HTML Similarity**](https://github.com/JulioSanchezD/TableVision2html/blob/main/notebooks/evaluation.ipynb): `0.770`
These metrics indicate that the model not only converged well during training but also performs accurately on semantic table reconstruction tasks.

## Limitations

- ❌ Not designed for full HTML document generation  
- ❌ May struggle with highly complex or nested tables  
- ⚠️ Requires reasonably clean and well-captured table images

## How to Use

```python
from src.models.granite_vision.transformers_library import LLM as granite_vision

model = granite_vision(
    model_path,
    adapter='lang_table_only_4'
)

prompt = "Convert table to HTML (<table> ... </table>)"
html = model.predict(image, max_new_tokens=1024, query=prompt)
```

## HuggingFace Model

[JulioSnchezD/granite-vision-3.2-2b-table2html](https://huggingface.co/JulioSnchezD/granite-vision-3.2-2b-table2html)

## Blog Post
👉 Read the full story behind this project:
["Fine-Tuning Granite-Vision 2B to Outperform 90B Giants (Table Extraction Task)"](https://medium.com/@julioe.sanchezd/how-i-fine-tuned-granite-vision-2b-to-beat-a-90b-model-insights-and-lessons-learned-ebec8fe8f9fb)

## Citation
If you use this model, please cite the work:

```bibtex
@misc{granite2025table2html,
  title={granite-vision-3.2-2b-table2html: Table HTML extraction from images},
  author={Julio Sánchez},
  year={2025},
  howpublished={\url{https://huggingface.co/JulioSnchezD/granite-vision-3.2-2b-table2html}},
}
```