# How I Fine-Tuned Granite-Vision 2B to Beat a 90B Model - Insights and Lessons Learned

## A hands-on journey exploring efficient fine-tuning techniques that unlock the power of small vision models.

Fine-tuning large language or vision-language models is a powerful technique that unlocks their potential on specialized tasks. However, despite their effectiveness, these approaches are often out of reach for many users due to their high computational cost and the need for GPUs with large VRAM — resources that only a small percentage of end users can access.

In this project, I fine-tuned IBM's [Granite-Vision 2B](https://huggingface.co/ibm-granite/granite-vision-3.2-2b), a relatively small yet powerful vision-language model, to tackle the challenge of converting images of tables into clean, structured HTML code.

What makes this project particularly exciting is that the fine-tuning was performed on a consumer-grade GPU — the NVIDIA RTX 4070 Ti Super — and yet, the resulting 2-billion-parameter model was able to outperform much larger models, including [meta-llama/Llama-3.2-90B-Vision](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision), on this image-to-text generation task. This success not only demonstrates the power of parameter-efficient fine-tuning methods like LoRA but also highlights the practical value of building specialized small models tailored to specific problems.

In this post, I’ll walk you through the motivation behind this work, the model and dataset choices, the custom HTML similarity metric I adapted, the experiments and results, and finally, the key insights and lessons learned throughout the process. Whether you’re interested in vision-language models, fine-tuning techniques, or practical AI applications, I hope this journey offers useful takeaways. The fine-tuning code used for this project was adapted from [HuggingFace's Granite Vision fine-tuning cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_granite_vision_sft_trl), authored by Eli Schwartz, who in turn adapted the original code from Sergio Paniego.


## Motivation

While working on Retrieval-Augmented Generation (RAG) projects, I encountered a major challenge: accurately extracting large and complex tables from PDFs, especially when these tables appeared as images. Despite trying different approaches — including tools like Unstructured and large vision-language models such as Meta's Llama 90B — the results often fell short of the accuracy needed.

This led me to consider a different approach: a **small, specialized vision-language model focused exclusively on table understanding and extraction**. Such a model could serve as a dedicated preprocessing step to significantly improve RAG pipelines that rely on accurate table extraction.

Around the same time, IBM released Granite-Vision 2B — a vision-language model with just the right balance of size and power. It’s capable enough to handle complex tables, yet small enough to be fine-tuned on consumer-grade GPUs with 16 GB of VRAM. This made it an ideal candidate for my project.


## The Task: Image to HTML (Table Extraction)

One important design choice was the target format: **HTML**. By converting tables into clean HTML code, we obtain a structured and widely supported representation that can be easily converted into other formats. For example, HTML tables can be readily imported into data analysis tools like Pandas as dataframes, making downstream processing and analysis much more efficient.

The original plan was to build a custom dataset by extracting HTML table tags, rendering them as images, and pairing each image with its corresponding HTML code. Fortunately, I found a solution: the [PubTabNet-HTML](https://huggingface.co/datasets/apoidea/pubtabnet-html) dataset, which includes over 568,000 image–HTML pairs, far more than needed for this project.

PubTabNet was developed by IBM and is based on scientific articles from the **PubMed Central Open Access Subset (commercial use collection)**. The tables were extracted by aligning PDF and XML versions of the articles.
The **annotations** (i.e., the HTML labels) are licensed under the **Community Data License Agreement – Permissive – Version 1.0**, and while **IBM does not own the images**, they are used in accordance with the [PMC Open Access Subset Terms of Use](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/). This makes the dataset suitable for both research and commercial applications, provided the license terms are followed.


## Custom Metric: HTML Similarity

Standard text similarity metrics like BLEU or ROUGE are insufficient for evaluating HTML table generation because they primarily focus on surface-level text matching and ignore important structural and stylistic aspects of HTML code.

To better capture the quality of generated HTML tables, I adapted a custom **HTML Similarity** metric that combines multiple complementary components, where the most important ones (style and structure) are imported from [**niteru**](https://raw.githubusercontent.com/ninoseki/niteru):

- **Style similarity (S):** Extracts CSS classes of each html document and calculates the jaccard similarity of the sets of classes.
- **Structural similarity (T):** Uses sequence comparison of the html tags to compute the similarity.
- **Content similarity (C):** Based on normalized edit distance between the extracted plain text content of the tables.
- **Token overlap similarity (J):** The Jaccard similarity between the sets of content tokens.

The final similarity score **M** is a weighted sum of these components:

![HTML Similarity Formula](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/html_similarity_formula.png)

I manually tested the metric on various example outputs, iteratively adjusting the weighting coefficients to better capture meaningful similarities. This process resulted in a balanced evaluation that fairly rewards accurate table structure and style, alongside precise textual content. Python implementation is as follows:

```python
from torchmetrics.text import EditDistance
from niteru import style_similarity, structural_similarity

ed_distance = EditDistance()

def extract_table_text(html):
    """Extracts only the text from an HTML table in row-wise space-separated format."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")  # Find the first table
    if not table:
        return ""

    # Extract rows and join cells with spaces
    return "\n".join(" ".join(cell.get_text(strip=True) for cell in row.find_all(["th", "td"])) for row in table.find_all("tr"))

def extract_html_table(html):
    """Extracts html table from text"""
    match = re.search(r'<table\b.*?</table>', html, re.DOTALL | re.IGNORECASE)
    if match:
        table_html = match.group()
        return table_html
    else:
        return html

def html_similarity(html1, html2):
    html1 = extract_html_table(html1)
    html2 = extract_html_table(html2)
    # Compute individual similarity scores
    style_sim = style_similarity(html1, html2)  # Assume returns [0,1]
    struct_sim = structural_similarity(html1, html2)  # Assume returns [0,1]
    txt1, txt2 = extract_table_text(html1), extract_table_text(html2)
    content_sim = 1 - (ed_distance(txt1, txt2) /
                                   max(len(txt1), len(txt2) + 1e-10))  # Avoid division by zero
    jaccard_sim = 1 - (len(set(txt1.split()).intersection(set(txt2.split()))) /
                        len(set(txt1.split()).union(set(txt2.split()))) + 1e-10)
    
    # Weighted sum of the similarities
    final_score = (0.10 * style_sim) + (0.40 * struct_sim) + (0.30 * content_sim) + (0.20 * jaccard_sim)

    # Ensure final score is in [0,1]
    final_score = max(0, min(1, final_score))

    return final_score
```
The metric also includes a regex-based function to extract only the HTML content within `<table>` tags. This was necessary because one of the reference models only generated incomplete or extra HTML outside of the table structure. By focusing the comparison strictly on the table content, the metric provides a more fair and meaningful evaluation across models.

Developing a custom evaluation metric like this is crucial for reliably tracking model improvements and benchmarking performance against reference models.


## Training Setup

To fine-tune the model efficiently on my NVIDIA RTX 4070 Ti Super, which has 16 GB VRAM, I used **LoRA** (Low-Rank Adaptation). This allowed me to update only a small number of parameters, significantly reducing GPU memory usage. In fact, during training, the model used only about half of the available VRAM — with enough headroom to play around with longer sequences, but not enough to handle more than one batch. Additionally, LoRA is generally faster to train than approaches like QLoRA.

### LoRA Setup

I used the following LoRA configuration:

```python
# Setup LoRA
target_modules = []
for layer_type in layers_to_tune:
    target_modules.extend(
        name for name, _ in model.named_modules()
        if (layer_type in name) 
        and '_proj' in name
    )

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules,
    use_dora=True,
    init_lora_weights="gaussian"
)
```
Key points:

- `r=16`: This low-rank dimension provides a good balance between model capacity and GPU memory usage.
- `use_dora=True`: [DoRA](https://arxiv.org/pdf/2402.09353) (Weight-Decomposed Low Rank Adaptation) improves the learning capacity and stability of LoRA by decomposing the pretrained weights into magnitude and direction components, helping the model better resemble the capacity of full fine-tuning — all without adding inference overhead. Performed slightly better than the default setting.
- `init_lora_weights="gaussian"`: No particular reason, I didn't want to experiment with this parameter.
- `target_modules`: This flexible setup allows selectively targeting vision layers, language layers, or both, depending on the experiment. In practice, vision layers remained unaffected — even with `use_dora=False` — since DoRA currently supports only embedding, linear, and Conv2d layers. As a result, I fine-tuned only the language layers.

### Dataset Setup

During my initial experiments, I kept running into **out-of-memory (OOM)** errors — even though there was still plenty of available GPU VRAM after loading model, LoRA layers and optimizer parameters (around 4GB still free). There were no memory spikes during training, but the crashes consistently happened at the same training step.

After some investigation, I realized that the problem was caused by large tables, which resulted in extremely long token sequences. To address this, I adjusted the `max_seq_length` parameter and filtered out samples that exceeded this limit. After experimentation, I found that using `max_seq_length = 1024` allowed me to fine-tune the model reliably without triggering OOM errors.

To filter out oversized tables, I wrote a simple data processing function that:

- Filters out samples whose HTML token length exceeds `max_seq_length`
- Automatically balances the number of training and test samples
- Uses **streaming** to avoid loading the entire dataset into memory (PubTabNet-HTML is quite large, around 10 GB on disk)

```python
def load_process_filter_dataset(dataset, max_seq_length, num_train_images, num_test_images, system_message):
    global processor
    ds = load_dataset(dataset, split='train', streaming=True)
    max_html_tokens = max_seq_length - len(processor.tokenizer.tokenize(system_message))
    num_total_needed = num_train_images + num_test_images

    filtered_samples = []
    p_bar = tqdm(total=num_total_needed, desc="Filtering dataset samples")
    for sample in ds:
        processed = process_and_filter_example(sample, max_html_tokens)
        if processed:
            filtered_samples.append(processed)
            p_bar.update(1)
        if len(filtered_samples) >= num_total_needed:
            break
    p_bar.close()

    # Convert to in-memory dataset
    ds_filtered = Dataset.from_list(filtered_samples)

    # Split into train/test
    ds_train = ds_filtered.select(range(num_train_images))
    ds_test = ds_filtered.select(range(num_train_images, num_total_needed))

    return ds_train, ds_test

def process_and_filter_example(example, max_html_tokens):
    global processor
    extracted_table = extract_html_table(example['html_table'])
    token_count = len(processor.tokenizer.tokenize(extracted_table))
    if token_count < max_html_tokens:
        example['html_table'] = extracted_table
        return example
    return None
```
The final configuration included `num_train_images=10000` and `num_test_images=250` to compute the evaluation loss.

### Fine-Tuning Configuration

For training, I used the **Transformers SFTTrainer** to fine-tune the model:

```python
    # Training arguments
    training_args = SFTConfig(
        output_dir=f"src/models/{model_name.split('/')[-1].replace('-', '_', 1).split('-')[0]}/checkpoints/{experiment_name}",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        warmup_steps=10,
        learning_rate=3e-4,
        weight_decay=0.01,
        logging_strategy="steps",
        eval_strategy='steps',
        logging_steps=25,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        greater_is_better=False,
        load_best_model_at_end=True,
        optim="adamw_torch_fused",
        bf16=True,
        push_to_hub=False,
        report_to="wandb" if not debug else "none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=8
    )

    # Setup Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer
    )
```
Key points:

- `num_train_epochs=1`: The dataset is very large, and to run multiple experiments efficiently, I chose to train for only one full epoch while maximizing learning per sample and number of training samples.
- `per_device_train_batch_size=1`: Larger batch sizes would not fit in GPU memory without significantly reducing max_seq_length — which would hurt performance on large tables. Keeping longer sequences was more important for this task.
- `gradient_accumulation_steps=8`: Used to effectively simulate a larger batch size and help stabilize the learning process, compensating for the small physical batch. This is the final value, but experimented with `gradient_accumulation_steps=4` as well.
- `optim="adamw_torch_fused"` and `bf16=True`: These settings leverage modern NVIDIA architectures (Ada Lovelace) to accelerate training and reduce memory usage — as recommended for this hardware.

### Evaluation Loss Workaround

At the time of developing the project, there is a known issue in the Transformers + LoRA integration that causes an error when running evaluation with a validation dataset during training. Fortunately, a community-tested [workaround](https://raw.githubusercontent.com/huggingface/transformers/issues/33420#issuecomment-2957417282) is available (although not yet merged into the main branch), and I successfully used this fix in my experiments.

## Evaluation (Inference) Setup

The evaluation dataset used for final scoring was completely independent from the `eval_dataset` used during training. It consists of **500 randomly selected images**, none of which were included in either the `train_dataset` or the training `eval_dataset`.

Once fine-tuning was complete, I used the **best model checkpoint** — selected based on the lowest evaluation loss — to run inference on these 500 samples.

Initially, I attempted to perform inference by simply **loading the LoRA/DoRA adapter** on top of the base model. However, I found that inference with **DoRA adapters is extremely slow** when not merged into the model weights (as explained in the [official PEFT docs](https://raw.githubusercontent.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md)). In fact, generating one test random sample took about **90 seconds** in this configuration.

To resolve this, I merged the adapter weights into the base model — which is the recommended practice — and after merging, inference speed improved dramatically: **down to ~20 seconds for the same sample**, making full evaluation runs much more practical.

The **reference models** used for comparison with my fine-tuned models are:

- [meta-llama/Llama-3.2-90B-Vision](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision): Meta’s massive **90-billion parameter** model — the main baseline I aimed to surpass through specialization and parameter-efficient fine-tuning of a much smaller VLM.

- [KennethTM/pix2struct-base-table2html](https://huggingface.co/KennethTM/pix2struct-base-table2html): A much smaller model fine-tuned from Google’s [pix2struct-base](https://huggingface.co/google/pix2struct-base), highly specialized for exactly the same dataset I used in this project. Thanks to its smaller size, the developer(s) was able to train it for many more samples and over longer training runs — demonstrating the key advantage of using **smaller, targeted models** for specific tasks.

These two baselines allowed me to benchmark both **scaling-based performance** (vs the 90B model) and **specialization efficiency** (vs the smaller, dedicated Pix2Struct model).


## Experiments & Results

A total of 9 experiments were conducted, iteratively modifying one or two components at a time. The goal was to understand the effect of each change on model performance, gradually refining the setup to achieve the best possible **HTML Similarity** score compared to reference models.

The experimental process was incremental: whenever a change improved the results, it was incorporated into the next round of experiments and continued exploring new variations.

The experiments focused on adjusting the following components:

1. **Vision vs. Language Layers**
    - 1.1 `lang_only`
    - 1.2 `vision_only`
    - 1.3 `lang_vision`
2. **Ground Truth Output Format**
    - 2.1 `lang_table_only`
3. **Training Framework**
    - 3.1 `lang_table_unsloth`
    - 3.2 `vision_table_unsloth`
4. **Gradient Accumulation**
    - 4.1 `lang_table_only_2`
5. **Prompt Format**
    - 5.1 `lang_table_only_3`
6. **Gradient Accumulation & Dataset Size**
    - 6.1 `lang_table_only_4`

Both the **evaluation loss** and the **HTML Similarity** metric were used to assess model performance, and I found them to be well correlated — confirming that HTML Similarity is a good proxy for how well the model is learning the task.

Before diving into the results of each experiment, let’s first look at **GPU memory utilization during training**, which is often the most critical factor in determining whether a model can be fine-tuned on consumer hardware.

![Training GPU Utilization](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/gpu_utilization.png)

As shown in the graph, GPU utilization remained stable throughout training — averaging around **75% VRAM usage**, or roughly **12 GB** on my GPU.
Most of VRAM usage (~5.5 GB) is the frozen model weights. LoRA gradients + optimizer states take very little (<< 1 GB). Activations + overhead should fill the rest (~5–6 GB), which depends on `batch_size` and `max_seq_length`.

### First Run: `lang_only`

This experiment uses the following initial components/parameters:

| Parameter                     | Value                           |
| ----------------------------- | ------------------------------- |
| `layers_to_tune`              | "language_model"                |
| `gradient_accumulation_steps` | 4                               |
| `num_train_images`            | 5000                            |
| Prompt                        | "Convert user's image to HTML"  |
| Output Format                 | "&lt;html&gt; ... &lt;/html&gt;" |
| Training Framework            | HuggingFace Transformers        |

These were the starting values for the first experiment. In subsequent runs, I modified many of them as I refined the approach. This first experiment focused only on tuning **language layers**, while training the model to predict the full raw HTML output — including everything inside and around the `<table>` tags.

Since this was the first run, I’ll include the **training loss curve** here to illustrate how it behaves. For later experiments, I’ll omit this graph — as the behavior was similar across runs, with minor variations. In practice, **evaluation loss** is more useful for comparing performance across experiments.

![Train Loss: lang_only](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/train_loss_lang_only.png)

One important note about the logging configuration:  
`logging_steps=25` means that the training loss is only logged after every 25 steps, where each logged value is the average over `gradient_accumulation_steps=4`. As a result, the largest drop in loss appears at the second log point — where most of the initial learning happens. After that, the model continues learning more gradually, with a slow decreasing trend, depending on the difficulty of the training samples.

Now, let’s take a look at the **evaluation loss**:

![Eval Loss: lang_only](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/eval_loss_lang_only.png)

Since we are evaluating on the same set of 250 validation samples, the evaluation loss curve gives us a more stable and meaningful view of model learning — and will serve as a baseline for comparisons across future runs.

Here, we observe a clear and consistent downward trend throughout training. The initial loss starts close to **0.03**, with a steady improvement as training progresses, eventually stabilizing just below **0.015**. 

The smooth nature of this curve — compared to the more variable training loss — reflects the regular structure of the validation set and confirms that the model is generalizing well to unseen samples, even with a small batch size and a single epoch of training.

Now, let's compare the performance of this fine-tuned model against the reference models on the **HTML Similarity** metric:

| Model                                         | HTML Similarity |
|-----------------------------------------------|-----------------|
| Granite-Vision 2B                             | 0.56            |
| meta-llama/Llama-3.2-90B-Vision               | 0.62            |
| Fine-tuned Granite-Vision 2B (`lang_only`)    | 0.74            |
| KennethTM/pix2struct-base-table2html          | **0.75**        |

As we can see, this first experiment already delivers **strong performance gains** — improving the base Granite-Vision 2B model by a large margin (+0.18) and clearly outperforming LLaMA 90B Vision on this specialized task. Only Pix2Struct retains a slight lead at this stage.

### Second Run: `vision_only`

There isn’t much to analyze in this run. I tested several variations that could potentially unblock learning in the vision layers — including drastically increasing the learning rate — but without success. 

While the base code suggests that fine-tuning vision layers **should** be possible, in practice I found it was not working in this setup. The following evaluation loss curve confirms that no learning occurred — the loss remained constant throughout training. To avoid wasting compute resources, I stopped the run early:

![Eval Loss: vision_only](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/eval_loss_vision_only.png)

Additionally, training was noticeably faster in this run compared to the previous `lang_only` experiment — suggesting that the language layers (which contain the bulk of the model’s parameters) remained frozen, and only the small vision layers were being processed:

![Eval Samples per Second: vision_only](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/eval_speed_vision_only.png)

---

### Third Run: `lang_vision`

At this point, it was clear that only language layers were being effectively trained. In this `lang_vision` run — where both language and vision layers were selected — I expected results similar to `lang_only`.

Indeed, the evaluation loss curve confirmed this expectation, showing nearly identical behavior to `lang_only`:

![Eval Loss: vision_lang](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/eval_loss_vision_lang.png)

Once this was clear, I again stopped training early to conserve resources, and proceeded to test new approaches.

### Fourth Run: `lang_table_only`

This experiment modified the following component:

| Parameter                     | Value                           |
| ----------------------------- | ------------------------------- |
| Output Format                 | "&lt;table&gt; ... &lt;/table&gt;" |

The goal of this run was to train the model to predict **only the table content**, without any surrounding HTML wrapper code. This approach could help improve learning — by removing unnecessary tokens — and also align the training behavior more closely with Pix2Struct’s model.

Additionally, by stripping out the wrapper HTML, the target sequences became shorter — which allowed longer and more complex tables to fit within the model’s context window. This change could also improve the model’s ability to generalize to larger or more detailed tables.

Let’s look at the **evaluation loss** compared to the first run:

![Eval Loss: lang_table_only](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/eval_loss_lang_table_only.png)

At first glance, the higher evaluation loss might seem counterintuitive. However, there’s a clear explanation: the wrapper HTML code is trivial for the model to learn — since it tends to be nearly identical across many training samples. These repetitive tokens reduce cross-entropy loss, artificially lowering the average loss in earlier runs. By removing them, the model now focuses entirely on the more challenging and variable table content — resulting in a higher but more meaningful loss value.

Now, let’s see how this change impacted the **HTML Similarity** metric:

| Model                                         | HTML Similarity |
|-----------------------------------------------|-----------------|
| Granite-Vision 2B                             | 0.56            |
| meta-llama/Llama-3.2-90B-Vision               | 0.62            |
| Fine-tuned Granite-Vision 2B (`lang_only`)     | 0.74            |
| Fine-tuned Granite-Vision 2B (`lang_table_only`)| 0.74            |
| KennethTM/pix2struct-base-table2html          | **0.75**        |

In this first test, we observe **no significant gain or degradation** from using this new output format. It is possible that the model would need **more epochs** or **larger training samples** to fully adapt to this new format. Another idea is to **update the prompt** — so that from the very first step the model understands it should focus solely on table content, rather than having to infer this behavior through training alone. 

This will be explored in the next experiments.


### Fifth / Sixth Run: `lang_table_unsloth`, `vision_table_unsloth`

In these experiments, I explored the following components:

| Parameter                     | Value                           |
| ----------------------------- | ------------------------------- |
| `layers_to_tune`              | `"language_model"`              |
| `layers_to_tune`              | `"vision_model"`                |
| Training Framework            | Unsloth |

At this point, I discovered the promising [Unsloth](https://raw.githubusercontent.com/unslothai/unsloth) framework — which claims to offer **2x faster training** with up to **70% lower memory usage**. Of course, I wanted to test whether it could accelerate my workflow.

My first idea was to leverage the improved memory handling to run longer sequences (`max_seq_length=2048`), but in my case this quickly led to **Out of Memory (OOM)** errors — so I reverted to my previous configuration.

The training speed improvements, however, were undeniable — almost **4x faster** than my earlier runs:

![Eval Speed: lang_table_unsloth](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/eval_speed_unsloth.png)

Unfortunately, this came at a clear **cost to loss performance**:

![Eval Loss: lang_table_unsloth](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/eval_loss_lang_unsloth.png)

Given this noticeable drop in quality, I paused the experiment to investigate further — particularly to see if **Unsloth would allow me to train vision layers**, which is one of its advertised advantages. However, I encountered exactly the same behavior as with HuggingFace Transformers — no actual learning in vision layers.

With these results in mind, I decided to **set aside Unsloth for this project** and continue using **HuggingFace Transformers**, which had shown more reliable learning in earlier runs.

### Seventh Run: `lang_table_only_2`

Here are the new parameters for this run:

| Parameter                     | Value                           |
| ----------------------------- | ------------------------------- |
| `layers_to_tune`              | "language_model"                |
| `gradient_accumulation_steps` | 8                               |
| Training Framework            | HuggingFace Transformers        |

Going back to the previous configuration, I wanted to analyze the impact of a **larger virtual batch size** (via higher `gradient_accumulation_steps`). 

The results were promising — the **evaluation loss** became smoother and trended closer to the original `lang_only` run, even though the model was now predicting **only the table content**:

![Eval Loss: lang_table_only_2](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/eval_loss_lang_table_only_2.png)

Based on this positive result, I decided to keep this `gradient_accumulation_steps=8` setting for the final experiment.

Evaluating this model on **HTML Similarity** resulted in a small but meaningful improvement — finally reaching parity with Pix2Struct:

| Model                                         | HTML Similarity |
|-----------------------------------------------|-----------------|
| Granite-Vision 2B                             | 0.56            |
| meta-llama/Llama-3.2-90B-Vision               | 0.62            |
| Fine-tuned Granite-Vision 2B (`lang_only`)     | 0.74           |
| Fine-tuned Granite-Vision 2B (`lang_table_only`)| 0.74          |
| Fine-tuned Granite-Vision 2B (`lang_table_only_2`)| **0.75**    |
| KennethTM/pix2struct-base-table2html          | **0.75**        |

Naturally, the goal is not just to match Pix2Struct — but to surpass it. Two important levers remained to explore: dataset size and prompt.

### Eighth Run: `lang_table_only_3`

The updated parameters for this run were:

| Parameter                     | Value                           |
| ----------------------------- | ------------------------------- |
| `gradient_accumulation_steps` | 4                               |
| `num_train_images`            | 10,000                           |
| Prompt                        | "Convert table to HTML (&lt;table&gt; ...&lt;/table&gt;)" |

I accidentally reverted `gradient_accumulation_steps` back to **4** in this run, only realizing it once the training was nearly complete — but this actually gave me an extra-chance to observe its effect on learning.

The main goal here was to **double the training size** (to 10K images) and to test the updated, clearer **prompt** format. Unfortunately, a random **CUDA error** caused training to halt around **80% completion** — but even so, the improvement was clear:

![Eval Loss: lang_table_only_3](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/eval_loss_lang_table_only_3.png)

As expected, some smoothness was lost due to the smaller virtual batch size, but the new **prompt** proved very effective — noticeably boosting model learning.

This set the stage perfectly for the **final experiment**, using this improved prompt, 10K training samples, and restoring `gradient_accumulation_steps` to 8.

### Final Run: `lang_table_only_4`

The final set of parameters are:

| Parameter                     | Value                           |
| ----------------------------- | ------------------------------- |
| `layers_to_tune`              | "language_model"                |
| `gradient_accumulation_steps` | 8                               |
| `num_train_images`            | 10000                            |
| Prompt                        | "Convert table to HTML (&lt;table&gt; ...&lt;/table&gt;)"  |
| Output Format                 | "&lt;table&gt; ... &lt;/table&gt;" |
| Training Framework            | HuggingFace Transformers        |

The evaluation loss for this final run:

![Eval Loss: lang_table_only_4](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/eval_loss_lang_table_only_4.png)

As expected, restoring the `gradient_accumulation_steps` to 8 smoothed the loss curve, reducing spikes and achieving slightly lower overall loss values. With a **full epoch** of training on 10K images, this became the best-performing model across all experiments.

Now, let's look at the final results on the **HTML Similarity** metric:

![HTML Similarity: Final Results](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/final_results.png)

The goal of this project was achieved — the fine-tuned model now **surpasses both reference models** on this task. Looking back at the original Granite-Vision 2B, the LoRA fine-tuning improved performance to **0.77**, a **+21 percentage point gain** — all accomplished in under **8 hours** on a **consumer-grade GPU**.

## Qualitative Results

To better illustrate how much the model improved through fine-tuning, let's look at a specific example: **Image ID 618932**.

![Test Image: 618932](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/test_img.png)

This table is particularly tricky — under the `Kappa` column there are **sub-headers** (`Present study` and `King et al. 2001`). These complex layouts typically challenge generic VLMs, especially when they haven’t been exposed to enough similar examples during training. Models can usually **understand** these sub-headers and **answer questions** about them, but generating the full table structure in HTML often requires further prompt tuning and specialized fine-tuning.

Let’s first see how a *base*, non-fine-tuned Granite-Vision 2B model performs on this task:

### Baseline: Raw Granite-Vision 2B

The model can answer questions based on the table correctly:

```python
prompt='What is the Kappa value for the question "Do you communicate with this power?" in the present study?'
res = predict(sample['image'], prompt=prompt)
print(res)
```
**Out[1]:**
```
74
```
However, when asked to generate the full HTML table, the model struggles:

```python
prompt = "Convert table to HTML (<table> ... </table>)"
html = predict(sample['image'], prompt=prompt)
html = '<table>' + html + '</table>' if '<table>' not in html else html
display(HTML(html))
```
**Out[2]:**

![Test Image Output 1: Raw Granite-Vision](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/test_img_output_1.png)

And the **HTML Similarity** metrics for this attempt:

```text
Style similarity: 1.0000
Structural similarity: 0.4091
Lev-Edit Distance: 0.1434
Final HTML Similarity Score: 0.3619
```
### Fine-Tuned Model: `lang_table_only_4`
Now, let's try the exact same test using the fine-tuned model:
```python
from src.models.granite_vision.transformers_library import LLM as granite_vision

model = granite_vision(
    model_path,
    adapter='lang_table_only_4'
)
```

**Out[4]:**

```text
Model loaded
Adapter 'lang_table_only_4' loaded
Adapter 'lang_table_only_4' merged
Using cuda: NVIDIA GeForce RTX 4070 Ti SUPER
```
And the same prediction prompt:
```python
prompt = "Convert table to HTML (<table> ... </table>)"
html = model.predict(sample['image'], max_new_tokens=1024, query=prompt)
display(HTML(html))
```
**Out[5]:**

![Test Image Output 2: Fine-tuned Granite-Vision](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/test_img_output_2.png)

The fine-tuned model now produces an output that closely matches the ground truth, correctly capturing the table structure and sub-headers — something the base model struggled with.

Final **HTML Similarity** metrics:

```text
Style similarity: 1.0000
Structural similarity: 0.9231
Lev-Edit Distance: 1.0000
Final HTML Similarity Score: 0.9615
```
This example shows a clear quantitative improvement as well: from a score of 0.36 to 0.96 on a complex table structure — confirming that fine-tuning on this specialized task dramatically boosts the model’s capability.

## Inference Speed

One major advantage of using a smaller model — aside from the ability to fine-tune on consumer-grade hardware — is **inference speed**. Even if larger models offer competitive performance, latency and throughput remain key factors, especially in production settings.

Let’s compare the inference speed of the different models:

![Tokens per Second by Model](https://raw.githubusercontent.com/JulioSanchezD/TableVision2html/main/images/model_speed_comparison.png)

As shown in the plot, **Pix2Struct** is by far the fastest model. For some use cases — such as **batch-processing thousands of documents for table extraction** — this speed advantage could translate into **significant time savings** and lower compute costs.

However, the **fine-tuned Granite-Vision 2B** achieves a good balance when the amount of documents to process is not massive, having a superior accuracy on this specialized task and reasonably fast inference without the need for extremely large compute infrastructure.

## Conclusions

This project demonstrated that with **LoRA-based fine-tuning** and a **targeted task** (table extraction → HTML), a **small vision-language model (Granite-Vision 2B)** can **outperform much larger models** — even Meta’s 90B LLaMA Vision — while requiring only a **consumer GPU** and less than a day of training.

A few key takeaways:

- **Small, specialized models matter** — you don’t always need 70B+ models to solve specific problems with high accuracy.
- **Parameter-efficient fine-tuning (LoRA)** is a game-changer: adapting large foundation models becomes accessible for most practitioners.
- **Prompt design and training targets** have a big influence — small changes (like switching to `lang_table_only` or refining the prompt) directly impacted performance.
- Having a **custom metric (HTML Similarity)** was critical to track meaningful progress beyond generic text-based metrics.
- Smaller models not only train faster, but also **infer faster** — ideal for production pipelines with high volume.

Finally — and maybe most importantly — this type of experimentation shows that **you can move fast and iterate** even with limited hardware. Fine-tuning powerful open models and adapting them to real-world tasks is **not reserved to big labs anymore**.

I hope this encourages other AI engineers to experiment with small VLMs and fine-tuning techniques for their own projects and solutions — and to see that powerful results are possible even without massive compute budgets!


## What’s Next?

There are definitely some interesting follow-up ideas that can be explored next:

- **Prompt engineering refinements**: Final tests (while writing this blog) showed that separating prompts into *system message* (defining behavior) and *user message* (providing task instructions) significantly improved the base model’s performance. Applying this strategy during fine-tuning could further enhance the model’s ability to consistently generate accurate HTML. This will be tested in upcoming experiments.

- **Training vision layers**: Currently, only the language layers are fine-tuned, as training the vision layers through text-only loss proved ineffective. A more advanced approach could involve adding an auxiliary vision loss — for example, contrastive learning between vision outputs and HTML structure — to better adapt the vision backbone for table extraction tasks.

- **Improved generalization**: The current model is fine-tuned on a single dataset. Expanding training to include more diverse document layouts, table styles, and noisy OCR scenarios could improve robustness and transferability to real-world data.


## Links

- [GitHub Repository](https://github.com/JulioSanchezD/TableVision2html)  
- [HuggingFace Model](https://huggingface.co/JulioSnchezD/granite-vision-3.2-2b-table2html)  

---

If you liked this post, feel free to reach out or share your own experiments!
