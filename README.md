<div align="center">
    
# OMG-VLM: One Model, Many Graphs with Vision-Language Models

<img width="1000" alt="OMG_figure2" src="assets/OMG_figure2.jpg" />

[English](README.md) | [简体中文](assets/README.zh-CN.md) | [日本語](assets/README.ja.md) | [한국어](assets/README.ko.md) | [Español](assets/README.es.md) | [Français](assets/README.fr.md)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Graph--Aware%20VLM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Qwen-VL](https://img.shields.io/badge/Backbone-Qwen--VL-6B5BFF?style=for-the-badge)
![DeepSpeed](https://img.shields.io/badge/Training-DeepSpeed%20ZeRO--2-1F7A8C?style=for-the-badge)

</div>

**OMG-VLM** is a unified vision-language framework for attributed graph learning under heterogeneous modality schemas. It builds on **Qwen-VL** and adds graph-aware adapters that inject neighborhood signals into the VLM-native embedding space for both image and text attributes.

## Project Snapshot

| Item | Description |
| --- | --- |
| Task family | Attributed graph learning with image, text, or mixed node attributes |
| Backbone | Qwen-VL / Qwen-VL-Chat style causal VLM |
| Graph modules | Graph-aware visual adapter and target-aware textual aggregation |
| Training style | Supervised fine-tuning with optional LoRA and DeepSpeed ZeRO-2 |

## Table of Contents

- [Note to Reviewers](#note-to-reviewers)
- [Highlights](#highlights)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Data Format](#data-format)
- [Training](#training)
- [Evaluation](#evaluation)
- [Outputs](#outputs)
- [Release Plan](#release-plan)
- [Acknowledgment](#acknowledgment)

## Note to Reviewers

Thank you for taking the time to review our work. This repository is prepared to make the implementation of OMG-VLM inspectable during the anonymous review period. It focuses on the method-specific modules and reproducible training/evaluation interfaces. The complete processed datasets will be released after paper acceptance.

During review, the repository exposes:

- the OMG-VLM model architecture and graph-adapter implementation;
- the Qwen-VL integration interface;
- supervised fine-tuning and evaluation entry points;
- prediction output and metric computation logic.

## Highlights

- **One model, many graphs:** a shared VLM backbone supports graph examples with image, text, or mixed attribute schemas.
- **Graph-aware visual adapter:** center image features attend to compressed neighbor visual features in the Qwen-VL embedding space.
- **Target-aware textual aggregation:** target node text retrieves and compresses neighbor text into learnable context tokens.
- **Reproducible interface:** training and evaluation use conversation-format graph examples with explicit neighbor and text-attribute files.


## File Structure

```text
OMG-VLM/
|-- README.md
|-- requirements.txt
|-- ds_config_zero2.json
|-- assets/
|   |-- OMG_figure2.jpg
|   |-- README.zh-CN.md
|   |-- README.ja.md
|   |-- README.ko.md
|   |-- README.es.md
|   `-- README.fr.md
|-- train_omg_vlm.py
|-- evaluate_omg_vlm.py
|-- omg_vlm/
|   |-- __init__.py
|   |-- image.py
|   `-- text.py
`-- Qwen_VL_Chat/
    |-- __init__.py
    |-- config.json
    |-- configuration_qwen.py
    |-- generation_config.json
    |-- modeling_qwen.py
    |-- qwen_generation_utils.py
    |-- qwen.tiktoken
    |-- tokenization_qwen.py
    |-- tokenizer_config.json
    `-- visual.py
```

Key components:

- `omg_vlm/image.py`: graph-aware image modules, including `GraphAwareVisualAdapter`, `PerNeighborVisualCompressor`, and `CenterConditionedVisualFusionLayer`.
- `omg_vlm/text.py`: target-aware textual aggregation modules for retrieving neighborhood context from text attributes.
- `Qwen_VL_Chat/`: Qwen-VL backbone, tokenizer, generation utilities, and model integration code.
- `train_omg_vlm.py`: supervised fine-tuning entry point.
- `evaluate_omg_vlm.py`: evaluation entry point that writes JSONL predictions.
- `ds_config_zero2.json`: DeepSpeed ZeRO-2 configuration used by the training script.
- `assets/`: centralized repository assets, including figures and localized README files.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Download the Qwen-VL-Chat base weights separately and keep them outside git:

```text
/path/to/Qwen_VL_Chat
```

## Data Format

Prepare graph data in conversation format. Text nodes are represented as `<text>node_id</text>`, and image nodes are represented as `<img>/path/to/image.jpg</img>`.

Conversation example:

```json
[
  {
    "id": "example_0",
    "dataset_idx": 0,
    "conversations": [
      {
        "from": "user",
        "value": "Classify the node: <text>node_0</text>"
      },
      {
        "from": "assistant",
        "value": "label_name"
      }
    ]
  }
]
```

Neighbor file:

```json
{
  "node_id": ["neighbor_1", "neighbor_2"]
}
```

Text-attribute file:

```json
{
  "node_id": "raw node text"
}
```

`neighbor_data_path` and `text_info_path` may also be comma-separated JSON paths. In that case, each example can use `dataset_idx` to select the corresponding dataset-specific neighbor and text-attribute dictionary.

## Training

Run supervised fine-tuning:

```bash
torchrun --nproc_per_node 8 train_omg_vlm.py \
  --model_name_or_path /path/to/Qwen_VL_Chat \
  --data_path /path/to/train.json \
  --eval_data_path /path/to/valid.json \
  --neighbor_data_path /path/to/neighbors.json \
  --text_info_path /path/to/text_info.json \
  --output_dir outputs/omg_vlm \
  --model_max_length 8192 \
  --use_lora true \
  --train_only_visual_adapter true \
  --visual_adapter_num_layers 1 \
  --use_visual_compressor true \
  --visual_adapter_num_queries 32 \
  --visual_adapter_num_heads 32 \
  --textual_aggregation_context_tokens 16
```

Useful options:

- `--max_neighbors`: maximum number of graph neighbors injected per node.
- `--use_lora`: enable PEFT LoRA fine-tuning.
- `--train_only_visual_adapter`: train only graph-aware visual modules.
- `--use_visual_compressor`: compress each neighbor image into learnable visual queries.
- `--textual_aggregation_context_tokens`: number of `<nbr>` context tokens produced for text-neighbor aggregation.

## Evaluation

Generate predictions:

```bash
python evaluate_omg_vlm.py \
  --model_name_or_path /path/to/Qwen_VL_Chat \
  --adapter_path outputs/omg_vlm \
  --data_path /path/to/test.json \
  --neighbor_data_path /path/to/neighbors.json \
  --text_info_path /path/to/text_info.json \
  --output_path outputs/omg_vlm/test_predictions.jsonl
```

The evaluation script writes one JSON object per example, including the generated prediction, optional target answer, and optional exact-match score.

## Outputs

Training saves the graph-aware modules separately from the base checkpoint:

```text
outputs/omg_vlm/
|-- visual_adapter.pth
|-- textual_aggregation.pth
`-- ...
```

When LoRA is enabled, the LoRA adapter is saved in the same output directory by the Hugging Face/PEFT training utilities.

## Release Plan

- Method-specific implementation: available in this repository.
- Training and evaluation scripts: available in this repository.
- Complete processed datasets: to be released after paper acceptance.

## Acknowledgment

This codebase builds on Qwen-VL and widely used open-source tooling from the VLM/LLM ecosystem. We thank the authors and maintainers of Qwen-VL, Hugging Face Transformers, PEFT, FastChat, DeepSpeed, and related projects for making this research possible!
