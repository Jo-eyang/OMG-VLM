<div align="center">
    
# OMG-VLM

### One Model, Many Graphs: Learning over Attributed Graphs across Heterogeneous Modalities with Vision-Language Models

**Accepted at EMNLP 2026 · Main Conference**

<p>Jiayi Yang<sup>†</sup> · Yifang Chen<sup>†</sup> · Yuanfu Sun · Jiajin Liu · Qiaoyu Tan<br>
<sub><em>† Equal contribution</em></sub></p>

[![Paper](https://img.shields.io/badge/arXiv-2607.19128-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.19128)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&logo=github)](https://github.com/Jo-eyang/OMG-VLM)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E?style=flat-square)](https://huggingface.co/oofwite/OMG-VLM)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Qwen-VL](https://img.shields.io/badge/Backbone-Qwen--VL-6B5BFF?style=flat-square)

[English](README.md) | [简体中文](assets/README.zh-CN.md) | [日本語](assets/README.ja.md) | [한국어](assets/README.ko.md) | [Español](assets/README.es.md) | [Français](assets/README.fr.md)

<br>

[Overview](#overview) · [Quick Start](#quick-start) · [Data Format](#data-format) · [Training](#training) · [Evaluation](#evaluation) · [Citation](#citation)

<br>

<img width="900" alt="Overview of the OMG-VLM framework" src="assets/OMG_figure2.jpg" />

<p><em>One shared vision-language backbone for text-attributed, image-attributed, and multimodal graphs.</em></p>

</div>

## Overview

**OMG-VLM** is a unified vision-language framework for attributed graph learning under heterogeneous modality schemas. It builds on **Qwen-VL** and adds graph-aware adapters that inject neighborhood signals into the VLM-native embedding space for both image and text attributes.

| Item | Description |
| --- | --- |
| Task family | Attributed graph learning with image, text, or mixed node attributes |
| Backbone | Qwen-VL / Qwen-VL-Chat style causal VLM |
| Graph modules | Graph-aware visual adapter and target-aware textual aggregation |
| Training style | Supervised fine-tuning with optional LoRA and DeepSpeed ZeRO-2 |

## Highlights

- **One model, many graphs:** a shared VLM backbone supports graph examples with image, text, or mixed attribute schemas.
- **Graph-aware visual adapter:** center image features attend to compressed neighbor visual features in the Qwen-VL embedding space.
- **Target-aware textual aggregation:** target node text retrieves and compresses neighbor text into learnable context tokens.
- **Reproducible interface:** training and evaluation use conversation-format graph examples with explicit neighbor and text-attribute files.


## File Structure

```text
OMG-VLM/
|-- README.md
|-- CITATION.cff
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
    |-- LICENSE
    |-- NOTICE
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

## Quick Start

> Python 3.9+ and a CUDA-capable environment are recommended.

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/Jo-eyang/OMG-VLM.git
cd OMG-VLM
conda create -n omg-vlm python=3.9 -y
conda activate omg-vlm
pip install -r requirements.txt
```

Download the [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) base weights separately and keep them outside git:

```text
/path/to/Qwen_VL_Chat
```

Prepare the conversation, neighbor, and text-attribute JSON files as described in [Data Format](#data-format), then use the commands in [Training](#training) and [Evaluation](#evaluation).

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

## Citation

If you find OMG-VLM useful, please cite our paper:

```bibtex
@inproceedings{yang2026one,
  title={One Model, Many Graphs: Learning over Attributed Graphs across Heterogeneous Modalities with Vision-Language Models},
  author={Yang, Jiayi and Chen, Yifang and Sun, Yuanfu and Liu, Jiajin and Tan, Qiaoyu},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026}
}
```

## Acknowledgment

This codebase builds on Qwen-VL and widely used open-source tooling from the VLM/LLM ecosystem. We thank the authors and maintainers of Qwen-VL, Hugging Face Transformers, PEFT, FastChat, DeepSpeed, and related projects for making this research possible!
