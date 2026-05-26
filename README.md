# OMG-VLM

Anonymous code release for **One Model, Many Graphs: Learning over Attributed Graphs across Heterogeneous Modalities with Vision-Language Models**.

OMG-VLM adapts a pretrained vision-language model for attributed graph learning across text-attributed graphs, image-attributed graphs, and multimodal-attributed graphs. The implementation uses Qwen-VL as the backbone and adds graph-aware image and text modules in the VLM embedding space.

## Note to Reviewers

Thank you very much for reviewing our submission. This repository has been prepared as an anonymous research-code release: model checkpoints, datasets, logs, local paths, cluster job files, and temporary scripts are intentionally excluded. The base Qwen-VL weights should be downloaded separately. This repository contains the method implementation, training entry point, evaluation entry point, and configuration files needed to inspect and reproduce the submitted approach once the required data and weights are available.

## File Structure

```text
OMG-VLM/
|-- README.md
|-- requirements.txt
|-- ds_config_zero2.json
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
- `omg_vlm/text.py`: target-aware textual aggregation for retrieving neighborhood context from text attributes.
- `Qwen_VL_Chat/`: Qwen-VL backbone, tokenizer, generation utilities, and model integration code.
- `train_omg_vlm.py`: supervised fine-tuning entry point.
- `evaluate_omg_vlm.py`: evaluation entry point that writes JSONL predictions.
- `ds_config_zero2.json`: DeepSpeed ZeRO-2 configuration used by the training script.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the Qwen-VL-Chat base weights separately and keep them outside git, for example:

```text
/path/to/Qwen_VL_Chat
```

Prepare graph data in conversation format. Text nodes are represented as `<text>node_id</text>`, and image nodes are represented as `<img>/path/to/image.jpg</img>`.

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

Train:

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

Evaluate:

```bash
python evaluate_omg_vlm.py \
  --model_name_or_path /path/to/Qwen_VL_Chat \
  --adapter_path outputs/omg_vlm \
  --data_path /path/to/test.json \
  --neighbor_data_path /path/to/neighbors.json \
  --text_info_path /path/to/text_info.json \
  --output_path outputs/omg_vlm/test_predictions.jsonl \
  --max_new_tokens 64
```

The evaluation script writes one JSON object per example, including the generated prediction, optional target answer, and optional exact-match score.

## Repository Scope

This release is intentionally lightweight. It includes the OMG-VLM modules and the Qwen-VL integration needed to run training and inference, but excludes:

- pretrained or fine-tuned model weights;
- raw or processed datasets;
- experiment outputs, logs, and cached features;
- machine-specific paths and cluster launch scripts.

These files should be stored outside the repository and referenced through command-line arguments.

## Outputs

Training saves the graph-aware modules separately from the base checkpoint:

```text
outputs/omg_vlm/
|-- visual_adapter.pth
|-- textual_aggregation.pth
`-- ...
```

When LoRA is enabled, the LoRA adapter is saved in the same output directory by the Hugging Face/PEFT training utilities.

## Acknowledgment

This codebase builds on Qwen-VL and widely used open-source tooling from the VLM/LLM ecosystem. We thank the authors and maintainers of Qwen-VL, Hugging Face Transformers, PEFT, FastChat, DeepSpeed, and related projects for making this research possible.
