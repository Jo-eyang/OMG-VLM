<div align="center">
    
# OMG-VLM

### One Model, Many Graphs: Learning over Attributed Graphs across Heterogeneous Modalities with Vision-Language Models

**已被 EMNLP 2026 Main Conference 接收**

Jiayi Yang<sup>†</sup> · Yifang Chen<sup>†</sup> · Yuanfu Sun · Jiajin Liu · Qiaoyu Tan

<sup>†</sup> 同等贡献

[![论文](https://img.shields.io/badge/arXiv-2607.19128-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.19128)
[![代码](https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&logo=github)](https://github.com/Jo-eyang/OMG-VLM)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Qwen-VL](https://img.shields.io/badge/Backbone-Qwen--VL-6B5BFF?style=flat-square)

[English](../README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md)

<br>

[概览](#概览) · [快速开始](#快速开始) · [数据格式](#数据格式) · [训练](#训练) · [评测](#评测) · [引用](#引用)

<br>

<img width="900" alt="OMG-VLM 框架概览" src="OMG_figure2.jpg" />

<p><em>使用同一个视觉语言主干统一学习文本属性图、图像属性图和多模态属性图。</em></p>

</div>

## 概览

**OMG-VLM** 是一个面向异构属性图学习的统一视觉语言框架。它基于 **Qwen-VL** 构建，并引入图感知适配器，将邻域结构信息直接注入 VLM 原生的图像和文本嵌入空间。

| 项目 | 说明 |
| --- | --- |
| 任务类型 | 面向图像、文本或图文混合节点属性的属性图学习 |
| 模型主干 | Qwen-VL / Qwen-VL-Chat 风格的因果 VLM |
| 图模块 | 图感知视觉适配器与目标感知文本聚合 |
| 训练方式 | 监督微调，支持 LoRA 与 DeepSpeed ZeRO-2 |

## 核心特性

- **一个模型，多类图：** 共享的 VLM 主干可处理图像、文本或图文混合属性图。
- **图感知视觉适配器：** 中心节点图像特征在 Qwen-VL 嵌入空间中融合压缩后的邻居图像特征。
- **目标感知文本聚合：** 目标节点文本检索邻居文本，并压缩为可学习的上下文 token。
- **可复现实验接口：** 训练和评测均使用 conversation 格式样本，并显式传入邻居文件和文本属性文件。


## 文件结构

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

主要组件：

- `omg_vlm/image.py`：图感知图像模块，包括 `GraphAwareVisualAdapter`、`PerNeighborVisualCompressor` 和 `CenterConditionedVisualFusionLayer`。
- `omg_vlm/text.py`：目标感知文本聚合模块，用于从文本属性中检索邻域上下文。
- `Qwen_VL_Chat/`：Qwen-VL 主干、tokenizer、生成工具和模型集成代码。
- `train_omg_vlm.py`：监督微调入口。
- `evaluate_omg_vlm.py`：评测入口，输出 JSONL 格式预测结果。
- `ds_config_zero2.json`：训练脚本使用的 DeepSpeed ZeRO-2 配置。
- `assets/`：集中存放仓库资源，包括图示和本地化 README 文件。

## 快速开始

> 推荐使用 Python 3.9+ 和支持 CUDA 的运行环境。

克隆仓库并创建独立环境：

```bash
git clone https://github.com/Jo-eyang/OMG-VLM.git
cd OMG-VLM
conda create -n omg-vlm python=3.9 -y
conda activate omg-vlm
pip install -r requirements.txt
```

请单独下载 [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) 基座权重，并将其放在 git 仓库之外：

```text
/path/to/Qwen_VL_Chat
```

按照[数据格式](#数据格式)准备 conversation、邻居和文本属性 JSON 文件，然后使用[训练](#训练)和[评测](#评测)中的命令。

## 数据格式

图数据使用 conversation 格式。文本节点写作 `<text>node_id</text>`，图像节点写作 `<img>/path/to/image.jpg</img>`。

conversation 样例：

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

邻居文件：

```json
{
  "node_id": ["neighbor_1", "neighbor_2"]
}
```

文本属性文件：

```json
{
  "node_id": "raw node text"
}
```

`neighbor_data_path` 和 `text_info_path` 也支持用逗号分隔的多个 JSON 路径。此时，每条样本可以通过 `dataset_idx` 选择对应数据集的邻居字典和文本属性字典。

## 训练

运行监督微调：

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

常用参数：

- `--max_neighbors`：每个节点注入的最大图邻居数量。
- `--use_lora`：启用 PEFT LoRA 微调。
- `--train_only_visual_adapter`：仅训练图感知视觉模块。
- `--use_visual_compressor`：将每个邻居图像压缩为可学习的视觉 queries。
- `--textual_aggregation_context_tokens`：文本邻居聚合生成的 `<nbr>` 上下文 token 数量。

## 评测

生成预测结果：

```bash
python evaluate_omg_vlm.py \
  --model_name_or_path /path/to/Qwen_VL_Chat \
  --adapter_path outputs/omg_vlm \
  --data_path /path/to/test.json \
  --neighbor_data_path /path/to/neighbors.json \
  --text_info_path /path/to/text_info.json \
  --output_path outputs/omg_vlm/test_predictions.jsonl
```

评测脚本会为每条样本写入一个 JSON 对象，包括生成预测、可选的目标答案以及可选的 exact-match 分数。

## 输出

训练会将图感知模块与基座 checkpoint 分开保存：

```text
outputs/omg_vlm/
|-- visual_adapter.pth
|-- textual_aggregation.pth
`-- ...
```

启用 LoRA 时，Hugging Face/PEFT 训练工具会将 LoRA adapter 保存到同一输出目录。

## 引用

如果 OMG-VLM 对您的研究有帮助，请引用我们的论文：

```bibtex
@inproceedings{yang2026one,
  title={One Model, Many Graphs: Learning over Attributed Graphs across Heterogeneous Modalities with Vision-Language Models},
  author={Yang, Jiayi and Chen, Yifang and Sun, Yuanfu and Liu, Jiajin and Tan, Qiaoyu},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026}
}
```

## 致谢

本代码库基于 Qwen-VL 以及 VLM/LLM 生态中的常用开源工具构建。感谢 Qwen-VL、Hugging Face Transformers、PEFT、FastChat、DeepSpeed 及相关项目的作者和维护者。
