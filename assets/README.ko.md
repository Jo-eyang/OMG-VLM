<div align="center">
    
# OMG-VLM

### One Model, Many Graphs: Learning over Attributed Graphs across Heterogeneous Modalities with Vision-Language Models

**EMNLP 2026 Main Conference 채택 논문**

Jiayi Yang · Yifang Chen · Yuanfu Sun · Jiajin Liu · Qiaoyu Tan

[![논문](https://img.shields.io/badge/arXiv-2607.19128-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.19128)
[![코드](https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&logo=github)](https://github.com/Jo-eyang/OMG-VLM)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Qwen-VL](https://img.shields.io/badge/Backbone-Qwen--VL-6B5BFF?style=flat-square)

[English](../README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md)

<br>

[개요](#개요) · [빠른 시작](#빠른-시작) · [데이터 형식](#데이터-형식) · [학습](#학습) · [평가](#평가) · [인용](#인용)

<br>

<img width="900" alt="OMG-VLM 프레임워크 개요" src="OMG_figure2.jpg" />

<p><em>하나의 비전-언어 백본으로 텍스트 속성, 이미지 속성, 멀티모달 속성 그래프를 통합 학습합니다.</em></p>

</div>

## 개요

**OMG-VLM**은 이질적인 모달리티 스키마를 가진 속성 그래프 학습을 위한 통합 비전-언어 프레임워크입니다. **Qwen-VL**을 기반으로 하며, 이미지와 텍스트 속성 모두에 대해 이웃 구조 정보를 VLM 고유 임베딩 공간에 주입하는 그래프 인식 어댑터를 추가합니다.

| 항목 | 설명 |
| --- | --- |
| 작업 범주 | 이미지, 텍스트 또는 혼합 노드 속성을 가진 속성 그래프 학습 |
| 백본 | Qwen-VL / Qwen-VL-Chat 스타일 causal VLM |
| 그래프 모듈 | 그래프 인식 비주얼 어댑터와 타깃 인식 텍스트 집계 |
| 학습 방식 | LoRA 및 DeepSpeed ZeRO-2를 지원하는 지도 미세조정 |

## 주요 특징

- **하나의 모델, 다양한 그래프:** 공유 VLM 백본으로 이미지, 텍스트, 또는 이미지-텍스트 혼합 속성 스키마의 그래프 예제를 처리합니다.
- **그래프 인식 비주얼 어댑터:** 중심 이미지 특징이 Qwen-VL 임베딩 공간에서 압축된 이웃 이미지 특징에 어텐션합니다.
- **타깃 인식 텍스트 집계:** 타깃 노드 텍스트가 이웃 텍스트를 검색하고 이를 학습 가능한 컨텍스트 token으로 압축합니다.
- **재현 가능한 인터페이스:** 학습과 평가는 conversation 형식의 그래프 예제와 명시적인 이웃 파일 및 텍스트 속성 파일을 사용합니다.


## 파일 구조

<details>
<summary>저장소 구조 펼치기</summary>

```text
OMG-VLM/
|-- README.md
|-- CITATION.cff
|-- NOTICE
|-- QWEN_LICENSE
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

</details>

주요 구성 요소:

- `omg_vlm/image.py`: `GraphAwareVisualAdapter`, `PerNeighborVisualCompressor`, `CenterConditionedVisualFusionLayer`를 포함한 그래프 인식 이미지 모듈.
- `omg_vlm/text.py`: 텍스트 속성에서 이웃 컨텍스트를 검색하는 타깃 인식 텍스트 집계 모듈.
- `Qwen_VL_Chat/`: Qwen-VL 백본, tokenizer, 생성 유틸리티, 모델 통합 코드.
- `train_omg_vlm.py`: 지도 미세조정 엔트리 포인트.
- `evaluate_omg_vlm.py`: JSONL 예측을 작성하는 평가 엔트리 포인트.
- `ds_config_zero2.json`: 학습 스크립트에서 사용하는 DeepSpeed ZeRO-2 설정.
- `assets/`: 그림과 현지화된 README 파일을 한곳에 모아 둔 저장소 자산 디렉터리.

## 빠른 시작

> Python 3.9 이상과 CUDA 지원 환경을 권장합니다.

저장소를 복제하고 독립된 환경을 만듭니다.

```bash
git clone https://github.com/Jo-eyang/OMG-VLM.git
cd OMG-VLM
conda create -n omg-vlm python=3.9 -y
conda activate omg-vlm
pip install -r requirements.txt
```

[Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) 베이스 가중치는 별도로 다운로드하고 git 저장소 밖에 보관하세요.

```text
/path/to/Qwen_VL_Chat
```

[데이터 형식](#데이터-형식)에 따라 conversation, 이웃 및 텍스트 속성 JSON 파일을 준비한 다음 [학습](#학습)과 [평가](#평가)의 명령을 사용하세요.

## 데이터 형식

그래프 데이터는 conversation 형식으로 준비합니다. 텍스트 노드는 `<text>node_id</text>`, 이미지 노드는 `<img>/path/to/image.jpg</img>`로 표현합니다.

conversation 예시:

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

이웃 파일:

```json
{
  "node_id": ["neighbor_1", "neighbor_2"]
}
```

텍스트 속성 파일:

```json
{
  "node_id": "raw node text"
}
```

`neighbor_data_path`와 `text_info_path`는 쉼표로 구분된 여러 JSON 경로도 지원합니다. 이 경우 각 예제는 `dataset_idx`를 사용해 해당 데이터셋의 이웃 사전과 텍스트 속성 사전을 선택할 수 있습니다.

## 학습

지도 미세조정을 실행합니다.

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

주요 옵션:

- `--max_neighbors`: 각 노드에 주입할 그래프 이웃의 최대 개수.
- `--use_lora`: PEFT LoRA 미세조정 활성화.
- `--train_only_visual_adapter`: 그래프 인식 비주얼 모듈만 학습.
- `--use_visual_compressor`: 각 이웃 이미지를 학습 가능한 비주얼 query로 압축.
- `--textual_aggregation_context_tokens`: 텍스트 이웃 집계에서 생성되는 `<nbr>` 컨텍스트 token 수.

## 평가

예측을 생성합니다.

```bash
python evaluate_omg_vlm.py \
  --model_name_or_path /path/to/Qwen_VL_Chat \
  --adapter_path outputs/omg_vlm \
  --data_path /path/to/test.json \
  --neighbor_data_path /path/to/neighbors.json \
  --text_info_path /path/to/text_info.json \
  --output_path outputs/omg_vlm/test_predictions.jsonl
```

평가 스크립트는 각 예제마다 생성된 예측, 선택적 정답, 선택적 exact-match 점수를 포함한 JSON 객체를 한 줄씩 기록합니다.

## 출력

학습은 그래프 인식 모듈을 베이스 checkpoint와 별도로 저장합니다.

```text
outputs/omg_vlm/
|-- visual_adapter.pth
|-- textual_aggregation.pth
`-- ...
```

LoRA를 활성화하면 Hugging Face/PEFT 학습 유틸리티가 LoRA adapter도 같은 출력 디렉터리에 저장합니다.

## 인용

OMG-VLM이 연구에 도움이 되었다면 다음 논문을 인용해 주세요.

```bibtex
@inproceedings{yang2026one,
  title={One Model, Many Graphs: Learning over Attributed Graphs across Heterogeneous Modalities with Vision-Language Models},
  author={Yang, Jiayi and Chen, Yifang and Sun, Yuanfu and Liu, Jiajin and Tan, Qiaoyu},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026}
}
```

## 감사의 말

이 코드베이스는 Qwen-VL과 VLM/LLM 생태계에서 널리 사용되는 오픈소스 도구를 기반으로 합니다. Qwen-VL, Hugging Face Transformers, PEFT, FastChat, DeepSpeed 및 관련 프로젝트의 저자와 유지보수자에게 감사드립니다.
