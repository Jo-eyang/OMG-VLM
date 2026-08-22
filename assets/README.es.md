<div align="center">
    
# OMG-VLM

### One Model, Many Graphs: Learning over Attributed Graphs across Heterogeneous Modalities with Vision-Language Models

**Aceptado en la conferencia principal de EMNLP 2026**

Jiayi Yang<sup>†</sup> · Yifang Chen<sup>†</sup> · Yuanfu Sun · Jiajin Liu · Qiaoyu Tan

<sup>†</sup> Contribución igual

[![Artículo](https://img.shields.io/badge/arXiv-2607.19128-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.19128)
[![Código](https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&logo=github)](https://github.com/Jo-eyang/OMG-VLM)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Qwen-VL](https://img.shields.io/badge/Backbone-Qwen--VL-6B5BFF?style=flat-square)

[English](../README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md)

<br>

[Resumen](#resumen) · [Inicio rápido](#inicio-rápido) · [Formato de datos](#formato-de-datos) · [Entrenamiento](#entrenamiento) · [Evaluación](#evaluación) · [Citación](#citación)

<br>

<img width="900" alt="Descripción general del framework OMG-VLM" src="OMG_figure2.jpg" />

<p><em>Un único backbone de visión-lenguaje para grafos con atributos textuales, visuales y multimodales.</em></p>

</div>

## Resumen

**OMG-VLM** es un marco unificado de visión-lenguaje para aprendizaje en grafos atribuidos con esquemas de modalidad heterogéneos. Está construido sobre **Qwen-VL** y añade adaptadores conscientes de la estructura del grafo que inyectan señales de vecindad en el espacio de embeddings nativo del VLM, tanto para atributos de imagen como de texto.

| Elemento | Descripción |
| --- | --- |
| Familia de tareas | Aprendizaje en grafos atribuidos con atributos de imagen, texto o mixtos |
| Backbone | VLM causal estilo Qwen-VL / Qwen-VL-Chat |
| Módulos de grafo | Adaptador visual consciente del grafo y agregación textual consciente del objetivo |
| Entrenamiento | Ajuste fino supervisado con LoRA opcional y DeepSpeed ZeRO-2 |

## Características principales

- **Un modelo, muchos grafos:** un backbone VLM compartido admite ejemplos de grafos con atributos de imagen, texto o esquemas mixtos.
- **Adaptador visual consciente del grafo:** las características de la imagen central atienden a características visuales vecinas comprimidas dentro del espacio de embeddings de Qwen-VL.
- **Agregación textual consciente del objetivo:** el texto del nodo objetivo recupera y comprime texto vecino en tokens de contexto aprendibles.
- **Interfaz reproducible:** el entrenamiento y la evaluación usan ejemplos de grafo en formato conversation, con archivos explícitos de vecinos y atributos textuales.


## Estructura de archivos

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

Componentes clave:

- `omg_vlm/image.py`: módulos de imagen conscientes del grafo, incluidos `GraphAwareVisualAdapter`, `PerNeighborVisualCompressor` y `CenterConditionedVisualFusionLayer`.
- `omg_vlm/text.py`: módulos de agregación textual conscientes del objetivo para recuperar contexto de vecindad desde atributos textuales.
- `Qwen_VL_Chat/`: backbone Qwen-VL, tokenizer, utilidades de generación y código de integración del modelo.
- `train_omg_vlm.py`: punto de entrada para ajuste fino supervisado.
- `evaluate_omg_vlm.py`: punto de entrada de evaluación que escribe predicciones JSONL.
- `ds_config_zero2.json`: configuración DeepSpeed ZeRO-2 usada por el script de entrenamiento.
- `assets/`: directorio centralizado para recursos del repositorio, incluidas figuras y archivos README localizados.

## Inicio rápido

> Se recomienda Python 3.9+ y un entorno compatible con CUDA.

Clona el repositorio y crea un entorno aislado:

```bash
git clone https://github.com/Jo-eyang/OMG-VLM.git
cd OMG-VLM
conda create -n omg-vlm python=3.9 -y
conda activate omg-vlm
pip install -r requirements.txt
```

Descarga por separado los pesos base de [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) y mantenlos fuera de git:

```text
/path/to/Qwen_VL_Chat
```

Prepara los archivos JSON de conversation, vecinos y atributos textuales como se describe en [Formato de datos](#formato-de-datos), y utiliza después los comandos de [Entrenamiento](#entrenamiento) y [Evaluación](#evaluación).

## Formato de datos

Prepara los datos de grafo en formato conversation. Los nodos de texto se representan como `<text>node_id</text>` y los nodos de imagen como `<img>/path/to/image.jpg</img>`.

Ejemplo conversation:

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

Archivo de vecinos:

```json
{
  "node_id": ["neighbor_1", "neighbor_2"]
}
```

Archivo de atributos textuales:

```json
{
  "node_id": "raw node text"
}
```

`neighbor_data_path` y `text_info_path` también pueden ser rutas JSON separadas por comas. En ese caso, cada ejemplo puede usar `dataset_idx` para seleccionar el diccionario de vecinos y el diccionario de atributos textuales correspondientes al conjunto de datos.

## Entrenamiento

Ejecuta ajuste fino supervisado:

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

Opciones útiles:

- `--max_neighbors`: número máximo de vecinos de grafo inyectados por nodo.
- `--use_lora`: habilita ajuste fino PEFT LoRA.
- `--train_only_visual_adapter`: entrena solo los módulos visuales conscientes del grafo.
- `--use_visual_compressor`: comprime cada imagen vecina en queries visuales aprendibles.
- `--textual_aggregation_context_tokens`: número de tokens de contexto `<nbr>` producidos para agregación de vecinos textuales.

## Evaluación

Genera predicciones:

```bash
python evaluate_omg_vlm.py \
  --model_name_or_path /path/to/Qwen_VL_Chat \
  --adapter_path outputs/omg_vlm \
  --data_path /path/to/test.json \
  --neighbor_data_path /path/to/neighbors.json \
  --text_info_path /path/to/text_info.json \
  --output_path outputs/omg_vlm/test_predictions.jsonl
```

El script de evaluación escribe un objeto JSON por ejemplo, incluyendo la predicción generada, la respuesta objetivo opcional y la puntuación exact-match opcional.

## Salidas

El entrenamiento guarda los módulos conscientes del grafo por separado del checkpoint base:

```text
outputs/omg_vlm/
|-- visual_adapter.pth
|-- textual_aggregation.pth
`-- ...
```

Cuando LoRA está habilitado, las utilidades de Hugging Face/PEFT guardan el adaptador LoRA en el mismo directorio de salida.

## Citación

Si OMG-VLM resulta útil para tu investigación, cita nuestro artículo:

```bibtex
@inproceedings{yang2026one,
  title={One Model, Many Graphs: Learning over Attributed Graphs across Heterogeneous Modalities with Vision-Language Models},
  author={Yang, Jiayi and Chen, Yifang and Sun, Yuanfu and Liu, Jiajin and Tan, Qiaoyu},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026}
}
```

## Agradecimientos

Este código se basa en Qwen-VL y en herramientas de código abierto ampliamente usadas en el ecosistema VLM/LLM. Agradecemos a los autores y mantenedores de Qwen-VL, Hugging Face Transformers, PEFT, FastChat, DeepSpeed y proyectos relacionados.
