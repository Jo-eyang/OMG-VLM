<div align="center">
    
# OMG-VLM: One Model, Many Graphs with Vision-Language Models

<img width="1000" alt="OMG_figure2" src="OMG_figure2.jpg" />

[English](../README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Graph--Aware%20VLM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Qwen-VL](https://img.shields.io/badge/Backbone-Qwen--VL-6B5BFF?style=for-the-badge)
![DeepSpeed](https://img.shields.io/badge/Training-DeepSpeed%20ZeRO--2-1F7A8C?style=for-the-badge)

</div>

**OMG-VLM** es un marco unificado de visión-lenguaje para aprendizaje en grafos atribuidos con esquemas de modalidad heterogéneos. Está construido sobre **Qwen-VL** y añade adaptadores conscientes de la estructura del grafo que inyectan señales de vecindad en el espacio de embeddings nativo del VLM, tanto para atributos de imagen como de texto.

## Resumen del proyecto

| Elemento | Descripción |
| --- | --- |
| Familia de tareas | Aprendizaje en grafos atribuidos con atributos de imagen, texto o mixtos |
| Backbone | VLM causal estilo Qwen-VL / Qwen-VL-Chat |
| Módulos de grafo | Adaptador visual consciente del grafo y agregación textual consciente del objetivo |
| Entrenamiento | Ajuste fino supervisado con LoRA opcional y DeepSpeed ZeRO-2 |

## Índice

- [Nota para revisores](#nota-para-revisores)
- [Características principales](#características-principales)
- [Estructura de archivos](#estructura-de-archivos)
- [Instalación](#instalación)
- [Formato de datos](#formato-de-datos)
- [Entrenamiento](#entrenamiento)
- [Evaluación](#evaluación)
- [Salidas](#salidas)
- [Plan de publicación](#plan-de-publicación)
- [Agradecimientos](#agradecimientos)

## Nota para revisores

Gracias por revisar nuestro trabajo. Este repositorio está preparado para que la implementación de OMG-VLM sea inspeccionable durante el periodo de revisión anónima. Se centra en los módulos específicos del método y en interfaces reproducibles de entrenamiento y evaluación. Los conjuntos de datos procesados completos se publicarán después de la aceptación del artículo.

Durante la revisión, el repositorio expone:

- la arquitectura de OMG-VLM y la implementación de los adaptadores de grafo;
- la interfaz de integración con Qwen-VL;
- puntos de entrada para ajuste fino supervisado y evaluación;
- lógica de salida de predicciones y cálculo de métricas.

## Características principales

- **Un modelo, muchos grafos:** un backbone VLM compartido admite ejemplos de grafos con atributos de imagen, texto o esquemas mixtos.
- **Adaptador visual consciente del grafo:** las características de la imagen central atienden a características visuales vecinas comprimidas dentro del espacio de embeddings de Qwen-VL.
- **Agregación textual consciente del objetivo:** el texto del nodo objetivo recupera y comprime texto vecino en tokens de contexto aprendibles.
- **Interfaz reproducible:** el entrenamiento y la evaluación usan ejemplos de grafo en formato conversation, con archivos explícitos de vecinos y atributos textuales.


## Estructura de archivos

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

Componentes clave:

- `omg_vlm/image.py`: módulos de imagen conscientes del grafo, incluidos `GraphAwareVisualAdapter`, `PerNeighborVisualCompressor` y `CenterConditionedVisualFusionLayer`.
- `omg_vlm/text.py`: módulos de agregación textual conscientes del objetivo para recuperar contexto de vecindad desde atributos textuales.
- `Qwen_VL_Chat/`: backbone Qwen-VL, tokenizer, utilidades de generación y código de integración del modelo.
- `train_omg_vlm.py`: punto de entrada para ajuste fino supervisado.
- `evaluate_omg_vlm.py`: punto de entrada de evaluación que escribe predicciones JSONL.
- `ds_config_zero2.json`: configuración DeepSpeed ZeRO-2 usada por el script de entrenamiento.
- `assets/`: directorio centralizado para recursos del repositorio, incluidas figuras y archivos README localizados.

## Instalación

Instala las dependencias de Python:

```bash
pip install -r requirements.txt
```

Descarga por separado los pesos base de Qwen-VL-Chat y mantenlos fuera de git:

```text
/path/to/Qwen_VL_Chat
```

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

## Plan de publicación

- Implementación específica del método: disponible en este repositorio.
- Scripts de entrenamiento y evaluación: disponibles en este repositorio.
- Conjuntos de datos procesados completos: se publicarán después de la aceptación del artículo.

## Agradecimientos

Este código se basa en Qwen-VL y en herramientas de código abierto ampliamente usadas en el ecosistema VLM/LLM. Agradecemos a los autores y mantenedores de Qwen-VL, Hugging Face Transformers, PEFT, FastChat, DeepSpeed y proyectos relacionados.
