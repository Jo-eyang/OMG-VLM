<div align="center">
    
# OMG-VLM: One Model, Many Graphs with Vision-Language Models

<img width="1000" alt="OMG_figure2" src="OMG_figure2.jpg" />

[English](../README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Graph--Aware%20VLM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Qwen-VL](https://img.shields.io/badge/Backbone-Qwen--VL-6B5BFF?style=for-the-badge)
![DeepSpeed](https://img.shields.io/badge/Training-DeepSpeed%20ZeRO--2-1F7A8C?style=for-the-badge)

</div>

**OMG-VLM** est un cadre vision-langage unifié pour l'apprentissage sur graphes attribués avec des schémas de modalités hétérogènes. Il s'appuie sur **Qwen-VL** et ajoute des adaptateurs sensibles à la structure du graphe, qui injectent les signaux de voisinage dans l'espace d'embedding natif du VLM pour les attributs image et texte.

## Aperçu du projet

| Élément | Description |
| --- | --- |
| Famille de tâches | Apprentissage sur graphes attribués avec attributs image, texte ou mixtes |
| Backbone | VLM causal de style Qwen-VL / Qwen-VL-Chat |
| Modules de graphe | Adaptateur visuel sensible au graphe et agrégation textuelle sensible à la cible |
| Entraînement | Affinage supervisé avec LoRA optionnel et DeepSpeed ZeRO-2 |

## Sommaire

- [Note aux relecteurs](#note-aux-relecteurs)
- [Points forts](#points-forts)
- [Structure des fichiers](#structure-des-fichiers)
- [Installation](#installation)
- [Format des données](#format-des-données)
- [Entraînement](#entraînement)
- [Évaluation](#évaluation)
- [Sorties](#sorties)
- [Plan de publication](#plan-de-publication)
- [Remerciements](#remerciements)

## Note aux relecteurs

Merci de prendre le temps d'examiner notre travail. Ce dépôt est préparé afin de rendre l'implémentation d'OMG-VLM inspectable pendant la période de relecture anonyme. Il se concentre sur les modules propres à la méthode et sur des interfaces reproductibles d'entraînement et d'évaluation. Les jeux de données entièrement traités seront publiés après l'acceptation de l'article.

Pendant la relecture, ce dépôt expose:

- l'architecture du modèle OMG-VLM et l'implémentation des adaptateurs de graphe;
- l'interface d'intégration avec Qwen-VL;
- les points d'entrée pour l'affinage supervisé et l'évaluation;
- la logique de sortie des prédictions et de calcul des métriques.

## Points forts

- **Un modèle, de nombreux graphes:** un backbone VLM partagé prend en charge des exemples de graphes avec attributs image, texte ou mixtes.
- **Adaptateur visuel sensible au graphe:** les caractéristiques de l'image centrale prêtent attention aux caractéristiques visuelles voisines compressées dans l'espace d'embedding de Qwen-VL.
- **Agrégation textuelle sensible à la cible:** le texte du noeud cible récupère et compresse le texte des voisins en tokens de contexte apprenables.
- **Interface reproductible:** l'entraînement et l'évaluation utilisent des exemples de graphe au format conversation, avec des fichiers explicites pour les voisins et les attributs textuels.


## Structure des fichiers

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

Composants principaux:

- `omg_vlm/image.py`: modules image sensibles au graphe, notamment `GraphAwareVisualAdapter`, `PerNeighborVisualCompressor` et `CenterConditionedVisualFusionLayer`.
- `omg_vlm/text.py`: modules d'agrégation textuelle sensibles à la cible pour récupérer le contexte de voisinage depuis les attributs textuels.
- `Qwen_VL_Chat/`: backbone Qwen-VL, tokenizer, utilitaires de génération et code d'intégration du modèle.
- `train_omg_vlm.py`: point d'entrée pour l'affinage supervisé.
- `evaluate_omg_vlm.py`: point d'entrée d'évaluation qui écrit des prédictions JSONL.
- `ds_config_zero2.json`: configuration DeepSpeed ZeRO-2 utilisée par le script d'entraînement.
- `assets/`: répertoire centralisé pour les ressources du dépôt, notamment les figures et les README localisés.

## Installation

Installez les dépendances Python:

```bash
pip install -r requirements.txt
```

Téléchargez séparément les poids de base Qwen-VL-Chat et conservez-les hors de git:

```text
/path/to/Qwen_VL_Chat
```

## Format des données

Préparez les données de graphe au format conversation. Les noeuds texte sont représentés par `<text>node_id</text>` et les noeuds image par `<img>/path/to/image.jpg</img>`.

Exemple conversation:

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

Fichier de voisins:

```json
{
  "node_id": ["neighbor_1", "neighbor_2"]
}
```

Fichier d'attributs textuels:

```json
{
  "node_id": "raw node text"
}
```

`neighbor_data_path` et `text_info_path` peuvent aussi être des chemins JSON séparés par des virgules. Dans ce cas, chaque exemple peut utiliser `dataset_idx` pour sélectionner le dictionnaire de voisins et le dictionnaire d'attributs textuels propres au jeu de données.

## Entraînement

Lancez l'affinage supervisé:

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

Options utiles:

- `--max_neighbors`: nombre maximal de voisins de graphe injectés par noeud.
- `--use_lora`: active l'affinage PEFT LoRA.
- `--train_only_visual_adapter`: entraîne uniquement les modules visuels sensibles au graphe.
- `--use_visual_compressor`: compresse chaque image voisine en queries visuelles apprenables.
- `--textual_aggregation_context_tokens`: nombre de tokens de contexte `<nbr>` produits pour l'agrégation des voisins textuels.

## Évaluation

Générez les prédictions:

```bash
python evaluate_omg_vlm.py \
  --model_name_or_path /path/to/Qwen_VL_Chat \
  --adapter_path outputs/omg_vlm \
  --data_path /path/to/test.json \
  --neighbor_data_path /path/to/neighbors.json \
  --text_info_path /path/to/text_info.json \
  --output_path outputs/omg_vlm/test_predictions.jsonl
```

Le script d'évaluation écrit un objet JSON par exemple, incluant la prédiction générée, la réponse cible facultative et le score exact-match facultatif.

## Sorties

L'entraînement sauvegarde les modules sensibles au graphe séparément du checkpoint de base:

```text
outputs/omg_vlm/
|-- visual_adapter.pth
|-- textual_aggregation.pth
`-- ...
```

Lorsque LoRA est activé, les utilitaires Hugging Face/PEFT sauvegardent l'adaptateur LoRA dans le même répertoire de sortie.

## Plan de publication

- Implémentation propre à la méthode: disponible dans ce dépôt.
- Scripts d'entraînement et d'évaluation: disponibles dans ce dépôt.
- Jeux de données entièrement traités: publication prévue après l'acceptation de l'article.

## Remerciements

Ce code s'appuie sur Qwen-VL et sur des outils open source largement utilisés dans l'écosystème VLM/LLM. Nous remercions les auteurs et mainteneurs de Qwen-VL, Hugging Face Transformers, PEFT, FastChat, DeepSpeed et des projets associés.
