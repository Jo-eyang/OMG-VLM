import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
import transformers

from train_omg_vlm import (
    NEIGHBOR_INFO_LITERAL,
    load_neighbor_data,
    load_text_info,
    process_image_tags_with_neighbors,
    process_text_tags_with_neighbors,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OMG-VLM on conversation-format graph examples.")
    parser.add_argument("--model_name_or_path", required=True, help="Path to the Qwen-VL/OMG-VLM base model directory.")
    parser.add_argument("--adapter_path", default=None, help="Optional fine-tuned adapter/checkpoint directory.")
    parser.add_argument("--data_path", required=True, help="Evaluation conversation JSON file.")
    parser.add_argument("--output_path", required=True, help="Path to write JSONL predictions.")
    parser.add_argument("--neighbor_data_path", default=None, help="Neighbor dictionary JSON, or comma-separated JSON paths.")
    parser.add_argument("--text_info_path", default=None, help="Text attribute dictionary JSON, or comma-separated JSON paths.")
    parser.add_argument("--max_neighbors", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--visual_adapter_num_layers", type=int, default=1)
    parser.add_argument("--visual_adapter_num_queries", type=int, default=32)
    parser.add_argument("--visual_adapter_num_heads", type=int, default=32)
    parser.add_argument("--use_visual_compressor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--textual_aggregation_num_heads", type=int, default=16)
    parser.add_argument("--textual_aggregation_pool_layers", type=int, default=1)
    parser.add_argument("--textual_aggregation_pool_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--textual_aggregation_context_tokens", type=int, default=16)
    parser.add_argument("--textual_aggregation_dropout", type=float, default=0.1)
    return parser.parse_args()


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected evaluation data to be a list of conversation examples.")
    return data


def first_turns(example: Dict[str, Any]) -> tuple[str, Optional[str]]:
    user_query = None
    target = None
    for turn in example.get("conversations", []):
        role = turn.get("from")
        value = turn.get("value", "")
        if user_query is None and role == "user":
            user_query = value
        elif user_query is not None and role == "assistant":
            target = value
            break
    if user_query is None:
        raise ValueError("Each example must contain at least one user turn.")
    return user_query, target


def prepare_query(
    query: str,
    tokenizer,
    neighbor_data: Dict[str, Any],
    text_info_data: Dict[str, Any],
    max_neighbors: int,
    textual_aggregation_context_tokens: int,
) -> tuple[str, List[List[List[int]]]]:
    query = process_image_tags_with_neighbors(query, neighbor_data, max_neighbors=max_neighbors)
    sample_neighbor_ids: List[List[List[int]]] = []
    if neighbor_data and text_info_data:
        query = process_text_tags_with_neighbors(
            query,
            neighbor_data,
            text_info_data,
            max_neighbors=max_neighbors,
            neighbor_ids_collector=sample_neighbor_ids,
            tokenizer=tokenizer,
            nbr_placeholder_count=textual_aggregation_context_tokens,
        )
    return query, [sample_neighbor_ids]


def configure_model(config, args):
    config.use_cache = False
    config.visual_adapter_num_layers = args.visual_adapter_num_layers
    config.use_visual_compressor = args.use_visual_compressor
    config.visual_adapter_num_queries = args.visual_adapter_num_queries
    config.visual_adapter_num_heads = args.visual_adapter_num_heads
    config.textual_aggregation_num_heads = args.textual_aggregation_num_heads
    config.textual_aggregation_pool_layers = args.textual_aggregation_pool_layers
    config.textual_aggregation_pool_mlp_ratio = args.textual_aggregation_pool_mlp_ratio
    config.textual_aggregation_context_tokens = args.textual_aggregation_context_tokens
    config.textual_aggregation_dropout = args.textual_aggregation_dropout
    return config


def configure_special_tokens(model, tokenizer):
    if not hasattr(model.config, "visual"):
        model.config.visual = {}
    model.config.visual["nbr_token_id"] = getattr(tokenizer, "nbr_id", tokenizer.convert_tokens_to_ids("<nbr>"))
    model.config.visual["text_start_id"] = tokenizer.convert_tokens_to_ids("<text>")
    model.config.visual["text_end_id"] = tokenizer.convert_tokens_to_ids("</text>")
    neighbor_info_ids = tokenizer(
        NEIGHBOR_INFO_LITERAL,
        add_special_tokens=False,
        return_attention_mask=False,
    ).input_ids
    model.config.visual["neighbor_info_token_ids"] = neighbor_info_ids
    model.config.visual["neighbor_info_token_length"] = len(neighbor_info_ids)
    model.config.visual["neighbor_info_string"] = NEIGHBOR_INFO_LITERAL


def load_optional_checkpoint(model, adapter_path: Optional[str]):
    if not adapter_path:
        return model

    try:
        from peft import PeftModel

        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, adapter_path)
    except ImportError:
        pass

    transformer = model.transformer
    visual_adapter_path = os.path.join(adapter_path, "visual_adapter.pth")
    if os.path.exists(visual_adapter_path):
        transformer.visual_adapter.load_state_dict(torch.load(visual_adapter_path, map_location="cpu"))

    textual_aggregation_path = os.path.join(adapter_path, "textual_aggregation.pth")
    if os.path.exists(textual_aggregation_path):
        transformer.textual_aggregation.load_state_dict(torch.load(textual_aggregation_path, map_location="cpu"))

    return model


def main():
    args = parse_args()
    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    config = configure_model(config, args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        device_map=args.device_map,
    )
    configure_special_tokens(model, tokenizer)
    model = load_optional_checkpoint(model, args.adapter_path)
    model.eval()

    neighbor_data_list = load_neighbor_data(args.neighbor_data_path)
    text_info_data_list = load_text_info(args.text_info_path)
    examples = load_json(args.data_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    exact_matches = 0
    counted = 0

    with open(args.output_path, "w", encoding="utf-8") as out:
        for idx, example in enumerate(examples):
            dataset_idx = example.get("dataset_idx", 0)
            neighbor_data = neighbor_data_list[dataset_idx] if dataset_idx < len(neighbor_data_list) else {}
            text_info_data = text_info_data_list[dataset_idx] if dataset_idx < len(text_info_data_list) else {}
            query, target = first_turns(example)
            query, text_neighbor_ids = prepare_query(
                query,
                tokenizer,
                neighbor_data,
                text_info_data,
                max_neighbors=args.max_neighbors,
                textual_aggregation_context_tokens=args.textual_aggregation_context_tokens,
            )
            response, _ = model.chat(
                tokenizer,
                query=query,
                history=None,
                max_new_tokens=args.max_new_tokens,
                text_neighbor_ids=text_neighbor_ids,
            )
            record = {
                "index": idx,
                "prediction": response,
                "target": target,
            }
            if target is not None:
                counted += 1
                is_match = response.strip() == target.strip()
                exact_matches += int(is_match)
                record["exact_match"] = is_match
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    if counted:
        print(f"Exact match: {exact_matches / counted:.4f} ({exact_matches}/{counted})")
    print(f"Wrote predictions to {args.output_path}")


if __name__ == "__main__":
    main()
