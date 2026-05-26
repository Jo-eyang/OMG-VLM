# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, GPTQConfig
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer import TrainerCallback  # For custom callback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
NEIGHBOR_INFO_LITERAL = " Neighbor Text Info:"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    neighbor_data_path: str = field(
        default=None, metadata={"help": "Path to the neighbor image data JSON file."}
    )
    text_info_path: str = field(
        default=None, metadata={"help": "Path to the text info JSON file (node_id -> raw_text mapping)."}
    )
    max_neighbors: int = field(
        default=10, metadata={"help": "Maximum number of neighbor IDs to inject per image."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    train_only_visual_adapter: bool = False
    freeze_textual_aggregation: bool = False
    freeze_image_processor: bool = False
    max_grad_norm: float = 1.0
    run_name: Optional[str] = field(default=None, metadata={"help": "Name for the wandb run"})
    # graph-aware visual adapter specific arguments
    visual_adapter_num_layers: int = field(default=1, metadata={"help": "Number of graph-aware visual adapter layers"})
    use_visual_compressor: bool = field(default=True, metadata={"help": "Whether to use per-neighbor visual compressor"})
    visual_adapter_num_queries: int = field(default=32, metadata={"help": "Number of graph-aware visual adapter neighbor queries"})
    visual_adapter_num_heads: int = field(default=32, metadata={"help": "Number of graph-aware visual adapter attention heads"})
    # TargetAwareTextualAggregation arguments
    textual_aggregation_num_heads: int = field(default=16, metadata={"help": "Number of attention heads in TargetAwareTextualAggregation cross-attention"})
    textual_aggregation_pool_layers: int = field(default=1, metadata={"help": "Number of Transformer encoder layers used for pooling the target text"})
    textual_aggregation_pool_mlp_ratio: float = field(default=4.0, metadata={"help": "MLP expansion ratio inside the pooling encoder"})
    textual_aggregation_context_tokens: int = field(default=16, metadata={"help": "Number of context embedding vectors produced by textual aggregation"})
    textual_aggregation_dropout: float = field(default=0.1, metadata={"help": "Dropout probability for TargetAwareTextualAggregation"})


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["attn.c_attn", "attn.c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def load_neighbor_data(neighbor_data_path: str) -> List[Dict]:
    """Load neighbor image data from JSON file(s). Supports comma-separated paths.
    
    Returns a list of dicts (one per dataset) to maintain dataset isolation.
    """
    if not neighbor_data_path:
        rank0_print("No neighbor data provided")
        return []
    
    # Support comma-separated multiple files
    paths = [p.strip() for p in neighbor_data_path.split(',')]
    data_list = []
    
    for idx, path in enumerate(paths):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            data_list.append(data)
            rank0_print(f"Loaded neighbor data [{idx}] from {path} with {len(data)} entries")
        else:
            rank0_print(f"Warning: neighbor data file not found: {path}")
            data_list.append({})  # Empty dict as placeholder
    
    rank0_print(f"Total neighbor data files loaded: {len(data_list)}")
    return data_list


def load_text_info(text_info_path: str) -> List[Dict]:
    """Load text info data (node_id -> raw_text mapping) from JSON file(s). Supports comma-separated paths.
    
    Returns a list of dicts (one per dataset) to maintain dataset isolation.
    """
    if not text_info_path:
        rank0_print("No text info provided")
        return []
    
    # Support comma-separated multiple files
    paths = [p.strip() for p in text_info_path.split(',')]
    data_list = []
    
    for idx, path in enumerate(paths):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            data_list.append(data)
            rank0_print(f"Loaded text info [{idx}] from {path} with {len(data)} entries")
        else:
            rank0_print(f"Warning: text info file not found: {path}")
            data_list.append({})  # Empty dict as placeholder
    
    rank0_print(f"Total text info files loaded: {len(data_list)}")
    return data_list


def get_neighbor_images(item_id: str, neighbor_data: Dict, max_neighbors: int = 10) -> List[str]:
    """Get neighbor image paths for a given item ID."""
    if str(item_id) in neighbor_data:
        neighbors = neighbor_data[str(item_id)][:max_neighbors]
        return neighbors
    return []


def resolve_textual_aggregation_module(transformer) -> Optional[torch.nn.Module]:
    """Return the active TargetAwareTextualAggregation module."""
    if transformer is None:
        return None
    if hasattr(transformer, 'textual_aggregation') and transformer.textual_aggregation is not None:
        return transformer.textual_aggregation
    return None


def process_image_tags_with_neighbors(text: str, neighbor_data: Dict, max_neighbors: int = 10) -> str:
    """Embed ONLY neighbor IDs (not full paths) into <img> tag to keep length < IMG_TOKEN_SPAN.

    Format produced (if neighbors exist):
        <img>full_main_image_path|id1,id2,id3</img>
    If inserting the id list would make raw byte length > 240 (safety margin <256), we skip neighbors.
    """
    import re
    import os
    image_pattern = r'<img>([^<]+)</img>'
    matches = re.findall(image_pattern, text)
    if not matches or not neighbor_data:
        return text
    modified = text
    for img_path in matches:
        filename = os.path.basename(img_path)
        item_id = os.path.splitext(filename)[0]
        raw_neighbors = neighbor_data.get(str(item_id), [])[:max_neighbors]
        # Normalize neighbors to plain IDs (strip path and extension)
        neighbor_ids = []
        for n in raw_neighbors:
            n_str = str(n)
            base = os.path.basename(n_str)
            base = os.path.splitext(base)[0]
            if base and base != item_id:
                neighbor_ids.append(base)
        if not neighbor_ids:
            continue
        # Compose candidate string main|id1,id2
        neighbor_id_str = ",".join(map(str, neighbor_ids))
        candidate_inner = f"{img_path}|{neighbor_id_str}"
        # Byte length check (approx to IMG_TOKEN_SPAN since each byte -> one token position in tokenizer scheme)
        # Progressively remove neighbors from the end until within 240-byte limit
        if len(candidate_inner.encode('utf-8')) > 240:  # leave margin for safety
            truncated_ids = neighbor_ids[:]
            while len(truncated_ids) > 0 and len(f"{img_path}|{','.join(map(str, truncated_ids))}".encode('utf-8')) > 240:
                truncated_ids = truncated_ids[:-1]  # Remove last neighbor ID
            if truncated_ids:  # If we have at least some neighbors that fit
                neighbor_id_str = ",".join(map(str, truncated_ids))
                candidate_inner = f"{img_path}|{neighbor_id_str}"
            else:
                continue  # Skip if even no neighbors can fit
        original_tag = f"<img>{img_path}</img>"
        new_tag = f"<img>{candidate_inner}</img>"
        modified = modified.replace(original_tag, new_tag)
    return modified


def process_text_tags_with_neighbors(
    text: str,
    neighbor_data: Dict,
    text_info_data: Dict,
    max_neighbors: int = 10,
    neighbor_ids_collector: List[List[List[int]]] = None,
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
    nbr_placeholder_count: int = 16,
) -> str:
    """Embed raw center text, bridge string, and repeated <nbr> placeholders per text tag.

    Neighbor token IDs are collected so the model can retrieve raw neighbor context later."""
    import re
    import os
    
    text_pattern = r'<text>([^<]+)</text>'
    matches = re.findall(text_pattern, text)
    if not matches:
        return text
    
    # Debug: Print first time processing
    if matches:
        print(f"process_text_tags_with_neighbors found {len(matches)} text nodes: {matches[:3]}...")
    
    modified = text
    for node_id in matches:
        node_id = node_id.strip()
        
        # Get raw text for center node
        if str(node_id) in text_info_data:
            center_raw_text = text_info_data[str(node_id)]
        else:
            # If not found, use node_id as-is
            center_raw_text = node_id
        
        raw_neighbors = neighbor_data.get(str(node_id), [])[:max_neighbors] if neighbor_data else []
        neighbor_token_sequences: List[List[int]] = []
        for n in raw_neighbors:
            n_str = str(n).strip()
            if not n_str or n_str == node_id:
                continue

            tokens: List[int] = []
            if tokenizer is not None:
                neighbor_text = text_info_data.get(n_str, n_str)
                tokenized = tokenizer(
                    neighbor_text,
                    add_special_tokens=False,
                    truncation=False,
                    return_attention_mask=False,
                )
                tokens = tokenized.input_ids if hasattr(tokenized, "input_ids") else []
                if not tokens and tokenizer.eod_id is not None:
                    tokens = [tokenizer.eod_id]
            neighbor_token_sequences.append(tokens)

        if neighbor_ids_collector is not None:
            neighbor_ids_collector.append(neighbor_token_sequences)

        # Reserve multiple <nbr> placeholders (one per query token) after the bridge text
        placeholder_block = "<nbr>" * max(1, nbr_placeholder_count)
        candidate_inner = f"{center_raw_text}{NEIGHBOR_INFO_LITERAL}{placeholder_block}"

        original_tag = f"<text>{node_id}</text>"
        new_tag = f"<text>{candidate_inner}</text>"
        modified = modified.replace(original_tag, new_tag)
    return modified


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant.",
    neighbor_data_list: List[Dict] = None,
    text_info_data_list: List[Dict] = None,
    dataset_idx: int = 0,
    max_neighbors: int = 10,
    nbr_placeholder_count: int = 16,
) -> Dict:
    """Modified to collect and return text_neighbor_ids.
    
    Args:
        neighbor_data_list: List of neighbor dicts, one per dataset
        text_info_data_list: List of text_info dicts, one per dataset
        dataset_idx: Index of which dataset this example belongs to
    """
    # Get the correct dict for this dataset
    neighbor_data = neighbor_data_list[dataset_idx] if neighbor_data_list and dataset_idx < len(neighbor_data_list) else {}
    text_info_data = text_info_data_list[dataset_idx] if text_info_data_list and dataset_idx < len(text_info_data_list) else {}
    
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    all_neighbor_ids = []
    
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        # Collect neighbor IDs for this sample
        sample_neighbor_ids = []  # List of List[str]: [[nid1, nid2], [nid3, nid4, nid5], ...]
        
        # Process each conversation turn with neighbor ID collection
        processed_source = []
        for sentence in source:
            new_sentence = sentence.copy()
            # Process text tags and collect neighbor IDs (using dataset-specific dicts)
            if neighbor_data and text_info_data:
                new_sentence["value"] = process_text_tags_with_neighbors(
                    new_sentence["value"],
                    neighbor_data,
                    text_info_data,
                    max_neighbors=max_neighbors,
                    neighbor_ids_collector=sample_neighbor_ids,
                    tokenizer=tokenizer,
                    nbr_placeholder_count=nbr_placeholder_count,
                )
            processed_source.append(new_sentence)
        
        all_neighbor_ids.append(sample_neighbor_ids)
        
        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(processed_source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        
        # Handle pad_token_id being None (use eod_id as fallback for Qwen tokenizer)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eod_id
        
        input_id += [pad_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    # Handle pad_token_id for attention mask
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eod_id
    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(pad_id),
        text_neighbor_ids=all_neighbor_ids
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        neighbor_data_list: List[Dict] = None,
        text_neighbor_data_list: List[Dict] = None,
        text_info_data_list: List[Dict] = None,
        max_neighbors: int = 10,
        nbr_placeholder_count: int = 16,
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        
        # Store dataset-specific data
        self.neighbor_data_list = neighbor_data_list if neighbor_data_list else []
        self.text_neighbor_data_list = text_neighbor_data_list if text_neighbor_data_list else []
        self.text_info_data_list = text_info_data_list if text_info_data_list else []
        self.max_neighbors = max_neighbors
        self.nbr_placeholder_count = nbr_placeholder_count
        
        # Process conversations - image tags with neighbors
        if self.neighbor_data_list:
            rank0_print("Processing image tags with neighbor IDs (length-safe)...")
            processed_data = []
            for example in raw_data:
                dataset_idx = example.get('dataset_idx', 0)  # Get dataset index
                neighbor_data = self.neighbor_data_list[dataset_idx] if dataset_idx < len(self.neighbor_data_list) else {}
                
                new_convs = []
                for conv in example["conversations"]:
                    new_conv = conv.copy()
                    # Process image tags with dataset-specific neighbor data
                    new_conv["value"] = process_image_tags_with_neighbors(
                        new_conv["value"], neighbor_data, max_neighbors=self.max_neighbors
                    )
                    new_convs.append(new_conv)
                ex = example.copy()
                ex["conversations"] = new_convs
                processed_data.append(ex)
            raw_data = processed_data
        
        # Preprocess each example with its dataset-specific data
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_text_neighbor_ids = []
        
        for example in raw_data:
            dataset_idx = example.get('dataset_idx', 0)
            sources = [example["conversations"]]
            
            data_dict = preprocess(
                sources, 
                tokenizer, 
                max_len,
                neighbor_data_list=self.text_neighbor_data_list,
                text_info_data_list=self.text_info_data_list,
                dataset_idx=dataset_idx,
                max_neighbors=max_neighbors,
                nbr_placeholder_count=self.nbr_placeholder_count,
            )
            
            all_input_ids.append(data_dict["input_ids"][0])
            all_labels.append(data_dict["labels"][0])
            all_attention_mask.append(data_dict["attention_mask"][0])
            all_text_neighbor_ids.append(data_dict["text_neighbor_ids"][0])

        self.input_ids = all_input_ids
        self.labels = all_labels
        self.attention_mask = all_attention_mask
        self.text_neighbor_ids = all_text_neighbor_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            text_neighbor_ids=self.text_neighbor_ids[i]
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        neighbor_data_list: List[Dict] = None,
        text_neighbor_data_list: List[Dict] = None,
        text_info_data_list: List[Dict] = None,
        max_neighbors: int = 10,
        nbr_placeholder_count: int = 16,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.neighbor_data_list = neighbor_data_list if neighbor_data_list else []
        self.text_neighbor_data_list = text_neighbor_data_list if text_neighbor_data_list else []
        self.text_info_data_list = text_info_data_list if text_info_data_list else []
        self.max_neighbors = max_neighbors
        self.nbr_placeholder_count = nbr_placeholder_count

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        example = self.raw_data[i]
        dataset_idx = example.get('dataset_idx', 0)
        conversations = example["conversations"]
        
        # Get dataset-specific neighbor and text_info data
        neighbor_data = self.neighbor_data_list[dataset_idx] if dataset_idx < len(self.neighbor_data_list) else {}
        
        # Process image tags with neighbors if needed
        if neighbor_data:
            new_convs = []
            for conv in conversations:
                new_conv = conv.copy()
                new_conv["value"] = process_image_tags_with_neighbors(
                    new_conv["value"], neighbor_data, max_neighbors=self.max_neighbors
                )
                new_convs.append(new_conv)
            conversations = new_convs

        # Text tag processing happens in preprocess() with neighbor_ids collection
        ret = preprocess(
            [conversations], 
            self.tokenizer, 
            self.max_len,
            neighbor_data_list=self.text_neighbor_data_list,
            text_info_data_list=self.text_info_data_list,
            dataset_idx=dataset_idx,
            max_neighbors=self.max_neighbors,
            nbr_placeholder_count=self.nbr_placeholder_count,
        )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            text_neighbor_ids=ret["text_neighbor_ids"][0]  # 
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_len,
    nbr_placeholder_count: int = 16,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning. Supports multiple datasets via comma-separated paths."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    # Load neighbor data if provided (supports comma-separated paths) - returns List[Dict]
    neighbor_data_list = load_neighbor_data(data_args.neighbor_data_path) if hasattr(data_args, 'neighbor_data_path') else []
    
    # Load text neighbor data (same as neighbor_data for text nodes)
    text_neighbor_data_list = neighbor_data_list if neighbor_data_list else []
    
    # Load text info data if provided (supports comma-separated paths) - returns List[Dict]
    text_info_data_list = load_text_info(data_args.text_info_path) if hasattr(data_args, 'text_info_path') else []

    # Support comma-separated multiple data files
    data_paths = [p.strip() for p in data_args.data_path.split(',')]
    train_json = []
    
    # Load each dataset and tag with dataset_idx to track which neighbor/text_info to use
    for dataset_idx, path in enumerate(data_paths):
        if os.path.exists(path):
            data = json.load(open(path, "r"))
            # Tag each example with its dataset index
            for example in data:
                example['dataset_idx'] = dataset_idx
            train_json.extend(data)
            rank0_print(f"Loaded {len(data)} examples from dataset[{dataset_idx}]: {path}")
        else:
            rank0_print(f"Warning: data file not found: {path}")
    
    rank0_print(f"Total training examples: {len(train_json)}")
    
    train_dataset = dataset_cls(
        train_json, 
        tokenizer=tokenizer, 
        max_len=max_len, 
        neighbor_data_list=neighbor_data_list, 
        text_neighbor_data_list=text_neighbor_data_list,
        text_info_data_list=text_info_data_list,
        max_neighbors=data_args.max_neighbors,
        nbr_placeholder_count=nbr_placeholder_count,
    )

    if data_args.eval_data_path:
        # Support comma-separated eval data paths
        eval_paths = [p.strip() for p in data_args.eval_data_path.split(',')]
        eval_json = []
        for dataset_idx, path in enumerate(eval_paths):
            if os.path.exists(path):
                data = json.load(open(path, "r"))
                # Tag each example with its dataset index
                for example in data:
                    example['dataset_idx'] = dataset_idx
                eval_json.extend(data)
                rank0_print(f"Loaded {len(data)} eval examples from dataset[{dataset_idx}]: {path}")
            else:
                rank0_print(f"Warning: eval data file not found: {path}")
        
        rank0_print(f"Total eval examples: {len(eval_json)}")
        
        eval_dataset = dataset_cls(
            eval_json,
            tokenizer=tokenizer,
            max_len=max_len,
            neighbor_data_list=neighbor_data_list,
            text_neighbor_data_list=text_neighbor_data_list,
            text_info_data_list=text_info_data_list,
            max_neighbors=data_args.max_neighbors,
            nbr_placeholder_count=nbr_placeholder_count,
        )
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, text_info_data_list=text_info_data_list)


class TextGraphDataCollator:
    """Custom collator that properly handles text_neighbor_ids."""
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        # Extract text_neighbor_ids before collation (it's not a tensor)
        text_neighbor_ids = [f.pop("text_neighbor_ids", []) for f in features]
        
        # Standard collation for tensors
        import torch
        from transformers.data.data_collator import pad_without_fast_tokenizer_warning
        
        # Get max length in batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        # Pad sequences
        batch = {}
        for key in features[0].keys():
            if key in ["input_ids", "labels", "attention_mask"]:
                # Determine padding value
                if key == "labels":
                    padding_value = IGNORE_TOKEN_ID
                elif key == "attention_mask":
                    padding_value = 0
                else:
                    # Handle pad_token_id being None (use eod_id as fallback for Qwen tokenizer)
                    padding_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eod_id
                
                # Pad and stack
                padded = []
                for f in features:
                    value = f[key]
                    padding_length = max_length - len(value)
                    if isinstance(value, list):
                        value = torch.tensor(value, dtype=torch.long)
                    padded_value = torch.cat([
                        value,
                        torch.full((padding_length,), padding_value, dtype=value.dtype)
                    ])
                    padded.append(padded_value)
                batch[key] = torch.stack(padded)
        
        # Add text_neighbor_ids back as a list (not tensor)
        batch["text_neighbor_ids"] = text_neighbor_ids
        
        return batch


class GradientLoggingCallback(TrainerCallback):
    """Custom callback to log graph-aware visual adapter gradient norms during training."""
    def __init__(self, model, log_interval=100):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            visual_adapter_grad_norm = 0.0
            num_params = 0
            if hasattr(self.model.transformer, 'visual_adapter'):
                for param in self.model.transformer.visual_adapter.parameters():
                    if param.grad is not None:
                        visual_adapter_grad_norm += param.grad.norm().item() ** 2
                        num_params += 1
            if num_params > 0:
                visual_adapter_grad_norm = (visual_adapter_grad_norm / num_params) ** 0.5
                print(f"Step {state.global_step}: graph-aware visual adapter average grad norm: {visual_adapter_grad_norm:.6f}")
            else:
                print(f"Step {state.global_step}: No graph-aware visual adapter gradients (possibly frozen)")


class GraphAwareVisualAdapterSaveCallback(TrainerCallback):
    """Custom callback to save graph-aware visual adapter and TargetAwareTextualAggregation states during checkpointing."""
    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        transformer = getattr(self.model, 'transformer', None)
        if transformer is None:
            print("Warning: GraphAwareVisualAdapterSaveCallback: transformer not found on model; skipping checkpoint save")
            return

        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        # Save PEFT adapters with checkpoints for easier resume/loading.
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                self.model.save_pretrained(checkpoint_folder)
                print(f"Saved PEFT adapter to {checkpoint_folder}")
        except Exception as e:
            print(f"Warning: skipped saving PEFT adapter at checkpoint due to: {e}")
        
        # Save graph-aware visual adapter state
        if hasattr(transformer, 'visual_adapter'):
            visual_adapter_path = os.path.join(checkpoint_folder, "visual_adapter.pth")
            torch.save(transformer.visual_adapter.state_dict(), visual_adapter_path)
            print(f"Saved graph-aware visual adapter state to {visual_adapter_path}")
        
        # Save TargetAwareTextualAggregation state (full module)
        textual_aggregation_module = resolve_textual_aggregation_module(transformer)
        if textual_aggregation_module is not None:
            textual_aggregation_path = os.path.join(checkpoint_folder, "textual_aggregation.pth")
            torch.save(textual_aggregation_module.state_dict(), textual_aggregation_path)
            print(f"Saved TargetAwareTextualAggregation state to {textual_aggregation_path}")


def check_model_for_nan_inf(model):
    """Check for NaN/Inf in model parameters and clamp if found."""
    has_nan_inf = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN/Inf found in {name}")
            has_nan_inf = True
            # Clamp to safe range
            param.data = torch.clamp(param.data, min=-1e6, max=1e6)
    if has_nan_inf:
        print("Warning: NaN/Inf detected and clamped")
    else:
        print("Model check: No NaN/Inf in parameters")

hook_step = 0
def train():
    global local_rank
    
    # Force HuggingFace to reload trust_remote_code modules (disable caching)
    import os
    os.environ['HF_MODULES_CACHE'] = '/tmp/hf_modules_nocache'
    print(f"Set HF_MODULES_CACHE to disable trust_remote_code caching")
    
    # Clear HuggingFace model code cache to ensure latest code is loaded
    import shutil
    cache_dir = os.path.expanduser("~/.cache/huggingface/modules")
    if os.path.exists(cache_dir):
        print(f"Clearing HuggingFace module cache at {cache_dir}")
        try:
            # Only clear transformers_modules (trust_remote_code cache)
            transformers_cache = os.path.join(cache_dir, "transformers_modules")
            if os.path.exists(transformers_cache):
                shutil.rmtree(transformers_cache)
                print(f"Cleared transformers_modules cache")
        except Exception as e:
            print(f"Warning: Failed to clear cache: {e}")
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # Initialize wandb if report_to includes wandb
    if "wandb" in training_args.report_to:
        try:
            import wandb
            
            # Get run name from training_args or create default
            run_name = getattr(training_args, 'run_name', None)
            if run_name is None:
                # Create default name from output_dir
                run_name = os.path.basename(training_args.output_dir)
            
            # Initialize wandb
            wandb.init(
                project="omg-vlm",
                name=run_name,
                config={
                    "model": model_args.model_name_or_path,
                    "learning_rate": training_args.learning_rate,
                    "epochs": training_args.num_train_epochs,
                    "batch_size": training_args.per_device_train_batch_size,
                    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                    "lora_r": lora_args.lora_r,
                    "lora_alpha": lora_args.lora_alpha,
                    "visual_adapter_num_layers": training_args.visual_adapter_num_layers,
                    "visual_adapter_num_queries": training_args.visual_adapter_num_queries,
                    "visual_adapter_num_heads": training_args.visual_adapter_num_heads,
                    "textual_aggregation_num_heads": training_args.textual_aggregation_num_heads,
                    "textual_aggregation_pool_layers": training_args.textual_aggregation_pool_layers,
                    "textual_aggregation_pool_mlp_ratio": training_args.textual_aggregation_pool_mlp_ratio,
                    "textual_aggregation_context_tokens": training_args.textual_aggregation_context_tokens,
                    "textual_aggregation_dropout": training_args.textual_aggregation_dropout,
                    "max_neighbors": data_args.max_neighbors,
                    "train_only_visual_adapter": training_args.train_only_visual_adapter,
                }
            )
            print(f"Initialized wandb with run name: {run_name}")
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")
            training_args.report_to = [r for r in training_args.report_to if r != "wandb"]
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            training_args.report_to = [r for r in training_args.report_to if r != "wandb"]

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0:
            logging.warning(
                "FSDP is incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    
    # Set graph-aware visual adapter configuration parameters
    config.visual_adapter_num_layers = training_args.visual_adapter_num_layers
    config.use_visual_compressor = training_args.use_visual_compressor
    config.visual_adapter_num_queries = training_args.visual_adapter_num_queries
    config.visual_adapter_num_heads = training_args.visual_adapter_num_heads
    
    # Set TargetAwareTextualAggregation parameters
    config.textual_aggregation_num_heads = training_args.textual_aggregation_num_heads
    config.textual_aggregation_pool_layers = training_args.textual_aggregation_pool_layers
    config.textual_aggregation_pool_mlp_ratio = training_args.textual_aggregation_pool_mlp_ratio
    config.textual_aggregation_context_tokens = training_args.textual_aggregation_context_tokens
    config.textual_aggregation_dropout = training_args.textual_aggregation_dropout
    
    print(f"Training Arguments - graph-aware visual adapter Configuration:")
    print(f"  - visual_adapter_num_layers: {training_args.visual_adapter_num_layers}")
    print(f"  - use_visual_compressor: {training_args.use_visual_compressor}")
    print(f"  - visual_adapter_num_queries: {training_args.visual_adapter_num_queries}")
    print(f"  - visual_adapter_num_heads: {training_args.visual_adapter_num_heads}")
    
    print(f"Training Arguments - TargetAwareTextualAggregation:")
    print(f"  - textual_aggregation_num_heads: {training_args.textual_aggregation_num_heads}")
    print(f"  - textual_aggregation_pool_layers: {training_args.textual_aggregation_pool_layers}")
    print(f"  - textual_aggregation_pool_mlp_ratio: {training_args.textual_aggregation_pool_mlp_ratio}")
    print(f"  - textual_aggregation_context_tokens: {training_args.textual_aggregation_context_tokens}")
    print(f"  - textual_aggregation_dropout: {training_args.textual_aggregation_dropout}")

    # Load model and tokenizer
    # Use local_files_only to ensure we load the latest local code, not cached version
    print(f"Loading model from {model_args.model_name_or_path} (forcing local code)")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,  # Force using local files, not cache
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        local_files_only=True,  # Force using local files
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token_id = tokenizer.eod_id
    
    # <nbr> token is already built into our modified tokenizer
    # Get <nbr> token ID from tokenizer (it should already exist)
    if hasattr(tokenizer, 'nbr_id'):
        nbr_token_id = tokenizer.nbr_id
        print(f"Found built-in <nbr> token with ID: {nbr_token_id}")
    else:
        # Fallback: try to get it from special_tokens
        nbr_token_id = tokenizer.convert_tokens_to_ids('<nbr>')
        print(f"Retrieved <nbr> token ID: {nbr_token_id}")
    
    # Store in model config for later use in forward()
    if not hasattr(model.config, 'visual'):
        model.config.visual = {}
    model.config.visual['nbr_token_id'] = nbr_token_id
    
    # Also store text_start_id and text_end_id for <text> token detection
    text_start_id = tokenizer.convert_tokens_to_ids('<text>')
    text_end_id = tokenizer.convert_tokens_to_ids('</text>')
    model.config.visual['text_start_id'] = text_start_id
    model.config.visual['text_end_id'] = text_end_id
    neighbor_info_ids = tokenizer(
        NEIGHBOR_INFO_LITERAL,
        add_special_tokens=False,
        return_attention_mask=False,
    ).input_ids
    model.config.visual['neighbor_info_token_ids'] = neighbor_info_ids
    model.config.visual['neighbor_info_token_length'] = len(neighbor_info_ids)
    model.config.visual['neighbor_info_string'] = NEIGHBOR_INFO_LITERAL
    
    print(f"Stored special token IDs in model.config.visual:")
    print(f"   - <nbr>: {nbr_token_id}")
    print(f"   - <text>: {text_start_id}")
    print(f"   - </text>: {text_end_id}")
    print(f"   - neighbor bridge tokens: {len(neighbor_info_ids)} from '{NEIGHBOR_INFO_LITERAL}'")

    # Save initial graph-aware visual adapter state for later comparison (to validate training)
    initial_visual_adapter_queries = None
    if hasattr(model.transformer, 'visual_adapter'):
        visual_adapter = model.transformer.visual_adapter
        # Check if GraphAwareVisualAdapter has per-neighbor visual compressor with queries
        if hasattr(visual_adapter, 'visual_compressor') and hasattr(visual_adapter.visual_compressor, 'queries'):
            initial_visual_adapter_queries = visual_adapter.visual_compressor.queries.data.cpu().clone()
            print(f"Saved initial GraphAwareVisualAdapter per-neighbor visual compressor queries shape: {initial_visual_adapter_queries.shape}")
        else:
            print(f"GraphAwareVisualAdapter found but no per-neighbor visual compressor queries to track")
    else:
        print(f"No GraphAwareVisualAdapter found in model.transformer")

    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        
        # Re-set config after PEFT wrapping (must set on base_model!)
        print(f"Re-setting special token IDs after PEFT wrapping")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Has base_model: {hasattr(model, 'base_model')}")
        
        # PEFT wrapper structure: PeftModel -> base_model (original model)
        # Need to set config on the base model that will be accessed in forward()
        if hasattr(model, 'base_model'):
            target_config = model.base_model.config
            print(f"   Setting on: model.base_model.config")
        else:
            target_config = model.config
            print(f"   Setting on: model.config")
        
        # Debug: check current state
        print(f"   Before: hasattr(target_config, 'visual') = {hasattr(target_config, 'visual')}")
        print(f"   Before: type(target_config.visual) = {type(target_config.visual) if hasattr(target_config, 'visual') else 'N/A'}")
        
        # Ensure visual dict exists
        if not hasattr(target_config, 'visual'):
            target_config.visual = {}
            print(f"   Created new visual dict")
        
        # Set the values
        target_config.visual['nbr_token_id'] = nbr_token_id
        target_config.visual['text_start_id'] = text_start_id
        target_config.visual['text_end_id'] = text_end_id
        target_config.visual['neighbor_info_token_ids'] = neighbor_info_ids
        target_config.visual['neighbor_info_token_length'] = len(neighbor_info_ids)
        target_config.visual['neighbor_info_string'] = NEIGHBOR_INFO_LITERAL
        
        print(f"Set special token IDs:")
        print(f"   - nbr_token_id: {nbr_token_id}")
        print(f"   - text_start_id: {text_start_id}")
        print(f"   - text_end_id: {text_end_id}")
        print(f"   - neighbor bridge tokens: {len(neighbor_info_ids)}")
        print(f"   After: 'text_start_id' in target_config.visual = {'text_start_id' in target_config.visual}")
        print(f"   After: target_config.visual.get('text_start_id') = {target_config.visual.get('text_start_id')}")
        print(f"   After: hasattr(target_config.visual, 'text_start_id') = {hasattr(target_config.visual, 'text_start_id')}")
    else:
        # When only the graph-aware visual adapter is trained, add a minimal
        # frozen PEFT wrapper so checkpoints can be loaded uniformly.
        if training_args.train_only_visual_adapter:
            try:
                candidate_patterns = [
                    "attn.c_attn", "attn.c_proj", "w1", "w2",
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]
                present = set()
                for name, _ in model.named_parameters():
                    for pat in candidate_patterns:
                        if pat in name:
                            present.add(pat)
                target_modules = sorted(present) if present else ["q_proj", "v_proj"]

                dummy_cfg = LoraConfig(
                    r=1,
                    lora_alpha=2,
                    target_modules=target_modules,
                    lora_dropout=0.0,
                    bias=lora_args.lora_bias,
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, dummy_cfg)
                # Freeze the dummy PEFT adapter; only the graph-aware visual adapter is trained.
                for n, p in model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = False
                print(f"Added minimal frozen PEFT adapter with targets: {target_modules}")
                
                # Re-set config after PEFT wrapping (must set on base_model!)
                print(f"Re-setting special token IDs after dummy PEFT wrapping")
                base_config = model.base_model.config if hasattr(model, 'base_model') else model.config
                if not hasattr(base_config, 'visual'):
                    base_config.visual = {}
                base_config.visual['nbr_token_id'] = nbr_token_id
                base_config.visual['text_start_id'] = text_start_id
                base_config.visual['text_end_id'] = text_end_id
                base_config.visual['neighbor_info_token_ids'] = neighbor_info_ids
                base_config.visual['neighbor_info_token_length'] = len(neighbor_info_ids)
                base_config.visual['neighbor_info_string'] = NEIGHBOR_INFO_LITERAL
                print(f"Set on base_model.config.visual")
            except Exception as e:
                print(f"Warning: failed to add minimal PEFT adapter: {e}. Proceeding without adapter wrapper.")

    # Freezing logic
    if training_args.train_only_visual_adapter:
        # Freeze the entire model first
        model.requires_grad_(False)
        print("train_only_visual_adapter=True: Froze entire model")
        
        # Unfreeze LoRA adapters (if using LoRA)
        if training_args.use_lora:
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            print("Unfroze LoRA adapter parameters")
        
        # Unfreeze graph-aware visual adapter (unless freeze_image_processor is True)
        if hasattr(model.transformer, 'visual_adapter'):
            if training_args.freeze_image_processor:
                # Keep GraphAwareVisualAdapter frozen for text-only training
                print("freeze_image_processor=True: Keeping GraphAwareVisualAdapter frozen")
            else:
                for param in model.transformer.visual_adapter.parameters():
                    param.requires_grad = True
                print("Unfroze graph-aware visual adapter parameters")
        
        # Unfreeze TargetAwareTextualAggregation module (unless freeze_textual_aggregation is True)
        textual_aggregation_module = resolve_textual_aggregation_module(model.transformer)
        if textual_aggregation_module is not None:
            if training_args.freeze_textual_aggregation:
                print("freeze_textual_aggregation=True: Keeping TargetAwareTextualAggregation frozen")
            else:
                for param in textual_aggregation_module.parameters():
                    param.requires_grad = True
                print("Unfroze TargetAwareTextualAggregation parameters")
        
        # Optionally unfreeze attn_pool in VIT (if it exists and you want it trainable)
        # if hasattr(model.transformer, 'visual') and hasattr(model.transformer.visual, 'attn_pool'):
        #     for param in model.transformer.visual.attn_pool.parameters():
        #         param.requires_grad = True
        #     print("Unfroze VIT attn_pool (optional)")
    else:
        # Standard freezing: Only VIT if fix_vit=True
        # if training_args.fix_vit and hasattr(model, 'transformer') and hasattr(model.transformer, 'visual'):
        #     model.transformer.visual.requires_grad_(False)
        #     if hasattr(model.transformer.visual, 'attn_pool'):
        #         model.transformer.visual.attn_pool.requires_grad_(True)  # If desired
        #     print("fix_vit=True: Froze VIT (except attn_pool if present)")
        
        # Unfreeze ViT if fix_vit=False (PEFT freezes everything by default)
        if not training_args.fix_vit and hasattr(model, 'transformer') and hasattr(model.transformer, 'visual'):
            for param in model.transformer.visual.parameters():
                param.requires_grad = True
            print("fix_vit=False: Unfroze ViT parameters")
        
        # Ensure graph-aware visual adapter trainable (unless freeze_image_processor is True)
        if hasattr(model.transformer, 'visual_adapter'):
            if training_args.freeze_image_processor:
                # Freeze GraphAwareVisualAdapter for text-only training
                for param in model.transformer.visual_adapter.parameters():
                    param.requires_grad = False
                print("freeze_image_processor=True: Froze graph-aware visual adapter parameters")
            else:
                for param in model.transformer.visual_adapter.parameters():
                    param.requires_grad = True
                print("Unfroze graph-aware visual adapter parameters")
        
        # Unfreeze TargetAwareTextualAggregation module (unless freeze_textual_aggregation is True)
        textual_aggregation_module = resolve_textual_aggregation_module(model.transformer)
        if textual_aggregation_module is not None:
            if training_args.freeze_textual_aggregation:
                for param in textual_aggregation_module.parameters():
                    param.requires_grad = False
                print("freeze_textual_aggregation=True: Froze TargetAwareTextualAggregation parameters")
            else:
                for param in textual_aggregation_module.parameters():
                    param.requires_grad = True
                print("Unfroze TargetAwareTextualAggregation parameters")

    # Global counter for logging every N steps


    def visual_adapter_grad_hook(grad):
        global hook_step
        if grad is not None:
            hook_step += 1
            if hook_step % 100 == 0:  # Log every 100 backward calls
                grad_norm = grad.norm().item()
                print(f"During backward (step ~{hook_step//100 * 100}): graph-aware visual adapter grad norm: {grad_norm:.6f}")
        return grad

    if hasattr(model.transformer, 'visual_adapter'):
        visual_adapter = model.transformer.visual_adapter
        if hasattr(visual_adapter, 'visual_compressor') and hasattr(visual_adapter.visual_compressor, 'queries'):
            # Only register hook if queries require gradients (not frozen)
            if visual_adapter.visual_compressor.queries.requires_grad:
                visual_adapter.visual_compressor.queries.register_hook(visual_adapter_grad_hook)
                print(f"Registered gradient hook for GraphAwareVisualAdapter per-neighbor visual compressor queries")
            else:
                print(f"GraphAwareVisualAdapter queries frozen (requires_grad=False), skipping gradient hook")
        else:
            print(f"GraphAwareVisualAdapter found but no per-neighbor visual compressor queries to hook")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable_params / all_params if all_params > 0 else 0
    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {percentage:.2f}")

    # Optional: Manual check for specific modules
    def check_requires_grad(model, module_name):
        for name, param in model.named_parameters():
            if module_name in name:
                print(f"{name}: requires_grad={param.requires_grad}")

    check_requires_grad(model, "visual_adapter")  # Should all be True
    check_requires_grad(model, "textual_aggregation")
    check_requires_grad(model, "textual_aggregation")
    check_requires_grad(model, "visual")   # Should be False if fix_vit=True or train_only_visual_adapter=True, except perhaps attn_pool if enabled
    check_requires_grad(model, "lora")     # LoRA adapters should be True

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Load graph-aware visual adapter and TargetAwareTextualAggregation states if resuming from checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint_dir = training_args.resume_from_checkpoint
        if os.path.isdir(checkpoint_dir):
            # Load graph-aware visual adapter state
            visual_adapter_path = os.path.join(checkpoint_dir, "visual_adapter.pth")
            if os.path.exists(visual_adapter_path) and hasattr(model.transformer, 'visual_adapter'):
                try:
                    visual_adapter_state = torch.load(visual_adapter_path, map_location='cpu')
                    model.transformer.visual_adapter.load_state_dict(visual_adapter_state)
                    print(f"Loaded graph-aware visual adapter state from {visual_adapter_path} for resume")
                except Exception as e:
                    print(f"Error: Failed to load graph-aware visual adapter state for resume: {e}")
            else:
                if not os.path.exists(visual_adapter_path):
                    print(f"Error: graph-aware visual adapter state file not found for resume: {visual_adapter_path}")
                if not hasattr(model.transformer, 'visual_adapter'):
                    print(f"Error: graph-aware visual adapter not found in model for resume")
            
            # Load TargetAwareTextualAggregation state if available
            textual_aggregation_module = resolve_textual_aggregation_module(model.transformer)
            textual_aggregation_path = os.path.join(checkpoint_dir, "textual_aggregation.pth")
            if os.path.exists(textual_aggregation_path) and textual_aggregation_module is not None:
                try:
                    textual_aggregation_state = torch.load(textual_aggregation_path, map_location='cpu')
                    textual_aggregation_module.load_state_dict(textual_aggregation_state)
                    print(f"Loaded TargetAwareTextualAggregation state from {textual_aggregation_path}")
                except Exception as e:
                    print(f"Error: Failed to load TargetAwareTextualAggregation state: {e}")
            else:
                if not os.path.exists(textual_aggregation_path):
                    print(f"Warning: TargetAwareTextualAggregation checkpoint not found in {checkpoint_dir}")
                if textual_aggregation_module is None:
                    print(f"Error: TargetAwareTextualAggregation module not present on model; skipping load")

    # Load data
    placeholder_count = getattr(training_args, 'textual_aggregation_context_tokens', 16)
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        max_len=training_args.model_max_length,
        nbr_placeholder_count=placeholder_count,
    )
    
    # Set text_info_data and neighbor_data on the model for text graph node processing
    # We need to merge all the dataset-specific dicts for the model's global lookup
    text_info_data_list = data_module.pop('text_info_data_list', [])
    neighbor_data_list = []
    
    if text_info_data_list:
        neighbor_data_list = load_neighbor_data(data_args.neighbor_data_path) if hasattr(data_args, 'neighbor_data_path') else []
        
        # Merge all dicts for model's global lookup
        # Note: IDs MUST be unique across datasets, or later datasets will overwrite earlier ones
        merged_text_info = {}
        merged_neighbor_data = {}
        
        for idx, text_info in enumerate(text_info_data_list):
            merged_text_info.update(text_info)
            if idx < len(neighbor_data_list):
                merged_neighbor_data.update(neighbor_data_list[idx])
        
        model.transformer.set_text_info_data(merged_text_info, merged_neighbor_data)
        print(f"Set text_info_data on model: merged {len(text_info_data_list)} datasets with {len(merged_text_info)} total entries")
        print(f"   Warning:  NOTE: If node IDs overlap between datasets, later datasets overwrite earlier ones in the model's lookup!")
        print(f"   Warning:  Training uses dataset-specific lookups, so each example uses correct neighbors")

    # Log dataset statistics to wandb
    if "wandb" in training_args.report_to:
        try:
            import wandb
            wandb.log({
                "dataset/num_train_examples": len(data_module['train_dataset']),
                "dataset/num_eval_examples": len(data_module['eval_dataset']) if data_module['eval_dataset'] else 0,
                "dataset/num_text_info_entries": len(merged_text_info) if text_info_data_list else 0,
                "dataset/num_neighbor_entries": len(merged_neighbor_data) if text_info_data_list else 0,
                "dataset/num_datasets": len(text_info_data_list) if text_info_data_list else 0,
            })
        except Exception as e:
            print(f"Warning: Failed to log dataset stats to wandb: {e}")

    # Create callbacks
    gradient_callback = GradientLoggingCallback(model, log_interval=100)
    visual_adapter_save_callback = GraphAwareVisualAdapterSaveCallback(model)
    
    # Create custom data collator that handles text_neighbor_ids
    data_collator = TextGraphDataCollator(tokenizer=tokenizer)
    print(f"Created TextGraphDataCollator for handling text_neighbor_ids")

    # Start trainer with callbacks and custom collator
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        callbacks=[gradient_callback, visual_adapter_save_callback],
        data_collator=data_collator,  # 
        **data_module
    )

    trainer.train()
    trainer.save_state()  # Save training metadata (optimizer, scheduler, etc.)

    # After training, validate graph-aware visual adapter was trained by comparing parameters
    if hasattr(model.transformer, 'visual_adapter') and initial_visual_adapter_queries is not None:
        visual_adapter = model.transformer.visual_adapter
        if hasattr(visual_adapter, 'visual_compressor') and hasattr(visual_adapter.visual_compressor, 'queries'):
            final_visual_adapter_queries = visual_adapter.visual_compressor.queries.data.cpu()
            queries_changed = not torch.allclose(initial_visual_adapter_queries, final_visual_adapter_queries, atol=1e-5)
            print(f"graph-aware visual adapter per-neighbor visual compressor parameters changed during training: {'YES (trained)' if queries_changed else 'NO (issue!)'}")
        else:
            print(f"graph-aware visual adapter found but no per-neighbor visual compressor queries to validate")

    # Check for NaN/Inf before saving
    check_model_for_nan_inf(model)

    # Save PEFT adapters (LoRA + modules_to_save) - small
    trainer.model.save_pretrained(training_args.output_dir)

    # Save graph-aware visual adapter state separately - small
    if hasattr(model.transformer, 'visual_adapter'):
        visual_adapter_path = os.path.join(training_args.output_dir, "visual_adapter.pth")
        torch.save(model.transformer.visual_adapter.state_dict(), visual_adapter_path)
        print(f"Saved graph-aware visual adapter state to {visual_adapter_path}")
    
    # Save TargetAwareTextualAggregation state
    textual_aggregation_module = resolve_textual_aggregation_module(model.transformer)
    if textual_aggregation_module is not None:
        textual_aggregation_path = os.path.join(training_args.output_dir, "textual_aggregation.pth")
        torch.save(textual_aggregation_module.state_dict(), textual_aggregation_path)
        print(f"Saved TargetAwareTextualAggregation state to {textual_aggregation_path}")

    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
