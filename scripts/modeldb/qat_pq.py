import argparse
import json
import itertools
from tqdm import tqdm
import pathlib
import os
import importlib

from ..utils.Namespace import UniConfig, load_config
from ..utils.Timer import tprint, Timer
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    import sys
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
    Timer('').start()
    # ================== Argument Parsing ==================
    parser = argparse.ArgumentParser(description="ModelDB")
    parser.add_argument("-f", "--file", type=str, help="Relative path to config.json. Relative to scripts/modeldb/configs/", required=True)
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name", required=True)
    parser.add_argument("-M", type=int, help="PQ config, number of sub-sections", required=True)
    parser.add_argument("--nbits", type=int, help="PQ config, number of bits per sub-section", required=True)
    parser.add_argument("--seed", type=int, help="Random seed", required=False, default=42)
    parser.add_argument("--save_dir", type=str, help="Directory to save the model", required=True)

    args = parser.parse_args()
    
    # ================== Config ==================
    config = UniConfig()
    config.device = 'cuda'
    
    config.root = pathlib.Path(__file__).parent.parent.parent
    config.config_root = config.root / "scripts" / "modeldb" / "configs"
    config.config_path = config.config_root / args.file
    config.save_dir = pathlib.Path(args.save_dir)

    # Load config
    config += load_config(config.config_root / "default.json")
    config += load_config(config.config_path)

    if args.M is not None:
        config.M = args.M
    if args.nbits is not None:
        config.nbits = args.nbits
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.pipeline is not None:
        config.pipeline = args.pipeline
    if args.seed is not None:
        config.seed = args.seed

    config.model_root = config.root / "models"
    config.datasets_root = config.root / "datasets" 

    config.model_path = config.model_root / config.folder
    config.cent_root = config.root / "centroids" / config.model_name / config.dataset

    from transformers import AutoConfig
    from .models.ModelContext import get_context

    config.model_config = AutoConfig.from_pretrained(config.model_path)
    config.context = get_context(config.model_config.model_type)

    # ================== Seed ==================
    seed_everything(config.seed)

    # ================== Load Model ==================
    tprint(f"Loading model {config.model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with config.context.init_context:
        model = AutoModelForCausalLM.from_pretrained(config.model_path).to(config.device)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            
    # ================== Initialize Codebook Register ==================
    tprint("Initializing codebook register")
    key_cent = torch.load(config.cent_root / f'key_cent_{config.M}_{config.nbits}.pq.pt', weights_only=True).to(config.device)
    val_cent = torch.load(config.cent_root / f'val_cent_{config.M}_{config.nbits}.pq.pt', weights_only=True).to(config.device)
    
    config.context.qat_codebook_register.init_register(model, key_cent, val_cent)
    
    # ================== Prepare Dataset ==================
    tprint(f"Preparing dataset {config.dataset}")
    from datasets import load_from_disk
    from transformers import DataCollatorForLanguageModeling
    
    ds = load_from_disk(str(config.datasets_root / config.dataset)).train_test_split(test_size=0.05)
    def tokenize_fn(ex):
        return tokenizer(ex["text"],
                        truncation=True,
                        max_length=2048,
                        padding=False)

    tok_ds = ds.map(tokenize_fn,
                    batched=True,
                    remove_columns=ds["train"].column_names)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # ================== Prepare Trainer ==================
    from transformers import Trainer, TrainingArguments
    args = TrainingArguments(
        output_dir                 = config.save_dir,
        bf16                       = True,          # A100 建议直接 bf16
        per_device_train_batch_size= 2,             # 每卡 2×2048≈16 k token
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps= 32,            # 2*32*8 ≈ 512 样本 ≈ 1 M token / step
        learning_rate              = 2e-5,          # 全参 7B 常用区间 1e‑5 – 5e‑5
        lr_scheduler_type          = "cosine",      # 余弦退火
        warmup_ratio               = 0.03,
        weight_decay               = 0.1,
        max_grad_norm              = 1.0,
        num_train_epochs           = 3,
        logging_steps              = 10,
        evaluation_strategy        = "epoch",
        save_strategy              = "epoch",
        save_total_limit           = 2,
        gradient_checkpointing     = True,
        deepspeed                  = "ds_config_zero3.json",
        ddp_find_unused_parameters = False,
        report_to                  = "none",
    )
    
    trainer = Trainer(
        model         = model,
        args          = args,
        train_dataset = tok_ds["train"],
        eval_dataset  = tok_ds["test"],
        data_collator = collator,
    )
    # # ================== QAT ===================
    tprint("Starting QAT")
    with config.context.qat_context, \
        config.context.qat_codebook_register:
            trainer.train()
            
            