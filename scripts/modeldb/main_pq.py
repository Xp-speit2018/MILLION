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
from ..utils.rotation_utils import fuse_layer_norms, rotate_model
from ..utils.quarot_utils import cleanup_memory, llama_down_proj_groupsize, DEV
from ..utils.quant_utils import add_actquant, find_qlayers, ActQuantWrapper
from ..utils.hadamard_utils import get_hadK
from ..utils.gptq_utils import gptq_fwrd, rtn_fwrd
from ..utils.data_utils import get_loaders
from ..utils.model_utils import get_model_type, LLAMA_MODEL, OPT_MODEL

supported_models = [
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'meta-llama/Llama-2-70b-hf',
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3-70B',
            'facebook/opt-125m'
            ]
supported_datasets = ['wikitext2', 'ptb', 'c4']

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
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name", required=False)
    parser.add_argument("-M", type=int, help="PQ config, number of sub-sections", required=False)
    parser.add_argument("--nbits", type=int, help="PQ config, number of bits per sub-section", required=False)
    parser.add_argument("-m", "--merged_training", action="store_true", help="Train a merged PQ", required=False)
    parser.add_argument("--opq", action="store_true", help="Prepand LT before PQ to maximize squred error across dimensions", required=False)
    parser.add_argument("--seed", type=int, help="Random seed", required=False, default=42)
    parser.add_argument("--half", action="store_true", help="Use half precision", required=False)
    parser.add_argument("--breakdown", action="store_true", help="Breakdown timing, could lead to overhead due to additional torch.cuda.synchronize", required=False)
    parser.add_argument(
        "-p", "--pipeline",
        nargs='+',
        choices=["quarot", "baseline", "sampling", "training", "evaluation"],
        help="List of pipeline stages to execute"
    )

    # QuaRot args
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Model to load;')
                        # help='Model to load;', choices=supported_models)
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                        help='Dataset for Evaluation (default: wikitext2)', choices=supported_datasets,)
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--bsz', type=int, default=32,
                        help='Batch-size for PPL evaluation (default:32)')

    # Rotation Arguments
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=False, 
                        help='''Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys''')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--rotation_seed', type=int, default=-1,
                        help='Random Seed for generating random matrix!!')
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric Activation quantization (default: False)')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio')


    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')


    # General Quantization Arguments
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
                        help='Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8')

    # KV-Cache Quantization Arguments
    parser.add_argument('--v_bits', type=int, default=16,
                        help='''Number of bits for V-cache quantization. 
                        Note that quantizing the V-cache does not need any other rotation''')
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric V-cache quantization')
    parser.add_argument('--v_clip_ratio', type=float, default=1.0,
        help='Clip ratio for v-cache quantization. new_max = max * clip_ratio')
    
    parser.add_argument('--k_bits', type=int, default=16,
                        help='''Number of bits for K-cache quantization. 
                        Note that quantizing the K-cache needs another rotation for the keys/queries''')
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False, 
                        help='ASymmetric K-cache quantization')
    parser.add_argument('--k_pre_rope', action=argparse.BooleanOptionalAction, default=False, 
                        help='Pre-RoPE quantization for K-cache (not Supported yet!)')
    parser.add_argument('--k_clip_ratio', type=float, default=1.0,
        help='Clip ratio for k-cache quantization. new_max = max * clip_ratio')


    # Save/Load Quantized Model Arguments
    parser.add_argument('--load_qmodel_path', type=str, default=None,
                        help='Load the quantized model from the specified path!')
    parser.add_argument('--save_qmodel_path', type=str, default=None, 
                        help='Save the quantized model to the specified path!')


    args = parser.parse_args()

    if args.opq:
        raise NotImplementedError("OPQ is not implemented for GPU yet.")
    # if args.merged_training is False:
    #     raise NotImplementedError("Only merged training is supported😈. Use --merged_training")
    # ================== Config ==================
    config = UniConfig()
    config.device = 'cuda' # TODO: support multi-gpu

    config.root = pathlib.Path(__file__).parent.parent.parent
    config.config_root = config.root / "scripts" / "modeldb" / "configs"
    config.config_path = config.config_root / args.file

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
    if args.half is not None:
        config.half = args.half
    if args.breakdown is not None:
        config.breakdown = args.breakdown
    

    config.scalar_t = torch.float16 if config.half else torch.float32

    config.model_root = config.root / "models"
    config.datasets_root = config.root / "datasets" 

    config.model_path = config.model_root / config.folder
    config.sample_root = config.root / "kv_samples" / config.model_name / config.dataset
    config.cent_root = config.root / "centroids" / config.model_name / config.dataset

    config.opq = args.opq
    config.merged_training = args.merged_training
    
    if config.merged_training:
        config.sample_root = config.sample_root / "merged"
        config.cent_root = config.cent_root / "merged"
    else:
        config.sample_root = config.sample_root / "per_layer"
        config.cent_root = config.cent_root / "per_layer"

    from transformers import AutoConfig
    from .models.ModelContext import get_context

    config.model_config = AutoConfig.from_pretrained(config.model_path)
    config.context = get_context(config.model_config.model_type)

    from ..utils.pq_utils import nbits2dtype
    config.cache_dtype = nbits2dtype(config.nbits)

    # ================== Seed ==================
    seed_everything(config.seed)

    # ================== Load Model ==================
    if not (len(config.pipeline) == 1 and "training" in config.pipeline):
        tprint(f"Loading model {config.model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with config.context.init_context:
            model = AutoModelForCausalLM.from_pretrained(config.model_path).to(config.device)
            tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            if config.half:
                model = model.half()
            model.seqlen = config.max_length

    if "quarot" in config.pipeline:
        # Rotate the weights
        if args.rotate:
            if model_type := get_model_type(model) not in [LLAMA_MODEL, OPT_MODEL]:
                print(f'Rotation is not supported for model type {model_type}. Skipping.')
            else:
                fuse_layer_norms(model)
                rotate_model(model, args)
                cleanup_memory(verbos=True)
                    
                add_actquant(model) #Add Activation Wrapper to the model
                qlayers = find_qlayers(model)
                for name in qlayers:
                    if 'down_proj' in name:
                        had_K, K = get_hadK(model.config.intermediate_size)
                        qlayers[name].online_full_had = True
                        qlayers[name].had_K = had_K
                        qlayers[name].K = K
                        qlayers[name].fp32_had = args.fp32_had
                    if 'o_proj' in name:
                        had_K, K = get_hadK(model.config.num_attention_heads)
                        qlayers[name].online_partial_had = True
                        qlayers[name].had_K = had_K
                        qlayers[name].K = K
                        qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                        qlayers[name].fp32_had = args.fp32_had
        else:
            add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present

        if args.w_bits < 16:
            save_dict = {}
            if args.load_qmodel_path: # Load Quantized Rotated Model
                assert args.rotate, "Model should be rotated to load a quantized model!"
                assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
                print("Load quantized model from ", args.load_qmodel_path)
                save_dict = torch.load(args.load_qmodel_path)
                model.load_state_dict(save_dict["model"])
                
            elif not args.w_rtn: # GPTQ Weight Quantization
                assert "llama" in args.model, "Only llama is supported for GPTQ!"
                
                trainloader = get_loaders(
                    args.cal_dataset, nsamples=args.nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False
                )
                quantizers = gptq_fwrd(model, trainloader, DEV, args)
                save_dict["w_quantizers"] = quantizers
            else: # RTN Weight Quantization
                quantizers = rtn_fwrd(model, DEV, args)
                save_dict["w_quantizers"] = quantizers
                
            if args.save_qmodel_path:
                save_dict["model"] = model.state_dict()
                torch.save(save_dict, args.save_qmodel_path)


        # Add Input Quantization
        if args.a_bits < 16 or args.v_bits < 16:
            qlayers = find_qlayers(model, layers=[ActQuantWrapper])
            down_proj_groupsize = -1
            if args.a_groupsize > 0 and "llama" in args.model:
                down_proj_groupsize = llama_down_proj_groupsize(model, args.a_groupsize)
            
            for name in qlayers:            
                layer_input_bits = args.a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not(args.a_asym)
                layer_a_clip = args.a_clip_ratio
                
                if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                    qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                                groupsize=args.v_groupsize,
                                                sym=not(args.v_asym),
                                                clip_ratio=args.v_clip_ratio)
                
                if 'lm_head' in name: #Skip lm_head quantization   
                    layer_input_bits = 16
                
                if 'down_proj' in name: #Set the down_proj precision
                    if args.int8_down_proj:
                        layer_input_bits = 8
                    layer_groupsize = down_proj_groupsize

                    
                qlayers[name].quantizer.configure(bits=layer_input_bits,
                                                groupsize=layer_groupsize,
                                                sym=layer_a_sym,
                                                clip_ratio=layer_a_clip)
                
    # ================== baseline ==================
    if "baseline" in config.pipeline:
        tprint("Baseline")
        from ..benchmarks import dataset2benchmark
        benchmark = dataset2benchmark[config.dataset]

        with config.context.baseline_context:
            score_baseline = benchmark(model, tokenizer, **(config.to_dict()))

        # write to jsonl
        with open(config.root / "scripts" / "modeldb" / "results.jsonl", "a") as f:
            f.write(json.dumps({"score": score_baseline, "model_name": config.model_name, "dataset": config.dataset, "baseline": True, "half": config.half}))
            f.write("\n")

    # ================== sampling ==================
    if "sampling" in config.pipeline and config.dataset != '_synthetic':
        tprint("Sampling")

        if config.sample_root.exists() is True:
            # ask for confirmation
            tprint(f"Sampling path already exist at {config.sample_root}")
            
            while True:
                tprint("Clear the directory(c) or exit(e)? (c/e)")
                char = input().strip().lower()
                if char == 'e':
                    tprint("Exiting...")
                    exit()
                elif char == 'c':
                    tprint("Clearing the directory...")
                    for file in config.sample_root.glob("*"):
                        file.unlink()
                    tprint("Directory cleared.")
                    tprint(f"Recreating sampling path at {config.sample_root}")
                    os.makedirs(config.sample_root, exist_ok=True)
                    break
            
        else:
            tprint(f"Creating sampling path at {config.sample_root}")
            os.makedirs(config.sample_root, exist_ok=True)


        from .Errors import SamplingComplete
        from ..benchmarks import dataset2benchmark
        from ..utils.Reservoir import Reservoir
        benchmark = dataset2benchmark[config.dataset]
        head_size = config.model_config.hidden_size // config.model_config.num_key_value_heads
        if config.merged_training is True:
            config.key_reservoir = Reservoir(
                max_size = 256 * 2**config.nbits,
                device = "cpu", # use CPU for reservoir to avoid GPU memory issues
                dim = head_size,
                dtype = model.dtype,
                name = f"{config.model_name}_{config.dataset}_merged"
            )
            config.value_reservoir = Reservoir(
                max_size = 256 * 2**config.nbits,
                device = "cpu",
                dim = head_size,
                dtype = model.dtype,
                name = f"{config.model_name}_{config.dataset}_merged"
            )
        else:
            config.key_reservoir = [
                Reservoir(
                    max_size = 256 * 2**config.nbits,
                    device = "cpu",
                    dim = head_size,
                    dtype = model.dtype,
                    name = f"{config.model_name}_{config.dataset}_layer{layer_idx}"
                ) for layer_idx in range(config.model_config.num_hidden_layers)
            ]
            config.value_reservoir = [
                Reservoir(
                    max_size = 256 * 2**config.nbits,
                    device = "cpu",
                    dim = head_size,
                    dtype = model.dtype,
                    name = f"{config.model_name}_{config.dataset}_layer{layer_idx}"
                ) for layer_idx in range(config.model_config.num_hidden_layers)
            ]
                
        with config.context.sampling_context:
            try:
                benchmark(model, tokenizer, **(config.to_dict()))
            except SamplingComplete as e:
                tprint(e)
                
        # serialize reservoirs
        from ..utils.fvecio import write_fvecs
        os.makedirs(config.sample_root, exist_ok=True)
        if config.merged_training is True:
            key_reservoir_path = config.sample_root / f'key_sampled_{config.M}_{config.nbits}.fvecs'
            value_reservoir_path = config.sample_root / f'value_sampled_{config.M}_{config.nbits}.fvecs'
            write_fvecs(key_reservoir_path, config.key_reservoir.reservoir[:config.key_reservoir.count].cpu().numpy())
            tprint(f"Reservoirs saved to {key_reservoir_path} and {value_reservoir_path}")
            del config.key_reservoir, config.value_reservoir
        else:
            for layer_idx in range(config.model_config.num_hidden_layers):
                key_reservoir_path = config.sample_root / f'key_sampled_{config.M}_{config.nbits}_layer{layer_idx}.fvecs'
                value_reservoir_path = config.sample_root / f'value_sampled_{config.M}_{config.nbits}_layer{layer_idx}.fvecs'
                write_fvecs(key_reservoir_path, config.key_reservoir[layer_idx].reservoir[:config.key_reservoir[layer_idx].count].cpu().numpy())
                write_fvecs(value_reservoir_path, config.value_reservoir[layer_idx].reservoir[:config.value_reservoir[layer_idx].count].cpu().numpy())
                tprint(f"Reservoirs saved to {key_reservoir_path} and {value_reservoir_path}")
            # del config.key_reservoir, config.value_reservoir

    # ================== training ==================
    if "training" in config.pipeline and config.dataset != '_synthetic':
        tprint("Training")
        from ..utils.fvecio import read_fvecs
        from ..utils.pq_utils import train_pq
        from ..utils.pq_utils import train_opq
        from torch import save

        os.makedirs(config.cent_root, exist_ok=True)

        if config.merged_training is True:
            key = read_fvecs(config.sample_root / f'key_sampled_{config.M}_{config.nbits}.fvecs')
            key_cent = train_pq(key, config.M, config.nbits)
            save(key_cent, config.cent_root / f'key_cent_{config.M}_{config.nbits}.pq.pt')
            del key, key_cent


            val = read_fvecs(config.sample_root / f'value_sampled_{config.M}_{config.nbits}.fvecs')
            val_cent = train_pq(val, config.M, config.nbits)
            save(val_cent, config.cent_root / f'val_cent_{config.M}_{config.nbits}.pq.pt')
            del val, val_cent
        else:
            for layer_idx in tqdm(range(config.model_config.num_hidden_layers), desc="Training PQ for each layer"):
                key = read_fvecs(config.sample_root / f'key_sampled_{config.M}_{config.nbits}_layer{layer_idx}.fvecs')
                key_cent = train_pq(key, config.M, config.nbits)
                save(key_cent, config.cent_root / f'key_cent_{config.M}_{config.nbits}_layer{layer_idx}.pq.pt')
                del key, key_cent

                val = read_fvecs(config.sample_root / f'value_sampled_{config.M}_{config.nbits}_layer{layer_idx}.fvecs')
                val_cent = train_pq(val, config.M, config.nbits)
                save(val_cent, config.cent_root / f'val_cent_{config.M}_{config.nbits}_layer{layer_idx}.pq.pt')
                del val, val_cent

    if "evaluation" not in config.pipeline:
        tprint("Exit")
        exit()

    # ================== PQ Config ==================
    if "evaluation" in config.pipeline:
        if config.dataset == '_synthetic':
            tprint("Using synthetic centroids for speed evaluation")
            if config.merged_training is True:
                key_cent = torch.randn(config.M, 2**config.nbits, config.d // config.M, dtype=model.dtype, device=config.device)
                val_cent = torch.randn(config.M, 2**config.nbits, config.d // config.M, dtype=model.dtype, device=config.device)
            else:
                key_cent = torch.randn(config.model_config.num_hidden_layers, config.M, 2**config.nbits, config.d // config.M, dtype=model.dtype, device=config.device)
                val_cent = torch.randn(config.model_config.num_hidden_layers, config.M, 2**config.nbits, config.d // config.M, dtype=model.dtype, device=config.device)
        else:
            if config.merged_training is True:
                tprint("Using merged centroids")
                key_cent = torch.load(config.cent_root / f'key_cent_{config.M}_{config.nbits}.pq.pt', weights_only=True)
                key_cent = key_cent.to(config.device).to(model.dtype)
                val_cent = torch.load(config.cent_root / f'val_cent_{config.M}_{config.nbits}.pq.pt', weights_only=True)
                val_cent = val_cent.to(config.device).to(model.dtype)
            else:
                tprint("Using per-layer centroids")
                key_cent = []
                val_cent = []
                for layer_idx in range(config.model_config.num_hidden_layers):
                    key_cent.append(torch.load(config.cent_root / f'key_cent_{config.M}_{config.nbits}_layer{layer_idx}.pq.pt', weights_only=True))
                    key_cent[-1] = key_cent[-1].to(config.device).to(model.dtype)
                    val_cent.append(torch.load(config.cent_root / f'val_cent_{config.M}_{config.nbits}_layer{layer_idx}.pq.pt', weights_only=True))
                    val_cent[-1] = val_cent[-1].to(config.device).to(model.dtype)
                key_cent = torch.stack(key_cent, dim=0).contiguous()
                val_cent = torch.stack(val_cent, dim=0).contiguous()


        from ..utils.pq_utils import DynamicPQCache
        cache = DynamicPQCache(
            bs = 1, # TODO: support batch size?
            num_key_value_heads=config.model_config.num_key_value_heads,
            nh = config.model_config.num_attention_heads,
            M = config.M,
            layer_num=config.model_config.num_hidden_layers,
            dtype=config.cache_dtype,
            nbits=config.nbits,
            d=config.d,
            scalar_t=config.scalar_t,
            merged_training=config.merged_training,
        )
        cache.set_cent(key_cent, val_cent)


        

        tprint("Evaluation")
        
        from ..benchmarks import dataset2benchmark
        benchmark = dataset2benchmark[config.dataset]

        with config.context.evaluation_context:
            score = benchmark(model, tokenizer, cache_clear_func=cache.init_cache, **(config.to_dict()))

        # write to jsonl
        with open(config.root / "scripts" / "modeldb" / "results.jsonl", "a") as f:
            f.write(json.dumps({"score": score, **config.to_serializable_dict()}))
            f.write("\n")

    tprint("Exit")
