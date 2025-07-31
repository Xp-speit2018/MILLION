import torch
from datasets import load_from_disk
from tqdm import tqdm
from ..utils.Timer import tprint
from ..utils.Namespace import UniConfig

def preprocess_dataset(dataset_name, root=None):
    supported = ['wikitext-2-raw-v1', 'wikitext-103-v1', 'ptb-text-only', 'wikitext-103-raw-v1']
    if dataset_name not in supported:
        raise ValueError('Dataset not supported')
    
    split = 'test'
    if dataset_name == 'wikitext-2-raw-v1':
        dataset = load_from_disk(str(root / dataset_name))[split]
    elif dataset_name == 'ptb-text-only':
        dataset = load_from_disk(str(root / 'PTB'))[split]
    # if dataset_name == 'ptb_text_only':
        dataset = dataset.rename_column('sentence', 'text')
    return dataset

def encode_dataset(dataset, tokenizer):
    merged_text = "\n\n".join(dataset['text'])
    res = tokenizer(merged_text, return_tensors='pt')
    return res

def perplexity(model, tokenizer, *, dataset, datasets_root, device, max_length, stride=None, verbose=True, cache_clear_func=None, **kwargs):
    model.to(device)
    model.eval()

    UniConfig().distort_recent = True
    if verbose:
        tprint(f'Loading dataset {dataset}')
    dataset_pr = preprocess_dataset(dataset, datasets_root)
    if verbose:
        tprint('Encoding dataset')

    encodings = encode_dataset(dataset_pr, tokenizer)

    stride = max_length
    
    seq_len = encodings['input_ids'].size(1)
    
    nlls = [] # negative log likelihoods
    total_length = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):

        # do some padding
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        if trg_len != max_length:
            continue

        input_ids = encodings['input_ids'][:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # -100 will be ignored by loss function
        
        if cache_clear_func is not None:
            cache_clear_func()
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs.loss.to(torch.float32)
        
        nlls.append(nll * trg_len)
        total_length += trg_len

        if verbose:
            current_ppl = torch.exp(torch.stack(nlls).sum() / total_length)
            print(f"current ppl: {current_ppl}")
        # tqdm.write(f"current ppl: {current_ppl}")
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

        # break
    
    ppl = torch.exp(torch.stack(nlls).sum() / total_length)

    # free memory
    del encodings
    del dataset_pr
    del input_ids
    del target_ids
    del outputs
    torch.cuda.empty_cache()

    return ppl.item()

if __name__ == '__main__':
    pass

    # # Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, help='Model name', default='gpt2-large')
    # parser.add_argument('--dataset', type=str, help='Dataset name', default='PTB')
    # parser.add_argument('--device', type=str, help='Device to use', default='cuda:7')
    # parser.add_argument('--verbose', action='store_true', help='Print perplexity after each sample')
    # parser.add_argument('--debug', action='store_true', help='Debug mode')
    # parser.add_argument('--stride', type=int, help='Stride for computing perplexity', default=None)
    # args = parser.parse_args()

    # # Load model
    # print(f'{t}Loading model {args.model}')
    # model, tokenizer = load_model(args.model, args.device)

    # # Compute perplexity
    # print(f'{t}Computing perplexity...')
    # if args.stride is None:
    #     tprint('Stride not specified, using default stride {model.config.n_positions}')
    #     args.stride = model.config.n_positions
    # if args.verbose:
    #     tprint('Using device {args.device}, stride {args.stride} over {model.config.n_positions} tokens per input')
    #     tprint('For stride meaning check https://huggingface.co/docs/transformers/perplexity')
    # ppl = perplexity(model, tokenizer, args.dataset, args.device, stride=args.stride, verbose=args.verbose, debug=args.debug)

    # # Print results
    # tprint(f'Perplexity: {ppl}')
