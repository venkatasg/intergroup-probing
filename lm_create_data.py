import torch
import pandas as pd
import random
import numpy as np
import ipdb
import argparse
from transformers import (
    AutoTokenizer,
    set_seed,
    logging as loggingt
)
from datasets import Dataset, logging as loggingd

# Logging level for datasets
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

# Turn off annoying pandas warnings
pd.set_option('chained_assignment',None)

if __name__ == '__main__':
    # initialize argument parser
    description = 'Which layer to collect states on'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Finetuned model to use to extract representations from'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Finetuned model to use to extract representations from'
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    random.seed(args.seed)
    
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,normalization=True)
    mask_token_id = tokenizer.mask_token_id
    
    # Let's create masked sentences
    df = pd.read_csv(args.data_path, sep='\t')
    df = df[df['Split']!='test']
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding=False), batched=False)
    
    def mask_rand_token(x, mask_token_id):
        rand_int = np.random.randint(1, len(x['input_ids']) - 1)
        x['answer_id'] = x['input_ids'][rand_int]
        x['input_ids'][rand_int] = mask_token_id
        x['mask_index'] = rand_int
        
        return x
        
    data = data.map(lambda x: mask_rand_token(x, mask_token_id), batched=False)
    
    new_df = {'tweet': [], 'mask_index': [], 'answer_id': []}
    for i, datum in enumerate(data):
        sid = datum['input_ids']
        new_df['tweet'].append(' '.join(tokenizer.convert_ids_to_tokens(sid)))
        new_df['answer_id'].append(datum['answer_id'])
        new_df['mask_index'].append(datum['mask_index'])
    
    new_df = pd.DataFrame(new_df)
    new_df.to_csv('lmacc-dev.tsv', sep='\t', index=False)
