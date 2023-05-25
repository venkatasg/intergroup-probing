'''
Script to save tensors for building counterfactual classifer on top for any attribute attr
'''
import torch
import pandas as pd
import random
import numpy as np
import ipdb
import argparse
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
    logging as loggingt
)
from accelerate import Accelerator
from datasets import Dataset, logging as loggingd

# Logging level for datasets
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

# Turn off chained assignment warnings from pandas
pd.set_option('chained_assignment',None)

def specificity_calc(x):
   '''
   0 if specificity is less than 3, and 1 if specificity is greater than 4
   '''

   if (x-3.0) < 0.00001:
      return 0
   elif (x-4.0) > 0.00001:
      return 1
   else:
      return 100
      
if __name__== "__main__":
    # initialize argument parser
    description = 'Script to extract representations corresponding to an attribute that we will intervene on downstream'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--layer',
        type=int,
        required=True,
        help='Layer from which to extract representations. Must be between 1 and 12.'
    )
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
        help='Batch size'
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
        help='Path to dataset'
    )
    parser.add_argument(
        '--attr',
        type=str,
        required=True,
        help='Attribute to collect values to train INLP on downstream. Must be one of affect, spec, doe, party, gen'
    )
    args = parser.parse_args()
    
    assert args.attr in ['spec', 'affect', 'doe', 'party', 'gen', 'pos', 'conc'], "attr should be one of spec, affect, doe, party, gen, conc, or pos"
    
    # Set seed
    set_seed(args.seed)
    random.seed(args.seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    device = accelerator.device
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, output_hidden_states=True)

    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, normalization=True)
    
    # Load the dataset. Generics uses different data from cong_data
    if args.attr != 'gen':
        df = pd.read_csv(args.data_path + '/cong_data.tsv', sep='\t')
    else:
        df = pd.read_csv(args.data_path + '/gen_data.tsv', sep='\t')
    
    # Each attribute has different processing and columns to look into
    if args.attr=='affect':
        # Positive affect and negative affect
        df['affect'] = df.apply(lambda x: 1 if (x['Feeling']=='warm' or x['Behavior']=='app') else 0, axis=1)
    elif args.attr=='spec':
        # Binarize specificity for now by excluding middle third
        df['spec'] = df['Specificity'].apply(lambda x: specificity_calc(x))
        df = df[df['spec']!=100]
    elif args.attr=='doe':
        doe_token = tokenizer.convert_tokens_to_ids(['@USER'])[0]
    elif args.attr=='party':
        df['party'] = df['party'].apply(lambda x: 1 if x=='D' else 0)
    elif args.attr=='gen':
        df['gen'] = df['gen'].apply(lambda x: 1 if x=='GENERIC' else 0)
    elif args.attr=='pos':
        pos = pd.read_csv(args.data_path + '/cong_data_pos.tsv', sep='\t')
        pos.set_index('TweetId', inplace=True)
        df['upos_tokens'] = df['TweetId'].apply(lambda x: pos.loc[x, 'upos_tokens'])
    elif args.attr=='conc':
        conc = pd.read_csv(args.data_path + '/concreteness.tsv', sep='\t')
        conc = conc[conc['Conc.SD'] < 0.5]
        df = conc.loc[:, ['Word', 'Conc.M']].reset_index(drop=True)
        df['conc'] = df['Conc.M'].apply(lambda x: specificity_calc(x))

    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    
    # If generic make train test split
    if args.attr=='gen':
        data = data.train_test_split(test_size=0.2, shuffle=True, seed=args.seed)
        # Rename test split to dev
        data['dev'] = data.pop('test')
    
    model = accelerator.prepare(model)
    
    # Set our model to evaluation mode
    model.zero_grad()
    model.eval()

    for split_name in ['train', 'dev']:
        # Create the split
        if args.attr != 'gen':
            split_data = data.filter(lambda x: x['Split']==split_name)
        else:
            split_data = data[split_name]
        
        # Set data format and dataloader
        if args.attr not in ['doe', 'pos']:
            split_data.set_format(type='torch', columns=['input_ids', 'attention_mask', args.attr])
        else:
            split_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        dataloader = torch.utils.data.DataLoader(split_data, batch_size=args.batch, shuffle=False)
        dataloader = accelerator.prepare(dataloader)

        all_reps = np.array([], dtype=np.float32).reshape(0, 768)
        # group = np.array([], dtype=np.int32).reshape(0, 1)
        attr = np.array([], dtype=np.int32).reshape(0, 1)
        
        if args.attr != 'gen':
            split_df = df[df['Split']==split_name]
        ind = 0         # To keep track of index element in each split
        
        for _, input_dict in enumerate(dataloader):
            with torch.no_grad():
                output = model(
                    input_ids=input_dict['input_ids'],
                    attention_mask=input_dict['attention_mask']
                )
            
            # Use args.layer by itself. Index 0 is embedding layer, we don't want that, and args.layer indexes from 1-12 corresponds to 1-12 hidden layers in roberta
            layer_hidden_state = output['hidden_states'][args.layer]
            batch_size = layer_hidden_state.shape[0]
            # b_group = input_dict['labels'].detach().cpu().numpy()
            
            if args.attr not in ['doe', 'pos']:
                b_attr = input_dict[args.attr].detach().cpu().numpy()
            
            sampled_inds_dim0 = torch.tensor([], dtype=torch.long)
            sampled_inds_dim1 = torch.tensor([], dtype=torch.long)
            
            for i, sent in enumerate(input_dict['attention_mask']):
                # Length of sentence to set cut off for sampling
                max_tok = torch.nonzero(sent==1, as_tuple=True)[0][-1] - 1
                
                if args.attr == 'doe':
                    # Get index of @USER token and a random token that's not doe_token
                    # Sometimes there's more than one username. This is because someone other than a congressman is mentioned
                    doe_token_ind = torch.where(input_dict['input_ids'][i]==(doe_token))[0][0]
                    other_ind = np.random.choice([i for i in range(1, max_tok.detach().cpu().item()) if i!=doe_token])
                    s_inds = torch.cat((torch.tensor([doe_token_ind]), torch.tensor([other_ind])))
                    rep_value = 2
                
                elif args.attr=='pos':
                    pos_tags = split_df['upos_tokens'].tolist()[ind]
                    noun_positions = [m for m, p in enumerate(pos_tags.split()) if p=='NOUN']
                    non_noun_positions = [m for m, p in enumerate(pos_tags.split()) if p!='NOUN']
                    
                    if len(noun_positions)<1 or len(non_noun_positions)<1:
                        ind += 1
                        continue
                    
                    # Sample 1 nouns and 1 non-nouns and save that in an array
                    s_inds = torch.tensor(random.sample(noun_positions, 1) + random.sample(non_noun_positions, 1))
                    attr = np.concatenate((attr, np.array([1,0], dtype=np.int32).reshape(2, 1)), axis=0)
                    rep_value = 2
                    ind += 1
                
                elif args.attr=='gen':
                    # Sample 3 tokens at random from each sentence
                    s_inds = torch.randint(low=0, high=max_tok.detach().cpu().item(), size=(3,),  dtype=torch.long)
                    rep_value = 3
                
                else:
                    # Sample 3 tokens at random from each sentence
                    s_inds = torch.randint(low=0, high=max_tok.detach().cpu().item(), size=(5,),  dtype=torch.long)
                    rep_value = 5
                
                rep_dim0 =  torch.repeat_interleave(torch.tensor(i), rep_value)
                sampled_inds_dim0 = torch.cat((sampled_inds_dim0, rep_dim0), axis=0)
                sampled_inds_dim1 = torch.cat((sampled_inds_dim1, s_inds))
                
            sampled_reps = layer_hidden_state[sampled_inds_dim0, sampled_inds_dim1,:].detach().cpu().numpy()
            all_reps = np.concatenate((all_reps, sampled_reps), 0)
            
            # rep_group = np.repeat(b_group,[rep_value for i in range(batch_size)], axis=0)[:, np.newaxis]
            # group = np.concatenate((group, rep_group), axis=0)
            
            if args.attr != 'pos':
                if args.attr != 'doe':
                    rep_attr = np.repeat(b_attr,[rep_value for i in range(batch_size)], axis=0)[:, np.newaxis]
                else:
                    rep_attr = np.array([(i+1)%2 for i in range(batch_size*2)])[:, np.newaxis]
                
                attr = np.concatenate((attr, rep_attr), axis=0)
        
        print(split_name, all_reps.shape)
        
        # Folder to save reps
        reps_path = 'reps_' + args.attr
        # Save representations to file
        with open(reps_path + '/acts_seed_' + str(args.seed) + '_layer_' + str(args.layer) + '_' + split_name + '.npy', 'wb') as f:
            np.save(f, all_reps)
            
        # with open(args.reps_path + '/group_seed_' + str(args.seed) + '_' + split_name + '.npy', 'wb') as f:
        #     np.save(f, group)
            
        with open(reps_path + '/attr_seed_' + str(args.seed) + '_' + split_name + '.npy', 'wb') as f:
            np.save(f, attr)
