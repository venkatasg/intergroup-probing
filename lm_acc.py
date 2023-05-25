'''
Calculate LM accuracy before and after intervention
'''

import torch
import pandas as pd
import random
import numpy as np
import ipdb
import argparse
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    set_seed,
    logging as loggingt
)
from accelerate import Accelerator
from datasets import Dataset, logging as loggingd
from inlp.debias import debias_by_specific_directions
from math import ceil

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
        '--layer',
        type=int,
        required=True,
        help='Layer on which to run the SVM classifier. Should be between 0 and 11'
    )
    parser.add_argument(
        '--num_classifiers',
        type=int,
        required=True,
        help='Number of inlp directions. 0 for no intervention'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        required=True,
        help='Alpha for alterrep. Set to positive for positive intervention, negative for negative intervention, and zero for amnesic'
    )
    parser.add_argument(
        '--control',
        action='store_true',
        help='Use random subspace projections instead of inlp'
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
        help='Batch size for training'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Finetuned model to use to extract representations from'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='vinai/bertweet-base',
        help='Huggingface model name to use'
    )
    parser.add_argument(
        '--reps_path',
        type=str,
        required=True,
        help='Path to saved tensors'
    )
    
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    random.seed(args.seed)
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    device = accelerator.device
    
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,normalization=True)
    mask_token_id = tokenizer.mask_token_id
    
    # We need to transfer the AutoModel in the finetuned model to the mlm model
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.roberta = finetuned_model.roberta
    
    # Now to test the mlm model
    
    # First load the data properly
    df = pd.read_csv('lmacc-dev.tsv', sep='\t')
    data = Dataset.from_pandas(df)
    max_len = 128
    pad_token = tokenizer.pad_token_id

    def get_dict(x, tokenizer):
        input_ids = tokenizer.convert_tokens_to_ids(x['tweet'].split())
        len_tweet = len(input_ids)
        input_ids += [pad_token for i in range(128-len_tweet)]
        
        new_dict = {
            'input_ids': input_ids,
            'attention_mask': [1 for i in range(len_tweet)] + [0 for i in range(128-len_tweet)],
            'answer_id': x['answer_id']
        }
        
        return new_dict
    data = data.map(lambda x: get_dict(x, tokenizer), batched=False)
    
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'answer_id', 'mask_index'])
    
    att_mask = data['attention_mask']
    att_mask_batched = [att_mask[i:i+args.batch] for i in range(0,att_mask.shape[0], args.batch)]

    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch, shuffle=False)
    
    # Define intervention hooks
    def total_intervention(h_out, P, ws, alpha):
        '''
        Perform positive or negative intervention on all tokens
        '''
        # Modify everything except the cls token
        h = h_out[0][:,1:,:]
        
        # Positive intervention is making text more AAVE
        signs = torch.sign(h@ws.T).long()
        
        # h_r component
        proj = (h@ws.T)
        if alpha>=0:
            proj = proj * signs
        else:
            proj = proj * (-signs)
        h_r = proj@ws*np.abs(alpha)
        
        # Get vector only in the direction of perpendicular to decision boundary
        h_n = h@P
    
        # Now pushing it either in positive or negative intervention direction
        h_alter = h_n + h_r
        
        # Return h_alter concatenated with the cls token
        return (torch.cat((h_out[0][:,:1,:], h_alter), dim=1),)
    
    # Sampled tokens Intervention hook function
    def intervention(h_out, P, ws, alpha):
        '''
        Perform positive or negative intervention on all tokens
        '''
        
        # Get attention mask from nonlocal variables att_mask and batch_ind
        att_mask = att_mask_batched[batch_ind]
        
        # Collect altered representations in this varaible
        h_final = None
        
        for batch_elem in range(h_out[0].shape[0]):
            # Sample 30% of active tokens to perform intervention on
            h_elem = h_out[0][batch_elem]
            max_tok = torch.nonzero(att_mask[batch_elem]==1, as_tuple=True)[0][-1].item()
            sample_size = ceil(0.2*max_tok)
            inds_to_alter = torch.randint(low=1, high=max_tok, size=(sample_size,))
            h_alter = h_elem[inds_to_alter,:]
            
            # AlterRep code starts here
            signs = torch.sign(h_alter@ws.T).long()
            
            # h_r component
            proj = (h_alter@ws.T)
            if alpha>=0:
                proj = proj * signs
            else:
                proj = proj * (-signs)
            h_r = proj@ws*np.abs(alpha)
            
            # Get vector only in the direction of perpendicular to decision boundary
            h_n = h_alter@P
        
            # Now pushing it either in positive or negative intervention direction
            h_alter = h_n + h_r
            
            for i, ind in enumerate(inds_to_alter):
                h_elem[ind,:] = h_alter[i,:]
            
            if h_final is None:
                h_final = h_elem.unsqueeze(0)
            else:
                h_final = torch.cat((h_final, h_elem.unsqueeze(0)), axis=0)
        
        # Return h_final
        return (h_final,)
        
    # Load iNLP parameters and add hook if at least one dimension is being removed
    if args.num_classifiers > 0:
        if not args.control:
            with open(args.reps_path + "/Ws.layer={}.seed={}.npy".format(args.layer, args.seed), "rb") as f:
                Ws = np.load(f)
        else:
            with open(args.reps_path + "/Ws.rand.layer={}.seed={}.npy".format(args.layer, args.seed), "rb") as f:
                Ws = np.load(f)
        
        # Reduce Ws to number of classifiers you want to set it to
        Ws = Ws[:args.num_classifiers,:]
        
        # Now derive P from Ws
        list_of_ws = [np.array([Ws[i, :]]) for i in range(Ws.shape[0])]
        P = debias_by_specific_directions(directions=list_of_ws, input_dim=Ws.shape[1])
        
        Ws = torch.tensor(Ws/np.linalg.norm(Ws, keepdims = True, axis = 1)).to(torch.float32).to(device)
        P = torch.tensor(P).to(torch.float32).to(device)
        
        # Insert newaxis for 1 classifier edge case
        if len(Ws.shape) == 1:
            Ws = Ws[np.newaxis,:]
        
        hook = model.roberta.encoder.layer[args.layer-1].register_forward_hook(lambda m, h_in, h_out: intervention(h_out=h_out, P=P, ws=Ws, alpha=args.alpha))
        
    model, dataloader = accelerator.prepare(model, dataloader)
    
    model.zero_grad()
    model.eval()
    correct = 0
    for batch_ind, input_dict in enumerate(dataloader):
        with torch.no_grad():
            output = model(
                input_ids=input_dict['input_ids'],
                attention_mask=input_dict['attention_mask']
            )
        
        answer_id = input_dict['answer_id'].detach().cpu().tolist()
        log_probs = torch.log_softmax(output['logits'], dim=2)
        
        for i in range(log_probs.shape[0]):
            ilog_probs = log_probs[i]
            topk_answers = torch.topk(ilog_probs[input_dict['mask_index'][i],:], 100)[1].detach().cpu().numpy().tolist()
            if answer_id[i] in topk_answers:
                correct += 1
    
    print("LM Top-100 accuracy: ", np.round((correct/df.shape[0])*100, 1))
