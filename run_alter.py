'''
Run AlterRep operation using learned iNLP directions
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
    get_scheduler,
    set_seed,
    logging as loggingt
)
from accelerate import Accelerator
from datasets import Dataset, logging as loggingd
import evaluate
from inlp.debias import debias_by_specific_directions
from math import ceil

# Logging level for datasets
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

# Turn off annoying pandas warnings
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

if __name__ == '__main__':
    # initialize argument parser
    description = 'Which layer to collect states on'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--layer',
        type=int,
        required=True,
        help='Layer on which to hook the intervention. Should be between 1 and 12'
    )
    parser.add_argument(
        '--num_classifiers',
        type=int,
        required=True,
        help='Number of inlp directions'
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
        '--dev',
        action='store_true',
        help='Test on dev or test set'
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
        help='Folder containing data for testing'
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
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,normalization=True)
    
    df = pd.read_csv(args.data_path, sep='\t')
    
    # Positive affect and negative affect
    df['affect'] = df.apply(lambda x: 1 if (x['Feeling']=='warm' or x['Behavior']=='app') else 0, axis=1)
    df['spec'] = df['Specificity'].apply(lambda x: specificity_calc(x))
    
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    # Rename group column to labels for use with BERT model
    data = data.rename_columns({'group': 'labels'})

    if args.dev:
        test_df = df[df['Split']=='dev']
        test_data = data.filter(lambda x: x['Split']=='dev')
    else:
        test_df = df[df['Split']=='test']
        test_data = data.filter(lambda x: x['Split']=='test')

    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    att_mask = test_data['attention_mask']
    att_mask_batched = [att_mask[i:i+args.batch] for i in range(0,att_mask.shape[0], args.batch)]

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False)
    
    # Total Intervention hook function
    def total_intervention(h_out, P, ws, alpha):
        '''
        Perform positive or negative intervention on all tokens
        '''
        # Take out the cls token
        h_tokens = h_out[0][:,1:,:]
        
        # AlterRep code starts here
        signs = torch.sign(h_tokens@ws.T).long()
        
        # h_r component
        proj = (h_tokens@ws.T)
        if alpha>=0:
            proj = proj * signs
        else:
            proj = proj * (-signs)
        h_r = proj@ws*np.abs(alpha)
        
        # Get vector only in the direction of perpendicular to decision boundary
        h_n = h_tokens@P
    
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
    
    test_dataloader, model = accelerator.prepare(test_dataloader, model)
    
    # Load evaluation metric and array to store predictions
    f1_metric = evaluate.load('f1')
    test_preds = np.array([])
    
    model.zero_grad()
    model.eval()
     
    for batch_ind, input_dict in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(
                input_ids=input_dict['input_ids'],
                attention_mask=input_dict['attention_mask']
            )
        
        predictions = output['logits'].argmax(dim=1)

        # Gather preds for evaluation
        predictions, references = accelerator.gather_for_metrics((predictions, input_dict['labels']))
        f1_metric.add_batch(predictions=predictions, references=references)
        test_preds = np.append(test_preds, predictions.detach().cpu().numpy().flatten())
    
    if args.num_classifiers > 0:
        hook.remove()
    
    print("F1 Score: ", np.round(f1_metric.compute(average='micro')['f1']*100, 1))
    
    test_df['pred'] = test_preds
    in_group_tot = test_df[(test_df['pred']==1)].shape[0]/test_df.shape[0]
    in_group_pos = test_df[(test_df['affect']==1) & (test_df['pred']==1)].shape[0]/test_df[(test_df['affect']==1)].shape[0]
    in_group_neg = test_df[(test_df['affect']==0) & (test_df['pred']==1)].shape[0]/test_df[(test_df['affect']==0)].shape[0]
    
    print("% in-group total:", np.round(in_group_tot*100, 1))
    print("% in-group on positive affect:", np.round(in_group_pos*100, 1))
    print("% in-group on negative affect:", np.round(in_group_neg*100, 1))
    
    in_group_in = test_df[(test_df['group']==1) & (test_df['pred']==1)].shape[0]/test_df[(test_df['group']==1)].shape[0]
    in_group_out = test_df[(test_df['group']==0) & (test_df['pred']==1)].shape[0]/test_df[(test_df['group']==0)].shape[0]
    
    print("% in-group on in-group:", np.round(in_group_in*100, 1))
    print("% in-group on out-group:", np.round(in_group_out*100, 1))
    
    in_group_spec = test_df[(test_df['spec']==1) & (test_df['pred']==1)].shape[0]/test_df[(test_df['spec']==1)].shape[0]
    in_group_gen = test_df[(test_df['spec']==0) & (test_df['pred']==1)].shape[0]/test_df[(test_df['spec']==0)].shape[0]
    
    print("% in-group on spec:", np.round(in_group_spec*100, 1))
    print("% in-group on gen:", np.round(in_group_gen*100, 1))
    
    if args.dev:
        test_df.to_csv('predictions-dev.tsv', sep='\t', index=False)
   
