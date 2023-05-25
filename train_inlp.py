'''
Train iNLP on your own dataset
'''

import pandas as pd
import numpy as np
import random
import ipdb
import argparse
from sklearn.svm import LinearSVC
from inlp import debias


def run_inlp(num_classifiers, x_train, y_train, x_dev, y_dev, seed):
    '''
    Main function that calls into inlp methods
    '''

    input_dim = x_train.shape[1]

    # Define classifier here
    clf = LinearSVC
    params = {"max_iter": 10000, "dual": False, "random_state": 0}


    _, _, Ws, accs = debias.get_debiasing_projection(
        classifier_class=clf,
        cls_params=params,
        num_classifiers=num_classifiers,
        input_dim=768,
        is_autoregressive=True,
        min_accuracy=0,
        X_train=x_train,
        Y_train=y_train,
        X_dev=x_dev,
        Y_dev=y_dev,
        by_class = False
    )
    
    _, _, Ws_rand, accs_rand = debias.get_random_projection(
        classifier_class=clf,
        cls_params=params,
        num_classifiers=num_classifiers,
        input_dim=768,
        is_autoregressive=True,
        min_accuracy=0,
        X_dev=x_dev,
        Y_dev=y_dev,
        by_class = False
    )
    
    print("Accs: ", accs)
    print("Random accs: ", accs_rand)
    return Ws, Ws_rand

if __name__ == '__main__':
    # initialize argument parser
    description = 'Parameters for running INLP'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--layer',
        type=int,
        required=True,
        help='Layer on which to run the SVM classifier'
    )
    parser.add_argument(
        '--num_classifiers',
        type=int,
        required=True,
        help='Number of inlp directions to train on'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed'
    )
    parser.add_argument(
        '--attr',
        type=str,
        required=True,
        help='Which attribute are we building a classifier on'
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility on a specific machine
    random.seed(args.seed)
    np.random.seed(args.seed)
    np.random.RandomState(args.seed)

    # Load the representations
    reps_train = np.load('reps_' + args.attr + '/acts_seed_' + str(args.seed) + '_layer_' + str(args.layer) + '_train' + '.npy')
    attr_train = np.load('reps_' + args.attr + '/attr_seed_' + str(args.seed) + '_train' + '.npy')

    reps_dev = np.load('reps_' + args.attr + '/acts_seed_' + str(args.seed) + '_layer_' + str(args.layer) + '_dev' + '.npy')
    attr_dev = np.load('reps_' + args.attr + '/attr_seed_' + str(args.seed) + '_dev' + '.npy')


    Ws, Ws_rand = run_inlp(
        num_classifiers=args.num_classifiers,
        x_train=reps_train,
        y_train=attr_train,
        x_dev=reps_dev,
        y_dev=attr_dev,
        seed=args.seed
    )

    # concatenate lists into one numpy array
    Ws = np.concatenate(Ws)
    Ws_rand = np.concatenate(Ws_rand)

    with open('reps_' + args.attr + "/Ws.layer={}.seed={}.npy".format(args.layer, args.seed), "wb") as f:
        np.save(f, Ws)
    
    with open('reps_' + args.attr + "/Ws.rand.layer={}.seed={}.npy".format(args.layer, args.seed), "wb") as f:
        np.save(f, Ws_rand)


