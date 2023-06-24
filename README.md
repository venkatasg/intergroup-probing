# Counterfactual Probing Intergroup Bias for affect and specificity

Code, data and analysis related to the paper [Counterfactual Probing for the influence of affect and specificity on Intergroup Bias](http://arxiv.org/abs/2305.16409), to be published at Findings of ACL 2023

## Data

Data used in the experiments can be found in the `data/` folder, with annotations for affect, and specificity scores. In addition the columns in [our earlier dataset](https://github.com/venkatasg/interpersonal-bias/tree/main/data), the following columns are added (for details on annotation question, refer to the paper):

- *Feeling*: annotation for feeling question
- *Behavior*: annotation for behavior question
- *Specificity*: specificity score from [specificityTwitter](https://github.com/cs329yangzhong/specificityTwitter)

## Code

The scripts `collect_states.py` and `train_inlp.py` sample hidden state embeddings and train INLP classifiers against our property of interest. `run_alter.py` performs the actual counterfactual intervention, using the learned matrices from INLP.

## Citation

```
@inproceedings{venkat-probing-2023,
    title = {Counterfactual Probing for the influence of Affect and Specificity on Intergroup Bias},
    author = {Govindarajan, Venkata S and Atwell, and Beaver, David I. and Mahowald, Kyle and Li, Junyi Jessy},
    booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
    month = july,
    year = {2023},
    address = {Toronto, Canada},
    publisher = {Association for Computational Linguistics},
    url = {http://arxiv.org/abs/2305.16409}
}
```
