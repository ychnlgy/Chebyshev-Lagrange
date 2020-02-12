# Original raw code for the Chebyshev-Lagrange experiments

Dependencies include: python3, numpy, scipy, sklearn, torch, tqdm, matplotlib, pandas

These may not be comprehensive and the user is encouraged to install whatever seems to be missing.

Implementation files of immediate interest are:
 - [```src/modules/polynomial/chebyshev.py```](src/modules/polynomial/chebyshev.py) for the Chebyshev nodes.
 - [```src/modules/polynomial/Activation.py```](src/modules/polynomial/Activation.py) for the basic Chebyshev-Lagrange activation.
 - [```src/modules/polynomial/ChebyshevActivation.py```](src/modules/polynomial/ChebyshevActivation.py) for the weighted Chebyshev polynomial activation.
 - [```src/modules/polynomial/LagrangeBasis.py```](src/modules/polynomial/LagrangeBasis.py) for the Lagrangian bases and gradients.
 - [```src/modules/polynomial/RegActivation.py```](src/modules/polynomial/RegActivation.py) for Chebyshev-Lagrange regression.
 - [```src/modules/polynomial/LinkActivation.py```](src/modules/polynomial/LinkActivation.py) for Chebyshev-Lagrange extrapolation.

**Run all code in the directory of this README, unless otherwise specified!**

## Synthetic datasets

Folders ```results-0.01``` and ```results-0.04``` hold the results for 0.01 and 0.04 standard deviations of Gaussian noise we produced in the paper. Note: "bogo" = "jump" in the paper. Files are named by ```<data_points>N-<noise_level>n-<model>-R<repeat_number>.png```. Models are: "actpoly" = basic Chebyshev-Lagrange (CL), "chebypoly" = weighted Chebyshev polynomials, "linkpoly" = CL-extrapolate, "regpoly" = CL-regression, "polyproto" = prototype-cosine-similarity CL, "reludeep" = ReLU (2x depth), "relurelu" = ReLU (2x layers per block) and "tanhpoly" = tanh-CL.

To reproduce the results on the synthetic data, do:

```
python3 main.py datanames=pendulum,arrhenius,bogo,gravity,prelu,sigmoid,step tiers=polyproto,tanhpoly,chebypoly,actpoly,cubic,linkpoly,regpoly,relu,tanh,relurelu,reludeep noise=0.01 N=1000 repeat=10
```

This will run most of the models in [```src/toy/polynomial.py```](src/toy/polynomial.py). The command options are described here:
  - ```tiers=polyproto,...``` refers to the ```Tier_*``` classes in this file
  - ```datanames=pendulum,...``` refers to the nicknames of the ```Dataset*``` classes in [```src/toy/datasets.py```](src/toy/datasets.py)
  - ```noise=0.01``` means 0.01 standard deviations of noise will be added to the output. In the experiments we also do ```noise=0.04```
  - ```N=1000``` means 1000 random samples will be generated per training and testing set
  - ```repeat=10``` means the procedure will be repeated with 10 random seeds

The results will be saved in a folder named ```results/``` in the current working directory. The images at the leaves of this directory contain the test RMSE scores, which need to be averaged across all 10 repeats per model per dataset to replicate the reported results.

## MNIST/CIFAR-10

```shake-shake_pytorch/results.txt``` holds the results we produced. "linkact" refers to the CL-extrapolate version.

To reproduce the results on these popular image classification tasks, change directories to the fork of the Shake-Shake architecture in ```shake-shake_pytorch```. Run:

```
python3 main.py 3
```

where the program will **repeat 3x with random seeds** the training and testing of a 14 layer residual network with activations of either Chebyshev-Lagrange extrapolation, Chebyshev-Lagrange regression or ReLU on MNIST and CIFAR-10. The results will be stored in "history.txt" in the current working directory.

## DementiaBank

Our results of 30 random initializations and sets of 10 fold cross validation are stored in ```linkvsrelu-visual.json```. To visualize these results, run:

```python3 linkvsrelu_report.py```.

We cannot provide the datasets for legal reasons. Please follow the following code to reproduce the above results.

#### Prerequisites
Make sure ```../../data/```, which in turn contains files directly downloaded from the authors of <https://arxiv.org/pdf/1811.12254.pdf> and renamed into:
  - ```db-34-am.csv``` (DementiaBank, version 34, American), 
  - ```ha-34-am.csv``` (HealthyAging, version 34, American), 
  - ```ha-34-ca.csv``` (HealthyAging, version 34, Canadian), 
  - ```fp-34-am.csv``` (FamousPeople, version 34, American), 
  - ```uf-34-am.csv``` (FamousPeople 5-utterances, version 34, American) and 
  - ```dfd_features2.json```, which should contain the following (which is replicated from <https://arxiv.org/pdf/1811.12254.pdf>):

```
["long_pause_count_normalized", "medium_pause_duration", "mfcc_skewness_13", "mfcc_var_32", "TTR", "honore", "tag_IN", "tag_NN", "tag_POS", "tag_VBD", "tag_VBG", "tag_VBZ", "pos_NOUN", "pos_ADP", "pos_VERB", "pos_ADJ", "category_inflected_verbs", "prp_ratio", "nv_ratio", "noun_ratio", "speech_rate", "avg_word_duration", "age_of_acquisition", "NOUN_age_of_acquisition", "familiarity", "VERB_frequency", "imageability", "NOUN_imageability", "sentiment_arousal", "sentiment_dominance", "sentiment_valence", "graph_avg_total_degree", "graph_pe_directed", "graph_lsc", "graph_density", "graph_num_nodes", "graph_asp", "graph_pe_undirected", "graph_diameter_undirected", "Lu_DC", "Lu_DC/T", "Lu_CT", "local_coherence_Google_300_avg_dist", "local_coherence_Google_300_min_dist", "constituency_NP_type_rate", "constituency_PP_type_prop", "constituency_PP_type_rate", "ADJP_->_JJ", "ADVP_->_RB", "NP_->_DT_NN", "NP_->_PRP", "PP_->_TO_NP", "ROOT_->_S", "SBAR_->_S", "VP_->_VB_ADJP", "VP_->_VBG", "VP_->_VBG_PP", "VP_->_VBG_S", "VP_->_VBZ_VP", "avg_word_length", "info_units_bool_count_object", "info_units_bool_count_subject", "info_units_bool_count", "info_units_count_object", "info_units_count_subject", "info_units_count"]
```

To reproduce the DementiaBank results of 84.7% accuracy (with 0.8% standard deviation) and its baselines, do:

```
python3 db_linkvsrelu.py repeats=30
```

This will repeat **with 30 different random seeds** the 10-fold cross-validation training and testing of 4 models with the same architecture but different activations ("linkact" := Chebyshev-Lagrange with extrapolation, "relu" := ReLU, "rand" := majority class predictor, "tanh":= Tanh) on DementiaBank v34. There should be 407 rows and 66 features. 178 rows should have the dementia label. Per repeat, the results of the 10 folds will be averaged and saved to "linkvsrelu.json" in the current working directory. To view the average, standard deviation, t-statistics and P-values of the repeats between the different activations, rename "linkvsrelu.json" to "linkvsrelu-visual.json" and run:

```
python3 linkvsrelu_report.py
```

To reproduce the best macro F1 score, do:

```
python3 db_linkvsrelu.py repeats=1 seed0=6
```

These are the results written on the paper.
