# Word relation detector
- connlu.py - parser
- binarize.py - base binarization utils
- syntagrusbin.py - specific syntagrus utils
- prepare_sets.py - balance, test/train split, scaling to train data
- scalers.py - scalers
- models.py - SVM, LogRegression (partial_fit and batches not supported yet)


# Installation (Linux)

```
conda env create -f environment.yml
conda activate rel
```

# Training model
```*_setup_config_.json``` - setup config.
The fields:

- exp_dir: experiment directory
- dataset_path: path to .conllu file
- max_sentences: max sentences in one batch (binarization)
- n_predictors: number of predictors (cascade)
- do_sort_bin: sort the classes when splitting
- batch_size, batch_sentence: batch sizes for bert
- use_bert: use bert
- bert_name: name of bert model
- unk_token: token for unknown values
- use_head: use head when analyzing
- predict_head: train head predictor
- use_pos: use part of speech when analyzing
- use_morph: use features when analyzing
- use_sentenceemb: use sentence embedding (bert)
- use_distance: use distance between words in sentence
- use_s_no: use position of a word
- binary: predict absence/presence of relation, not relation itself
- strip_colon: strip everything after the colon in relation name
- ignored: list of ignored relations (treated as absence)
- scaler: scaler (from scalers.py)
- methods: balance methods
- test_size: ratio of test subset (not used during training)
- k: k neighbours for knn-based methods
- model: model from models.py
- zero_threshold: when the max of the one hot embedding is less than this value, it is considered an empty vector
- stages: which stages to initiate (1 - parsing, 2 - binarization, 3 - balance, 4 - training)

Use it in ```main.py``` (CLI input)
