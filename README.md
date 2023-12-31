# Amano2
## Training Corpus
- Training Corpus is Dr. [Oliver Hellwig's Sanskrit corpus](https://github.com/OliverHellwig/sanskrit/tree/master/dcs/data/conllu/files).
- Upon training, the documents used for evaluation (viz., MS, KS and TS) are excluced.

## Models
### Word2Vec
- workers (worker threads to train the model) is set to 4.
- sg (Training algorithm) is set to 1 (skip-gram).
- Other parameters are default. Regarding the other parameters, see [the document](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html).

### Transformers
- SentenceTransformer is used.
- GPT-2 is used as the model (default).
- Other parameters are default. Regarding the other parameters, see [the document](https://www.sbert.net/docs/package_reference/SentenceTransformer.html).
- [chronbmm/xlm-roberta-vedic](https://huggingface.co/chronbmm/xlm-roberta-vedic) is used as the model for Sanskrit.

## Evaluation
See output folder.
### Corpus Division
The evaluation dataset is devided with the following units:
- chapter (average? and distribution?)
- 200 tokens (average? and distribution?)
- 100 tokens (average? and distribution?)
- 20 tokens (average? and distribution?)

### Chapters to be compared
1. MS.1.1 vs. MS.1.6 
2. MS.1.1 vs. MS.1.7 
3. MS.1.6 vs. KS.8 
4. MS.1.7 vs. KS.9.1

## TODO
- adding statistics like avg. tokens/sentence, distribution of words, etc.