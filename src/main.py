import os
from configparser import ConfigParser
from gensim.models import Word2Vec
from functions import load_training_dataset, create_embeddings, compare_embeddings, create_w2vmodel, cal_stat # When executing this script in console, it should be src/functions

# TODO: include lemma200.

def main():
    config = ConfigParser()
    config.read('src/config.ini')

    # extract training dataset for Word2Vec
    if not os.path.exists(config['paths']['model_path']):
        sents = load_training_dataset(config)
        w2v_model = create_w2vmodel(config, sents)
    else:
        w2v_model = Word2Vec.load(config['paths']['model_path'])

    # Create embeddings
    embeddings = create_embeddings(config, w2v_model)

    # load files to be compared with each other.
    comparison = compare_embeddings(config, embeddings)

    # calculate statistics
    cal_stat(comparison)

if __name__ == '__main__':
    main()