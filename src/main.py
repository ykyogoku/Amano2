import os
from configparser import ConfigParser
from functions import load_training_dataset, create_embeddings, compare_embeddings

os.chdir('/home/yuki/Dropbox/Arbeit/20181016_古代インド文献成立過程解明への文体計量分析及びデータ可視化の利用/20231218_Workshop/git/Amano2/')

# TODO: include chapter and lemma200.
def main():
    config = ConfigParser()
    config.read('src/config.ini')

    # extract training dataset for Word2Vec
    sents = load_training_dataset(config)

    # Create embeddings
    embeddings = create_embeddings(config, sents)

    # load files to be compared with each other.
    compare_embeddings(config, embeddings)

if __name__ == '__main__':
    main()