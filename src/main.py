import os
import csv
# import pyconll
from gensim.models import Word2Vec
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import seaborn as sns
# import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from configparser import ConfigParser
import conllu

os.chdir('/home/yuki/Dropbox/Arbeit/20181016_古代インド文献成立過程解明への文体計量分析及びデータ可視化の利用/20231218_Workshop/git/Amano2/')

def main():
    config = ConfigParser()
    config.read('src/config.ini')
    # paths = config['paths']
    # strings = config['strings']
    # evaluation_paths = config['evaluation']

    files_excluded = []
    with open(config['paths']['files_excluded'], 'r') as f:
        files_excluded = [line.strip() for line in f.readlines()]

    filelist = []
    for root, dirs, files in os.walk(config['paths']['training_path']):
        if os.path.basename(root) in files_excluded:
            continue
        for file in files:
            file_name, extension = os.path.splitext(file)
            if 'conllu' in extension:
                filelist.append(os.path.join(root,file_name))

    # extract unique file names.
    filelist = list(set(filelist))
    sents = []
    token_cnt = 0
    for f in filelist:
        file_ext = f + '.conllu'
        print(os.path.basename(f))
        while True:
            try:
                data = open(file_ext, mode='r', encoding='utf-8')
                break
            except FileNotFoundError:
                print('No such file or directory')
                file_ext = f + '.conllu_parsed'
                # data = open(file_ext, mode='r', encoding='utf-8')
        sentences = conllu.parse(data.read())
        for sentence in sentences:
            s = []
            for token in sentence:
                s.append(token['lemma'])
                token_cnt += 1
            sents.append(s)

    # show staticstics about the traning corpus
    print('The nubmer of documents: ', len(filelist))
    print('Avg. sentences/document: ', len(sents)/len(filelist))
    print('Avg. tokens/sentence: ', token_cnt/len(sents))

    # Create a model
    # default size = 100, default windows = 5, default min_count=5, sg is Training algorithm 1 for skip-gram; otherwise CBOW.
    w2v_model = Word2Vec(sents, workers=4, sg=1)
    trans_model = SentenceTransformer(config['strings']['model_name'])

    # Show similar words with the paramater. Default measure is Cosine Similarity
    print(w2v_model.wv.most_similar('keśa'))

    # load files to be compared with each other.
    compared_documents = pd.read_csv(config['paths']['compared_documents'], sep='\t', header=None)
    for key in config['evaluation']:
        files = [root + file for file in files for root, dirs, files in os.walk(config['evaluation'][key])]

    # Create embeddings
    embeddings = {'word2vec': {}, 'transformers': {}}
    for file in files:
        df = pd.read_csv(file, sep=',')
        for i, row in df.iterrows():
            sent = row['lemma'].split(' ')
            embeddings['transformers'][row['index']] = trans_model.encode(row['lemma'])

    # todo: make embeddings of word2vec!!!
    # embeddings += create_transformer_embeddings(trans_model, )
    # vectors = Create_Chapters_Vectors(os.getcwd() + '/' + target_file, w2v_model)

if __name__ == '__main__':
    main()