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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from pylab import savefig
from statistics import mean
from matplotlib import pyplot as plt

os.chdir('/home/yuki/Dropbox/Arbeit/20181016_古代インド文献成立過程解明への文体計量分析及びデータ可視化の利用/20231218_Workshop/git/Amano2/')

# TODO: include chapter and lemma200.
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

    # load files names to be processed.
    file_path = {}
    for key in config['evaluation']:
        # print(key)
        file_path[key] = []
        for root, dirs, files in os.walk(config['evaluation'][key]):
            for file in files:
                file_path[key].append(root + file)


    # Create embeddings
    embeddings = {'word2vec': {}, 'transformers': {}}
    vocab_list = list(w2v_model.wv.index_to_key)
    for k, v in file_path.items():
        embeddings['word2vec'][k] = {}
        embeddings['transformers'][k] = {}
        for file in v:
            print(file)
            df = pd.read_csv(file, sep=',')
            for i, row in df.iterrows():
                sent = row['lemma'].split(' ')
                cnt = 0
                embedding_temp = 0
                for token in sent:
                    if token in vocab_list:
                        embedding_temp += w2v_model.wv[token]
                        cnt += 1
                if cnt == 0:
                    print('The following devision does not contain any word in the trained model.')
                    print(file, row['index'])
                else:
                    embeddings['word2vec'][k][row['index']] = embedding_temp/cnt
                embeddings['transformers'][k][row['index']] = trans_model.encode(row['lemma']) # transfomers

    # load files to be compared with each other.
    compared_documents = pd.read_csv(config['paths']['compared_documents'], sep='\t', header=None)
    comparison = {}
    for model in embeddings.keys():
        comparison[model] = {}
        for div in embeddings[model].keys():
            comparison[model][div] = {}
            for index, row in compared_documents.iterrows():
                compared_1, compared_2 = row
                comparison_name = '-'.join([compared_1, compared_2])
                comparison[model][div][comparison_name] = {}
                # comparison[model][div][comparison_name] = pd.DataFrame()
                comp_temp_1 = {}
                comp_temp_2 = {}
                for key, value in embeddings[model][div].items():
                    if compared_1 in key:
                        comp_temp_1[key] = value
                    if compared_2 in key:
                        comp_temp_2[key] = value
                for c1 in comp_temp_1.keys():
                    for c2 in comp_temp_2.keys():
                        # c_name = '-'.join([c_1, c_2])
                        vec1 = comp_temp_1[c1].reshape(1, -1)
                        vec2 = comp_temp_2[c2].reshape(1, -1)
                        comparison[model][div][comparison_name][(c1, c2)] = cosine_similarity(vec1, vec2)[0][0]
                        # print(comparison[model][div][comparison_name][(c1, c2)])

                # export similarity datasets
                # similarity_avg = mean(comparison[model][div][comparison_name].values())
                data_expo = [(key[0], key[1], value) for key, value in comparison[model][div][comparison_name].items()]
                df_expo = pd.DataFrame(data_expo, columns=['c1', 'c2', 'value'])
                # df_pivot = df_expo.pivot(index='c1', columns='c2')['value']
                df_pivot = df_expo.pivot(index='c1', columns='c2', values='value')
                folder_path = '/'.join(['output', model, div, comparison_name])
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                df_path = '/'.join([folder_path, 'similarity.tsv'])
                df_pivot.to_csv(df_path, sep='\t')

                # generate and export heatmaps
                fig = plt.figure()
                figure_path = '/'.join([folder_path, 'heatmap.png'])
                ax = sns.heatmap(df_pivot)
                ax.figure.tight_layout()
                plt.savefig(figure_path)
                plt.close()



if __name__ == '__main__':
    main()