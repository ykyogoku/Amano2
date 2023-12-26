#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 21:42:25 2021

@author: yuki
"""
import os
import conllu
# import pyconll
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from simcse import SimCSE

def Type_Process(file_paths, input_type):
    if input_type == 'plain':
        chapter = process_plain(file_paths)
    else:
        chapter = Shared_Fucntion(file_paths, input_type)

    return chapter

def process_plain(file_paths):
    text_conllu = dict()
    texts = file_paths['target_folders'].split(',')
    for t in texts:
        folder_path = file_paths['target_dir'] + t
        for filename in os.listdir(folder_path):
            if filename.endswith(".conllu_parsed"):
                # if the file ends with ".conllu_parsed", load it using conllu library
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    data = f.read()
                # parse the data using conllu
                text_conllu[filename.split('-')[2]] = conllu.parse(data)

    # concatenate the sentences.
    text_conllu = {k: v for k, v in text_conllu.items() if v}
    chapter = dict()
    for k, v in text_conllu.items():
        chapter[k] = str()
        for sent in text_conllu[k]:
            if len(sent) > 0 and isinstance(sent, conllu.models.TokenList):
                chapter[k] += sent.metadata['text'] + '. '

    return chapter

def Shared_Fucntion(file_paths, input_type):
    texts = file_paths['target_folders'].split(',')
    chapter = {}
    for t in texts:
        folder_path = file_paths['target_dir'] + t
        for filename in os.listdir(folder_path):
            if filename.endswith(".conllu_parsed"):
                key = filename.split('-')[2]
                chapter[key] = str()
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    data = f.read()
                if '\t\t' in data:
                    data = data.replace('\t\t', '\t_\t')
                conll = pyconll.load_from_string(data)
                for sentence in conll:
                    sent = str()
                    for token in sentence:
                        if input_type == 'no-sandhi':
                            if len(token.misc) > 0:
                                sent += list(token.misc['Unsandhied'])[0] + ' '
                        elif input_type == 'lemma':
                            if isinstance(token.lemma, str):
                                sent += token.lemma + ' '
                    if len(sent) > 0:
                        sent += '.'
                    sent = sent.replace(' .', '. ')
                    chapter[key] += sent

    chapter = {k: v for k, v in chapter.items() if v != ''}
    return chapter

def create_transformer_embedding(chapter, target, model):
    embedding = dict()
    for k, v in chapter.items():
        embedding[k] = model.encode(v)

    # extract MS chapters in the list of target chapters.
    MS = dict()
    for ch, emb in embedding.items():
        if ch in target.index:
            MS[ch] = emb
    MS = pd.DataFrame.from_dict(MS).transpose()

    # remove MS chapters from the dict embedding.
    cmp_chapters = {key: value for key, value in embedding.items() if 'MS' not in key}
    cmp_chapters = pd.DataFrame.from_dict(cmp_chapters).transpose()

    return MS, cmp_chapters

def Evaluator(top_chapters, target_chapters):
    """
    :param top_chapters:
    :param target_chapters:
    :return: precision, recall, F1
    """
    TP = 0
    FN = 0
    TP_FP = 0
    for index, row in top_chapters.iterrows():
        if index in target_chapters.index:
            chap_temp = target_chapters.loc[index].dropna()
            top_list = [j[0] for j in row]
            TP_FP += len(top_list)
            if len(chap_temp) <= 1:
                if chap_temp[1] in top_list:
                    TP += 1
                else:
                    FN += 1
            else:
                for chap in chap_temp:
                    if chap in top_list:
                        TP += 1
                    else:
                        FN += 1
    precision = TP / TP_FP
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, F1


def Create_Embedding_Doc2Vec(chapter, target, key):

    # create tagged texts
    training_docs = []
    for k, v in chapter.items():
        tags = [k]
        sentence = v
        words = sentence.split(' ')
        training_docs.append(TaggedDocument(words, tags))
        
        
    print('Generating model')
    # hyperparameter
    # https://arxiv.org/pdf/1607.05368.pdf

    # config
    vector_size = 300
    epoch_num = 1000
    # PV-DM: 1, PV-DBOW: 0
    dm = 1

    model_path = f'./model/doc2vec_{key}.model'
    if os.path.exists(model_path):
        model = Doc2Vec.load(f'./model/doc2vec_{key}.model')

    else:
        model = Doc2Vec(documents=training_docs, dm=1, vector_size=vector_size, alpha=0.025, min_alpha=0.025, window=5, min_count=1)
    
        for epoch in tqdm(range(epoch_num)):
            model.train(training_docs, total_examples=model.corpus_count, epochs=model.epochs)
            model.alpha -= (0.025 - 0.0001) / (epoch_num  - 1)
            model.min_alpha = model.alpha

        model_name = f'./model/doc2vec_{key}.model'
        print('Saved model', model_name)
        model.save(model_name)


    # embedding
    embedding = dict()
    for k, v in chapter.items():
        embedding[k] = model.infer_vector(v.split(' '))

    # extract MS chapters in the list of target chapters.
    MS = dict()
    for ch, emb in embedding.items():
        if ch in target.index:
            MS[ch] = emb
    MS = pd.DataFrame.from_dict(MS).transpose()

    # remove MS chapters from the dict embedding.
    cmp_chapters = {key: value for key, value in embedding.items() if 'MS' not in key}
    cmp_chapters = pd.DataFrame.from_dict(cmp_chapters).transpose()

    return MS, cmp_chapters

    return(model)

# This function should be modified.
def Create_Chapters_Vectors(target_dir, model):
    content = Clean_File(target_dir)
    lines = content.split(sep='\n')

    cnt = 0
    chapters = {}
    line_str = str()
    key_name = str()
    for line in lines:
        if 'chapter_id' in line:
            if key_name != '':
                chapters[key_name] = line + '\n' + line_str
                line_str = str()
            key_name = lines[cnt - 1].split(': ')[1]
        else:
            line_str += line + '\n'
        cnt += 1

    vectors = {}
    vocab_list = list(model.wv.vocab)
    for i, chapter in chapters.items():
        corpus = pyconll.load_from_string(chapter)
        cnt = 0
        vectors[i] = 0
        for j, sentence in enumerate(corpus):
            for token in sentence:
                if token.lemma in vocab_list:
                    vectors[i] += model[token.lemma]
                    cnt += 1
        if i != 0:
            vectors[i] = vectors[i] / cnt

    return vectors