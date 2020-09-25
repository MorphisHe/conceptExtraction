'''
This file test a trained model on its performance of ranking trained
documents against the training corpus.

Run Guild:
    - run in the dir of this script
    - python3 evaluate.py [model_path] [train_corpus_path]
'''
from gensim.models.doc2vec import Doc2Vec
from embed_rank.MyCorpus import MyCorpus
import sys
from collections import Counter
import pprint


model_path, train_corpus_path = sys.argv[1:]
MODEL_PATH = f"model_results/{model_path}"
TRAIN_CORPUS_PATH = f"extracted_data/train/{train_corpus_path}"

model = Doc2Vec.load(MODEL_PATH)
train_corpus = list(MyCorpus(TRAIN_CORPUS_PATH)) # each doc is a TaggedDocument object

ranks = []
for tagged_doc in train_corpus:
    words, doc_id = enumerate(tagged_doc)
    doc_id = doc_id[-1][0] # remove tuple and list
    words = words[-1] # remove tuple

    inferred_vector = model.infer_vector(words, 4)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

counter = Counter(ranks)
pprint.pprint(counter)