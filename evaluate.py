'''
This file test a trained model on either training set or test set.

Run Guild:
    - run in the dir of this script
    - python3 evaluate.py [train|test] [model_path] [corpus_path]
        - if "train" is passed, the "corpus_path" need to point to training dataset
        - if "test" is passed, the "corpus_path" need to point to a triplet dataset
'''
from gensim.models.doc2vec import Doc2Vec
from embed_rank.MyCorpus import MyCorpus
import sys
import pprint
import time


train_or_test, model_path, corpus_path = sys.argv[1:]
MODEL_PATH = f"model_results/{model_path}"
CORPUS_PATH = f"extracted_data/{train_or_test}/{corpus_path}"

model = Doc2Vec.load(MODEL_PATH)
corpus = list(MyCorpus(CORPUS_PATH)) # each doc is a TaggedDocument object


s = time.time()
if train_or_test == "train":
    from collections import Counter
    ranks = []
    for tagged_doc in corpus:
        words, doc_id = enumerate(tagged_doc)
        doc_id = doc_id[-1][0] # remove tuple and list
        words = words[-1] # remove tuple

        inferred_vector = model.infer_vector(words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

    counter = Counter(ranks)
    pprint.pprint(sorted(counter.items()))
else:
    import numpy as np
    def cos_sim(vec1, vec2):
        dot = np.dot(vec1, vec2.T)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot/norms

    def evaluate_triplet(model, test_corpus, print_out=False):
        num_triplets = 0
        num_correct = 0
        cur_triplet = []
        for tagged_doc in test_corpus:
            words,_ = enumerate(tagged_doc)
            words = words[-1] # remove tuple
            cur_triplet.append(model.infer_vector(words))
            
            if len(cur_triplet) == 3:
                a, p, n = cur_triplet
                if cos_sim(a, p) > cos_sim(p, n):
                    num_correct += 1
                num_triplets += 1
                cur_triplet = []
        accuracy = round((num_correct/num_triplets)*100, 3)
        if print_out:
            print("\n==========================================================")
            print("Model:", model)
            print("Accuracy:", accuracy)
            print("==========================================================\n")
    
    evaluate_triplet(model, corpus, print_out=True)

    
print("Total Time Taken", round((time.time()-s)/60, 2), "mins")