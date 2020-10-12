'''
Run Guild:
    - run in same dir as this file
    - python3 train.py [train_corpus_path] [test_corpus_path1] [test_corpus_path2]
'''

from gensim.models.doc2vec import Doc2Vec
from multiprocessing import cpu_count
from embed_rank.MyCorpus import MyCorpus
import numpy as np
import logging
import sys
from collections import defaultdict
import pickle
import pprint
import time
# verbose mode for model
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
        cur_triplet.append(model.infer_vector(words, epochs=4))
        
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
    return accuracy




# get path parameters
train_corpus_path, test_corpus_path1, test_corpus_path2 = sys.argv[1:]

# create train corpus
TRAIN_CORPUS_PATH = f"extracted_data/train/{train_corpus_path}"
train_corpus = list(MyCorpus(TRAIN_CORPUS_PATH)) # each doc is a TaggedDocument object
# create test corpus
TEST_CORPUS_PATH = f"extracted_data/test/wiki_hand/{test_corpus_path1}"
wiki_hand_test_corpus = list(MyCorpus(TEST_CORPUS_PATH)) # each doc is a TaggedDocument object

TEST_CORPUS_PATH = f"extracted_data/test/wiki_2014/{test_corpus_path2}"
wiki_2014_test_corpus = list(MyCorpus(TEST_CORPUS_PATH)) # each doc is a TaggedDocument object




# search for the best hyperparameters
'''
parameters = {'vector_size':[20, 50, 100, 150, 200, 250, 300], 'min_count':[3, 5, 7, 8, 9, 10],
              'epochs':[10, 20, 30, 40], 'window':[2, 3, 4, 5, 10], 'negative':[5, 7, 9, 11]}
'''
parameters = {'vector_size':[300, 400, 500], 'min_count':[7, 8, 10],
              'epochs':[30, 40, 50], 'window':[2, 3, 4], 'negative':[7, 9, 11]}
num_cores = cpu_count()

s = time.time()
model = None
best_model = None
tunning_log = []
best_para_dict = {}
best_accuracy = 0
for cur_epochs in parameters["epochs"]:
    for cur_vec_size in parameters["vector_size"]:
        for cur_negative in parameters["negative"]:
            for cur_window in parameters["window"]:
                for cur_mc in parameters["min_count"]:
                    '''
                    vector_size: size of embedding
                    min_count: threshold for word freq to filter out. keep all >= min_count
                    dm: 1 to use PV_DM model
                    window: max distance between pred word and cur word
                    workers: number of threads to use
                    epochs: number of time to train over whol corpus
                    hs: If 1, hierarchical softmax will be used for model training. If set to 0, 
                        and negative is non-zero, negative sampling will be used
                    negative: If > 0, negative sampling will be used, the int for negative specifies how many 
                            “noise words” should be drawn (usually between 5-20). If set to 0, no negative 
                            sampling is used
                    dm_mean: if 1, averaging doc_vec and word_vec
                    '''
                    # starting training model
                    model = Doc2Vec(vector_size=cur_vec_size, window=cur_window, min_count=cur_mc, seed=1,
                                    dm=1, workers=num_cores, hs=0, negative=cur_negative, dm_mean=1)
                    model.build_vocab(train_corpus)
                    print("\n===============================================")
                    print("Starting Training model with new parameter:")
                    print("     - Epoch:", cur_epochs)
                    print("     - Vec_Size:", cur_vec_size)
                    print("     - Window:", cur_window)
                    print("     - Negative:", cur_negative)
                    print("     - Min_Count:", cur_mc, "\n\n")
                    model.train(train_corpus, total_examples=model.corpus_count, epochs=cur_epochs)
                    wiki_hand_accuracy = evaluate_triplet(model, wiki_hand_test_corpus)
                    wiki_2014_accuracy = evaluate_triplet(model, wiki_2014_test_corpus)
                    accuracy = (wiki_2014_accuracy+wiki_hand_accuracy)/2
                    print("\n**************************")
                    print(f"Wiki_hand Accuracy: {wiki_hand_accuracy}%")
                    print(f"Wiki_2014 Accuracy: {wiki_2014_accuracy}%")
                    print(f"Accuracy: {accuracy}%")
                    print("**************************\n")
                    tunning_log.append((str(model),f"Epoch: {cur_epochs}",f"Accuracy:{accuracy}"))
                    
                    if accuracy > best_accuracy:
                        print("\n\nModel Updated!!!!!!!!!")
                        best_accuracy = accuracy
                        best_para_dict = {"vector_size":cur_vec_size, "window":cur_window, 
                                        "epochs":cur_epochs, "min_count":cur_mc, 'negative':cur_negative}
                        best_model = model
                    print("===============================================\n")




print("\n\n\n")
pprint.pprint(tunning_log)
print("\n")
pprint.pprint(best_para_dict)
# save the results
best_model.save("model_results/D2V.model")
with open("model_results/tunning_log.pickle", "wb") as fd:
    pickle.dump(tunning_log, fd, protocol=pickle.HIGHEST_PROTOCOL)
with open("model_results/best_para_dict.pickle", "wb") as fd:
    pickle.dump(best_para_dict, fd, protocol=pickle.HIGHEST_PROTOCOL)
print("Total Time Taken", round((time.time()-s)/60, 2), "mins")