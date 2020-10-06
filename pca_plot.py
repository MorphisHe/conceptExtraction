'''
This file plots a document and it's ckps to 2d graph

Run Guild:
    - run in the dir of this script
    - python3 pca_plot.py [mode=(w2v | d2v)] [model_path]
        - if "w2v" is passed, plots the word matrix in model to 2d graph
        - if "d2v" is passed, plots doc, ckps, selected ckps to 2d graph
'''

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from embed_rank.EmbedRank import EmbedRank
import numpy as np
import time
import sys
import pandas as pd
import pprint

from sklearn.manifold import TSNE

def pca_w2v_plot(model, top_n_words=150, n_components=2):
    '''
    Display the word2vec in 2d plot using PCA

    Parameter:
    ---------------
    model: Doc2Vec model

    top_n_words: (int) number of words to plot

    n_components: [2 | 3] number of dim to reduce the vectors to
    '''
    # get all word embeddings, model.wv.index2entity returns list of vocab with highest freq at top
    X = model[model.wv.index2entity][:top_n_words]

    # decomposition
    pca = PCA(n_components=n_components)
    reduced_X = pca.fit_transform(X)

    # create scatter plot and annotate each point with word string
    ax = sns.scatterplot(x=reduced_X[:, 0], y=reduced_X[:, 1])
    words = list(model.wv.index2entity)[:top_n_words]
    for i, word in enumerate(words):
        ax.annotate(word, xy=(reduced_X[i, 0], reduced_X[i, 1]))
    plt.title(f"PCA Decomposition Of Word2Vec's Top {top_n_words} Words")
    plt.show()


def doc_kps_plot(er, DOC_PATH, n_components=2, top_n=50, beta=0.55):
    doc_tag = DOC_PATH
    text = er.extract_information(doc_tag)
    doc_token = er.tokenize(text) # tokenize
    doc_token = er.pos_tag(doc_token) # pos tagging
    doc_token = er.preprocess(doc_token) # final preprocessing
    doc_embed, ckps_embed = er.embed_doc_ckps(doc_tag, doc_token) # embed to vector space
    selected_ckps, selected_ckps_idxs= er.mmr(doc_embed, ckps_embed, top_n=top_n, beta=beta) # pick top_n kp using mmr
    pprint.pprint(selected_ckps)

    # construct dataset using pandas
    embeds = np.append(np.array(list(ckps_embed.values())), doc_embed[1].reshape(1, -1), axis=0)
    labels = ["unselected_ckp"]*len(ckps_embed)
    labels.append("doc")
    for index in selected_ckps_idxs:
        labels[index] = "selected_ckp"

    # decomposition
    #pca = PCA(n_components=n_components)
    model = TSNE(n_components=n_components, random_state=1, perplexity=50, n_iter=5000)
    #embeds = pca.fit_transform(embeds)
    embeds = model.fit_transform(embeds)
    data = pd.DataFrame({
        "v1": embeds[:, 0],
        "v2": embeds[:, 1],
        "label": labels
    })

    ax = sns.scatterplot(x=data.v1, y=data.v2, style=data.label, hue=data.label)
    # annotating each point
    for i, ckp in enumerate(list(ckps_embed.keys())):
        if i in selected_ckps_idxs:
            if len(ckp) > 20:
                ckp = ckp[:20] + '...'
            ax.annotate(ckp, xy=(embeds[i, 0], embeds[i, 1]))

    plt.title(f"EmbedRank Wtih top {top_n} Keyphrases")
    plt.show()

    



MODE, MODEL_PATH = sys.argv[1:]
s = time.time()

er = EmbedRank(model_path=f"model_results/{MODEL_PATH}")
if MODE == "w2v":
    pca_w2v_plot(er.model)
elif MODE == "d2v":
    DOC_PATH = "test_doc.pdf"
    doc_kps_plot(er, DOC_PATH, top_n=20, beta=1)

print("Total Time Taken", round((time.time()-s), 2), "secs")