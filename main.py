from embed_rank.EmbedRank import EmbedRank
from embed_rank.MyCorpus import MyCorpus

er = EmbedRank()
'''
text = er.extract_information("test_doc.pdf")
doc_token = er.tokenize(text) # tokenize
doc_token = er.pos_tag(doc_token) # pos tagging
doc_token = er.preprocess(doc_token) # final preprocessing
print(doc_token)
'''

TRAIN_CKP_PATH = "extracted_data/text.txt"
DOC2FILE_PATH = "extracted_data/doc2file.txt"

my_corpus = MyCorpus(TRAIN_CKP_PATH, DOC2FILE_PATH)
for doc in my_corpus:
    print(doc)
    ipt = input("press 0 to quit")
    if ipt == 0 or ipt == "0":
        break
