'''
This file extracts text from triplet dataset

Run Guild:
    - python3 [arix|wiki_2014|wiki_hand] max_triplet
        - max_triplet = number of triplet set to extract
'''

from embed_rank.EmbedRank import EmbedRank
import os
import sys
import time

def gen_triplet_data(file_path, ckp_path, doc2file_path, er, max_triplet=999999):
    # arxiv dataset contains  60k documents
    # WIKI_2014: 20K documents
    # wiki_hand: < 1k
    lines = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    ckp_fd = open(ckp_path, 'w+')
    ckp_fd.truncate(0)

    doc2file_fd = open(doc2file_path, 'w+')
    doc2file_fd.truncate(0)

    doc_id = 0
    num_triplet_done = 0
    for line in lines:
        if num_triplet_done == max_triplet:
            break

        file_paths = line.split() # a line is 3 paths. (pos, pos, neg)
        cur_triplet = [] # map doc_id -> (doc_token, file_path)
        for file_path in file_paths:
            text = er.extract_information(file_path)
            # handle server error
            retry_time = 0
            while text == "" and retry_time < 5:
                os.system("ps aux | grep java | pgrep -f Tika | xargs -I{} kill -9 {}")
                er = EmbedRank()
                er.extract_information(file_path)
                retry_time += 1
            if text == "":
                print("Skip current triplet due to failing to extract text of atleast 1 document.")
                break
            doc_token = er.tokenize(text) # tokenize
            doc_token = er.pos_tag(doc_token) # pos tagging
            doc_token = er.preprocess(doc_token) # final preprocessing
            
            if len(doc_token) == 0:
                print("Skip current triplet due to failing to extract text of atleast 1 document.")
                break
            cur_triplet.append((doc_id, doc_token, file_path))
            doc_id += 1
        
        # if all 3 triplet is readed to buffer
        if len(cur_triplet) == 3:
            num_triplet_done += 1
            for docID, doc_token, file_path in cur_triplet:
                # writing candidate key phrases to txt file in csv style
                for sent_id in range(len(doc_token)):
                    for ckp_token in doc_token[sent_id]:
                        ckp_fd.write(f"{docID},{sent_id},{ckp_token}\n")
                
                # write doc_id, file_path pair to txt file
                doc2file_fd.write(f"{docID},{file_path}\n")

    ckp_fd.close()
    doc2file_fd.close()




args = sys.argv[1:]
mode = args[0]
max_triplet = 500
if len(args) > 1:
    max_triplet = args[-1]

er = EmbedRank()
if mode == "arxiv":
    ARXIV_PATH = "extracted_data/triplets_data/arxiv_2014_09_27_examples.txt"
    CKP_PATH = "extracted_data/test/arxiv/arxiv_ckp.txt"
    DOC2FILE_PATH = "extracted_data/test/arxiv/arxiv_doc2file.txt"

    s = time.time()
    gen_triplet_data(ARXIV_PATH, CKP_PATH, DOC2FILE_PATH, er=er, max_triplet=max_triplet)
    print("\nTotal Time Taken for Extracting Arxiv: {}mins".format(round((time.time()-s)/60, 2)))
elif mode == "wiki_hand":
    WIKI_PATH = "extracted_data/triplets_data/wikipedia-hand-triplets-release.txt"
    CKP_PATH = "extracted_data/test/wiki_hand/wiki_hand_ckp.txt"
    DOC2FILE_PATH = "extracted_data/test/wiki_hand/wiki_hand_doc2file.txt"

    s = time.time()
    gen_triplet_data(WIKI_PATH, CKP_PATH, DOC2FILE_PATH, er=er)
    print("\nTotal Time Taken for Extracting Wiki-hand: {}mins".format(round((time.time()-s)/60, 2)))
elif mode == "wiki_2014":
    WIKI_PATH = "extracted_data/triplets_data/wikipedia_2014_09_27_examples.txt"
    CKP_PATH = "extracted_data/test/wiki_2014/wiki_2014_ckp.txt"
    DOC2FILE_PATH = "extracted_data/test/wiki_2014/wiki_2014_doc2file.txt"

    s = time.time()
    gen_triplet_data(WIKI_PATH, CKP_PATH, DOC2FILE_PATH, er=er, max_triplet=max_triplet)
    print("\nTotal Time Taken for Extracting Wiki-2014: {}mins".format(round((time.time()-s)/60, 2)))