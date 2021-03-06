'''
This file converts a data file to 1 document per line format

Run Guild: 
    - run in same dir as this file
    - python3 doc_per_line.py [test|train] [optional (arxiv|wiki_hand|wiki_2014)] [corpus_path] [doc2file_path] [output_path]
'''
from smart_open import open
import time
import sys

def docID_to_filename(doc2file_path):
    # get next document's file_name
    for line in open(doc2file_path):
        # list contain [doc_id, filename]
        yield line.replace("\n", "").split(",", 1)


def convert(corpus_path, doc2file_path, output_path):
    corpus_fd = open(corpus_path, "r")
    doc2file_fd = open(doc2file_path, "r")
    out_fd = open(output_path, "w")
    doc2file_gen = docID_to_filename(doc2file_path)

    cur_docID = '0'
    cur_Doc = ''
    counter = 1
    for line in corpus_fd:
        '''
        # for visualization only
        if counter%100000==0:
            print("@", end="")
        else:
            print("#", end="")
        counter += 1
        '''

        doc_ID, _, ckp = line.replace("\n", "").split(",")
        if doc_ID == cur_docID:
            cur_Doc = cur_Doc + " " + ckp
        else:
            id_key, filename = next(doc2file_gen)
            if id_key != cur_docID:
                print(f"Unable to get document name.  id_key:{id_key} cur_docID:{cur_docID}")
                return False
            cur_Doc = cur_Doc.strip()
            out_fd.write(f"{filename},{cur_Doc}\n")

            # reset variables
            cur_docID = doc_ID
            cur_Doc = ckp
    # finish the last document
    id_key, filename = next(doc2file_gen)
    if id_key != cur_docID:
                print(f"Unable to get document name.  id_key:{id_key} cur_docID:{cur_docID}")
                return False
    cur_Doc = cur_Doc.strip()
    out_fd.write(f"{filename},{cur_Doc}\n")

    corpus_fd.close()
    doc2file_fd.close()
    out_fd.close()
    return True



# running the script
test_or_train = sys.argv[1]
triplet_version, ckp_path, doc2file_path, output_path = 0,0,0,0
CORPUS_PATH, DOC2FILE_PATH, OUTPUT_PATH = 0,0,0
if test_or_train == "test":
    triplet_version, ckp_path, doc2file_path, output_path = sys.argv[2:]
    CORPUS_PATH = f"../extracted_data/{test_or_train}/{triplet_version}/{ckp_path}"
    DOC2FILE_PATH = f"../extracted_data/{test_or_train}/{triplet_version}/{doc2file_path}"
    OUTPUT_PATH = f"../extracted_data/{test_or_train}/{triplet_version}/{output_path}"
elif test_or_train == "train":
    ckp_path, doc2file_path, output_path = sys.argv[2:]
    CORPUS_PATH = f"../extracted_data/{test_or_train}/{ckp_path}"
    DOC2FILE_PATH = f"../extracted_data/{test_or_train}/{doc2file_path}"
    OUTPUT_PATH = f"../extracted_data/{test_or_train}/{output_path}"

if convert(CORPUS_PATH, DOC2FILE_PATH, OUTPUT_PATH):
    s = time.time()
    print("Convert training data to one document per line done!!!")
    print("Total Time Taken: {}mins".format(round((time.time()-s)/60, 2)))