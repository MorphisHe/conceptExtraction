import os
from smart_open import open
from embed_rank.EmbedRank import EmbedRank

def gen_corpus(directory, seen_dir, doc2file_path, corpus_path, doc_id):
    '''
    This method does the following:
        - extract all raw text pdf from current directory and all sub-directories
        - extract all candidate key phrases and preprocess them from each doc
        - store (doc_id, sent_id, ckp) data as txt file
        - store (doc_id, file_path) data as txt file

    Parameter:
    ---------------
    directory: root directory

    seen_dir: stores all directories that is already seen (for recursion)

    doc2file_path: file path to write (doc_id, path_name) data

    corpus_path: file path to write (doc_id, sent_id, ckp) data

    doc_id: int to indicate current document id (unqiue)

    Return:
    ---------------
    seen_dir: for recursion

    doc_id: for recursion
    '''

    print("New Directory")
    doc2file_fd = open(doc2file_path, "a")
    corpus_fd = open(corpus_path, "a")
    for filename in os.listdir(directory):
        # visualization only
        if (doc_id+1)%100 == 0:
            print(str(doc_id))
        else:
            print("#", end="")

        path = directory + '/' + filename
        # if this is a dir
        if os.path.isdir(path) and path not in seen_dir:
            doc2file_fd.close()
            corpus_fd.close()
            seen_dir.append(path)
            seen_dir, doc_id = gen_corpus(path, seen_dir, doc2file_path, corpus_path, doc_id)
            doc2file_fd = open(doc2file_path, "a")
            corpus_fd = open(corpus_path, "a")
        # if this is a file/pdf
        else:
            text = er.extract_information(path)
            # check if text is None
            if text is None or text is '':
                continue # TODO handle scanned pdf
            else:
                doc_token = er.tokenize(text) # tokenize
                doc_token = er.pos_tag(doc_token) # pos tagging
                doc_token = er.preprocess(doc_token) # final preprocessing

                # writing candidate key phrases to txt file in csv style
                for sent_id in range(len(doc_token)):
                    for ckp_token in doc_token[sent_id]:
                        corpus_fd.write(f"{doc_id},{sent_id},{ckp_token}\n")
                
                # write doc_id, file_path pair to txt file
                file_path = path.split("data/")[-1]
                doc2file_fd.write(f"{doc_id},{file_path}\n")
                doc_id += 1

    doc2file_fd.close()
    corpus_fd.close()

    return seen_dir, doc_id




# run the script
directory = os.getcwd() + "/data"
TRAIN_CKP_PATH = "../extracted_data/ckp_data.txt"
DOC2FILE_PATH = "../extracted_data/doc2file.txt"
er = EmbedRank()
gen_corpus()