from smart_open import open
import time

def docID_to_filename(doc2file_path):
    # get next document's file_name
    for line in open(doc2file_path):
        # list contain [doc_id, filename]
        yield line.replace("\n", "").split(",")


def convert(corpus_path, doc2file_path, output_path):
    corpus_fd = open(corpus_path, "r")
    doc2file_fd = open(doc2file_path, "r")
    out_fd = open(output_path, "w")
    doc2file_gen = docID_to_filename(doc2file_path)

    cur_docID = '0'
    cur_Doc = ''
    counter = 1
    for line in corpus_fd:
        # for visualization only
        if counter%100000==0:
            print("@", end="")
        else:
            print("#", end="")
        counter += 1

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
CORPUS_PATH = "extracted_data/ckp_data.txt"
DOC2FILE_PATH = "extracted_data/doc2file.txt"
OUTPUT_PATH = "extracted_data/train_corpus.txt"

if convert(CORPUS_PATH, DOC2FILE_PATH, OUTPUT_PATH):
    s = time.time()
    print("Convert training data to one document per line done!!!")
    print("Total Time Taken: {}mins".format(round((time.time()-s)/60, 2)))