from smart_open import open
from gensim.models.doc2vec import TaggedDocument

class MyCorpus:
    '''
    This class is a generator object. Each iteration returns a
    TaggedDocument object that represent a document.

    Attributes:
    ---------------
    corpus_path: path for the corpus file

    doc2file_path: path for the file that stores (doc_ID, file_name)
    '''
    def __init__(self, corpus_path, doc2file_path):
        self.corpus_path = corpus_path
        self.doc2file_path = doc2file_path

    def __iter__(self):
        doc2file_gen = self.docID_to_filename()
        for doc in open(self.corpus_path):
            # get next document
            cur_docID, doc_tokens = doc.replace("\n", "").split(",")
            # get the file_name for next document
            doc_ID, file_name = next(doc2file_gen)
            while doc_ID != cur_docID:
                doc_ID, file_name = next(doc2file_gen)
            yield TaggedDocument(doc_tokens.split(" "), [file_name])

    def docID_to_filename(self):
        # get next document's file_name
        for line in open(self.doc2file_path):
            # list contain [doc_id, filename]
            yield line.replace("\n", "").split(",")
