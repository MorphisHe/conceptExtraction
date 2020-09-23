from smart_open import open
from gensim.models.doc2vec import TaggedDocument

class MyCorpus:
    '''
    This class is a generator object. Each iteration returns a
    TaggedDocument object that represent a document.

    Attributes:
    ---------------
    corpus_path: path for the corpus file
    '''
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def __iter__(self):
        doc2file_gen = self.docID_to_filename()
        for doc in open(self.corpus_path):
            # get next document
            doc_ID, doc_tokens = doc.replace("\n", "").split(",")
            yield TaggedDocument(doc_tokens.split(" "), [doc_ID])
