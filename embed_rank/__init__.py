'''
This file downloads 
'''
import nltk
import os

# set-up enviornment for off line tika server
os.environ['TIKA_SERVER_JAR'] = os.getcwd() + "/embed_rank/tika_server/tika-server-1.24.1.jar"

# check nltk data avaliability. Download if not avaliable 
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK Tokenizer\n")
    nltk.download("punkt")

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading NLTK Averaged Perceptron Tagger")
    nltk.download('averaged_perceptron_tagger\n')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK WordNet\n")
    nltk.download('wordnet')


