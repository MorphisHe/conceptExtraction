from tika import parser
import nltk
import unidecode
from collections import defaultdict
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class EmbedRank:
    '''
    Attributes:
    ---------------
    sent_tokenizer: nltk sentence tokenizer
    
    punct_tokenizer: nltk punctuation remover
    
    stop_words: stop word list from nltk
    
    pos_tagger: nltk POS (Part Of Speech) tagger
    
    chucker: nltk regex parser used to chunk POS tagged words into specific phrase

    lemmatizer: nltk tool to lemmatize word tokens using POS tag

    freq_dict: a dictionary to keep the frequency of each word in corpus

    parser: tika parser to extract text from files
    '''
    def __init__(self):
        self.sent_tokenizer = nltk.tokenize.sent_tokenize
        self.punct_tokenizer = nltk.RegexpTokenizer(r"[^\W_]+|[^\W_\s]+")
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.pos_tagger = nltk.pos_tag
        self.chucker = nltk.RegexpParser("""NP:{<JJ>*<NN.*>{0,3}}  # Adjectives (0) plus Nouns (1-3)""")
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.freq_dict = defaultdict(int)
        self.parser = parser

    def extract_information(self, pdf_path):
        '''
        This extracts raw text from the given path parameter

        Parameter:
        ---------------
        pdf_path: path of pdf to extract text from
        
        Parameter:
        ---------------
        Extracted raw text
        '''
        text = ''
        try:
            pdf_parser = self.parser.from_file(pdf_path)
            text = pdf_parser['content']

            # check if we got the text, if text is none, then the pdf is probably a scan
            if text == None:
                # TODO handle scanned pdf
                print("Activating OCR")
                #OCR_headers = {'X-Tika-PDFextractInlineImages': 'true'}
                #pdf_parser = parser.from_file(pdf_path, serverEndpoint="http://localhost:9998/rmeta/text",
                #                              headers=OCR_headers)
                #text = pdf_parser['content']
        except Exception as e:
            print(pdf_path, "\n\n")
        
        return text
        #return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',
        #             '', unidecode.unidecode(text))

    def tokenize(self, text):
        '''
        This method dones the following
            - remove accents from accented chars
            - remove URLs
            - remove emails
            - remove digits
            - expand word contractions
            - splits text into sentence tokens then word tokens
            - normalize each sentence with lower case
            - remove punctuations

        Parameter:
        ---------------
        text: raw text to be tokenized

        Return:
        ---------------
        list of cleanned word tokens
        '''
        # remove accents
        text = unidecode.unidecode(text)

        # remove URLs (both http and www), emails, and digits
        # http\S+: regex for http links
        # www\S+: regex for www links
        # \S*@\S*\s?: regex for emails
        # [0-9]: regex for digits
        text = re.sub(r"http\S+|www\S+|\S*@\S*\s?|[0-9]", "", text)

        # expand contractions
        text = contractions.fix(text)

        # tokenize by sent and remove empty line
        sent_token = self.sent_tokenizer(text)
        sent_token = [sent for sent in sent_token if len(sent)]

        # remove punctuations and lower case all
        for i in range(len(sent_token)):
            sent_token[i] = self.punct_tokenizer.tokenize(sent_token[i].lower())
        
        return sent_token

    def pos_tag(self, lst_word_tokens):
        '''
        This method applies POS tag to each word

        Parameter:
        ---------------
        lst_word_tokens: 2d list where first dimension holds sentence level tokens
                         then each sentence token holds it's word tokens
        Return:
        ---------------
        lst_word_tokens: 2d list with first dimension representing the sentence of text corpus
        and second dimension the tuple of word token with its POS tag. (word, POS_tag)
        '''
        # apply POS tag to each word
        for i in range(len(lst_word_tokens)):
            lst_word_tokens[i] = self.pos_tagger(lst_word_tokens[i])

        # filter out unwanted tags
        wanted_tags = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS']
        lst_word_tokens = [[word_token for word_token in sent_token if word_token[1] in wanted_tags]
                           for sent_token in lst_word_tokens]

        return lst_word_tokens

    def get_wordnet_pos(self, word_POS_tag):
        '''
        Helper method of preprocess, returns pos parameter of nltk lemmatizer

        Parameter:
        ---------------
        word_POS_tag: POS tag for the word token

        Return:
        ---------------
        str representation of nltk wordnet tag type
        '''
        return 'n' if word_POS_tag.startswith("N") else 'a'

    def preprocess(self, lst_word_tokens):
        '''
        This method does the following:
            - remove stop words
            - remove words with <= 2 chars or > 21 chars
            - apply lemmatization on each word token using POS tag
            - parses the tagged wordswith nltk regex parser to construct phrases 
              in "adjective(0+)" plus "noun(1-3)" pattern
            - remove duplicate phrases
            - remove empty sentences

        Parameter:
        ---------------
        lst_word_tokens: 2d list where first dimension holds sentence level tokens
                         then each sentence token holds it's word tokens and POS_tag
        Return:
        ---------------
        lst_word_tokens: 2d list with first dimension representing the sentence of text corpus
        and second dimension the candidate key phrases extracted from the sentence
        [["kp1", "kp2", ...], [], ....., []]
        '''
        # remove stop words and words with <= 2 chars or > 21 chars
        lst_word_tokens = [[word_token for word_token in sent_token
                            if (word_token[0] not in self.stop_words and 
                            (len(word_token[0])>2 and len(word_token[0])<=21))] 
                            for sent_token in lst_word_tokens]

        # filter out empty sentence
        lst_word_tokens = [sent_token for sent_token in lst_word_tokens if len(sent_token)]
        
        # map all NN* tags to NN and all JJ* tags to JJ
        lst_word_tokens = [[(word_token[0], "NN") if "NN" in word_token[1] else (word_token[0], "JJ") 
                            for word_token in sent_token] for sent_token in lst_word_tokens]

        # lemmatization
        lst_word_tokens = [[(self.lemmatizer.lemmatize(word_token[0], 
                            pos=self.get_wordnet_pos(word_token[1])), word_token[1])
                            for word_token in sent_token] 
                            for sent_token in lst_word_tokens]

        # chunk the tagged sentence token using "adj(0+)" plus "noun(1-3)" pattern
        # outputing a tree structure
        lst_word_tokens = [self.chucker.parse(sent_token) for sent_token in lst_word_tokens]

        # reconstruct the tree to phrases
        lst_word_tokens = [[' '.join(leaf[0] for leaf in subtree.leaves())
                            for subtree in sent_token.subtrees()
                            if subtree.label() == "NP"]
                            for sent_token in lst_word_tokens]

        # remove duplicated phrases
        seen_phrase = set()
        for sent_index in range(len(lst_word_tokens)):
            sent_token = lst_word_tokens[sent_index]
            new_sent_token = []
            for phrase_token in sent_token:
                if phrase_token not in seen_phrase:
                    seen_phrase.add(phrase_token)
                    new_sent_token.append(phrase_token)
            lst_word_tokens[sent_index] = new_sent_token
        
        # filter out empty sentence
        lst_word_tokens = [sent_token for sent_token in lst_word_tokens if len(sent_token)]

        return lst_word_tokens