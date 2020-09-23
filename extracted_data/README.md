# ckp_data.txt
There are three columns in ckp_data.txt

    - Doc_ID: 1st column
        - int in string format
    - Sent_ID: 2nd column
        - int in string format
        - multiple sent in a document
    - ckp: 3rd column
        - candidate key phrase
        - there are multiple ckp in a sent

# doc2file.txt
There are two columns in doc2file.txt

    - Doc_ID: 1st column
        - int in string format
    - File_Name: 2nd column
        - file name of the original document file

# train_corpus.txt
There are two columns in train_corpus.txt

    - File_Name: 1st column
        - file name of the original document file
    - doc_text: 2nd column
        - concatenated candidate key phrase of current text