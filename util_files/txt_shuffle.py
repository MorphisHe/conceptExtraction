'''
This file shuffles a txt file by line

Run Guild: python3 txt_shuffle.py [file_path_to_shuffle]
'''
import sys
import random
from smart_open import open

file_path = sys.argv[1]
lines = open(file_path).readlines()
random.shuffle(lines)
open(file_path, 'w').writelines(lines)