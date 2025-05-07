import os

FILE_NAMES = file_names = ['train.json', 'dev.json', 'test.json']
FILE_NAMES = [os.path.join("metadata", x) for x in FILE_NAMES]
NP_FOLDER = "NP"

BS = 4
SORT_TRAIN = True
