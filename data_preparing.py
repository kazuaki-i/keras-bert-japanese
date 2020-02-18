import glob
import os
import pandas as pd
import tarfile
from urllib.request import urlretrieve

dir = 'data/livedoor'
if not os.path.exists(dir):
    os.makedirs(dir)

FILEURL = 'https://www.rondhuit.com/download/ldcc-20140209.tar.gz'
FILEPATH = 'data/ldcc-20140209.tar.gz'
EXTRACTDIR = 'data/livedoor'

urlretrieve(FILEURL, FILEPATH)

mode = "r:gz"
tar = tarfile.open(FILEPATH, mode)
tar.extractall(EXTRACTDIR)
tar.close()


def extract_txt(filename):
    with open(filename) as text_file:
        # 0: URL, 1: timestamp
        text = text_file.readlines()[2:]
        text = [sentence.strip() for sentence in text]
        text = list(filter(lambda line: line != '', text))
        return ''.join(text)


categories = [
    name for name
    in os.listdir( os.path.join(EXTRACTDIR, "text") )
    if os.path.isdir( os.path.join(EXTRACTDIR, "text", name) ) ]

categories = sorted(categories)

table = str.maketrans({
    '\n': '',
    '\t': '　',
    '\r': '',
})

all_text = []
all_label = []

for cat in categories:
    files = glob.glob(os.path.join(EXTRACTDIR, "text", cat, "{}*.txt".format(cat)))
    files = sorted(files)
    body = [ extract_txt(elem).translate(table) for elem in files ]
    label = [cat] * len(body)

    all_text.extend(body)
    all_label.extend(label)

df = pd.DataFrame({'text': all_text, 'label': all_label})
df = df.sample(frac=1, random_state=23).reset_index(drop=True)

df[:len(df) // 5].to_csv('test.tsv', sep='\t', index=False)
df[len(df) // 5:len(df)*2 // 5].to_csv('dev.tsv', sep='\t', index=False)
df[len(df)*2 // 5:].to_csv('train.tsv', sep='\t', index=False)
