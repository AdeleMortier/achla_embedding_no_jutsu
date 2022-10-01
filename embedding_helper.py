
from collections import defaultdict
import os
import gc
from datetime import datetime

import gensim

if gensim.__version__ == '3.6.0':
    from gensim.models import FastText
else:
    from gensim.models.keyedvectors import load_word2vec_format



def build_dict_from_keyedvectors(model):
    pretrained_embedding = defaultdict()
    for word in list(model.wv.key_to_index.keys()):
        pretrained_embedding[word] = model.wv[word]
    print(f'Embedding entries: {len(pretrained_embedding.keys())}')
    print(f'Embedding dimension: {len(pretrained_embedding[list(pretrained_embedding.keys())[0]])}')
    return pretrained_embedding

def build_dict_from_keyedvectors_gensim_4(model):
    pretrained_embedding = defaultdict()
    for word in list(model.key_to_index.keys()):
        pretrained_embedding[word] = model[word]
    print(f'Embedding entries: {len(pretrained_embedding.keys())}')
    print(f'Embedding dimension: {len(pretrained_embedding[list(pretrained_embedding.keys())[0]])}')
    return pretrained_embedding

def build_dict_from_fasttext_model(model):
    pretrained_embedding = defaultdict()
    for word in list(model.words):
        pretrained_embedding[word] = model[word]
    print(f'Embedding entries: {len(pretrained_embedding.keys())}')
    print(f'Embedding dimension: {len(pretrained_embedding[list(pretrained_embedding.keys())[0]])}')
    return pretrained_embedding

def build_dict_from_old_fasttext_model(model):
    pretrained_embedding = defaultdict()
    for word in list(model.wv.vocab.keys()):
        pretrained_embedding[word] = model.wv[word]
    print(f'Embedding entries: {len(pretrained_embedding.keys())}')
    print(f'Embedding dimension: {len(pretrained_embedding[list(pretrained_embedding.keys())[0]])}')
    return pretrained_embedding

def build_dict_from_vector_file(path_to_vecs, filename):
    """ For processing GloVe .txt output. Might no longer be needed, as the gensim load word2vec
    format function can handle GloVe outputs as well"""
    pretrained_embedding = {}
    if filename not in os.listdir(path_to_vecs):
        print(f'File not found. Upload it, or create it on your local computer using GloVe.')
    with open(path_to_vecs+filename, 'r') as f:
        f = f.read().split('\n')
        f = [l.split(' ') for l in f]
        n_entries = len(f)
        for i, l in enumerate(f):
            w = l[0]
            try:
                v = np.array([float(x) for x in l[1:]])
            except ValueError:
                print(f'Line: {i}')
                print(f'Word: {w}')
                print(f'Vector: {l[1:]}')
            pretrained_embedding[w] = v
            if (i % 20000 == 0):
                print(f'Processed {i} / {n_entries} entries')
    print(f'Embedding entries: {len(pretrained_embedding.keys())}')
    print(f'Embedding dimension: {len(pretrained_embedding[list(pretrained_embedding.keys())[0]])}')
    return pretrained_embedding




def load_pretrained_embedding(model, language):
    if model == 'glove' and language == 'hebrew':
        """GloVe requires local training, as the training dataset is
        too big to be stored in RAM, which is necessarily what happens
        if the library glove_python is being used. Local training builds
        a file called vectors.txt that has to be uploaded to this Colab,
        in order to be parsed and stored within a dictionary"""

        print(f'Loading started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        m = load_word2vec_format('./vectors.txt', no_header=True) # works with gensim 4+
        print(f'Loading ended {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        pretrained_embedding = build_dict_from_keyedvectors_gensim_4(m)


    if model == 'glove' and language == 'english':
        if 'glove.42B.300d.txt' not in os.listdir('.'):
        # We'll have to regenerate the model
            if 'glove.42B.300d.zip' not in os.listdir('.'):
            # We'll have to re-download the model
                URL = "https://nlp.stanford.edu/data/glove.42B.300d.zip"
                response = wget.download(URL, "glove.42B.300d.zip")
            with ZipFile('glove.42B.300d.zip', 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall()
                #!wget https://nlp.stanford.edu/data/glove.42B.300d.zip
                #!unzip glove.42B.300d.zip
        print(f'Loading started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        m = load_word2vec_format('./glove.42B.300d.txt', no_header=True) # works with gensim 4+
        print(f'Loading ended {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        pretrained_embedding = build_dict_from_keyedvectors_gensim_4(m)


    if model == 'word2vec' and language == 'hebrew':
        """We use gensim to train a Word2Vec model on Hebrew Wikidumps"""
        crps.create_wiki_corpus()
        train(inp = "wiki.he.word2vec.text", out_model = "wiki.he.word2vec.model")
        print(f'Loading started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        m = Word2Vec.load("wiki.he.word2vec.model")
        print(f'Loading ended {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        pretrained_embedding = build_dict_from_keyedvectors(m)

    if model == 'word2vec' and language == 'english':
        if 'word2vec_skipgram_CoNLL17.txt' not in os.listdir('.'):
        # We'll have to regenerate the model
            if 'word2vec_skipgram_CoNLL17.zip' not in os.listdir('.'):
            # We'll have to re-download the model
                URL = "http://vectors.nlpl.eu/repository/20/40.zip"
                response = wget.download(URL, 'word2vec_skipgram_CoNLL17.zip')
            with ZipFile('word2vec_skipgram_CoNLL17.zip', 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall()
            os.remove('model.bin')
            os.remove('meta.json')
            os.remove('README')
            os.remove('LIST')
            os.rename('model.txt', 'word2vec_skipgram_CoNLL17.txt')
        print(f'Loading started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        m = load_word2vec_format('./word2vec_skipgram_CoNLL17.txt', no_header=False, unicode_errors='ignore') # works with gensim 4+
        print(f'Loading ended {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        pretrained_embedding = build_dict_from_keyedvectors_gensim_4(m)

    if model == 'fasttext' and language == 'hebrew':
        if 'cc.he.300.bin' not in os.listdir('.'):
        # We'll have to regenerate the model
            if 'cc.he.300.bin.gz' not in os.listdir('.'):
            # We'll have to re-download the model
                URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.he.300.bin.gz"
                response = wget.download(URL, "cc.he.300.bin.gz")

                # Extracting the archive
                with gzip.open('cc.he.300.bin.gz', 'rb') as f_in:
                    with open('cc.he.300.bin', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                #!gzip -d cc.he.300.bin.gz

        # Trick from https://vasnetsov93.medium.com/
        #shrinking-fasttext-embeddings-so-that-it-fits-google-colab-cd59ab75959e
        print(f'Loading started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        m = FastText.load_fasttext_format('cc.he.300.bin')
        print(f'Loading ended {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        # we are not saving trainables here
        m.wv.save('fasttext_gensim.model')
        print(f'Saving ended {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        pretrained_embedding = build_dict_from_old_fasttext_model(m)

    if model == 'fasttext' and language == 'english':
        if 'fasttext_skipgram_wikidump2017.txt' not in os.listdir('.'):
        # We'll have to regenerate the model
            if 'fasttext_skipgram_wikidump2017.zip' not in os.listdir('.'):
            # We'll have to re-download the model
                URL = "http://vectors.nlpl.eu/repository/20/10.zip"
                response = wget.download(URL, "fasttext_skipgram_wikidump2017.zip")
                #!wget http://vectors.nlpl.eu/repository/20/10.zip
            with ZipFile('10.zip', 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall()
            os.remove("model.bin")
            os.remove("meta.json")
            os.remove("README")
            os.rename('model.txt', 'fasttext_skipgram_wikidump2017.txt')
        print(f'Loading started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        m = load_word2vec_format('./fasttext_skipgram_wikidump2017.txt', no_header=False, unicode_errors='ignore') # works with gensim 4+
        print(f'Loading ended {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        pretrained_embedding = build_dict_from_keyedvectors_gensim_4(m)

    del m 
    gc.collect()
    return pretrained_embedding