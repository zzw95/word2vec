from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
from collections import Counter
import random
import numpy as np

class DLProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def load_text():
    if not isfile('data/text8'):
        if not isfile('data/text8.zip'):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc="Text8 Dataset") as pbar:
                urlretrieve('http://mattmahoney.net/dc/text8.zip','data/text8.zip',pbar.hook)
        with zipfile.ZipFile('data/text8.zip') as zf:
            zf.extractall('data/')
    with open('data/text8') as f:
        text = f.read()
    return text

def preprocess(text):
    print("Replace punctuation with tokens ......")
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace(':', ' <COLON> ')

    words = text.split()

    print("Remove all words with  5 or fewer occurences ......")
    word_counter = Counter(words)
    trimmed_words = [word for word in words if word_counter[word] > 5]
    return trimmed_words

def create_lookup_tabels(words):
    word_counter = Counter(words)
    vocab_to_int = {}
    int_to_vocab = {}
    for index,(word, cnt) in enumerate(word_counter.most_common()):
        vocab_to_int[word] = index
        int_to_vocab[index] = word
    return vocab_to_int, int_to_vocab

def subsampling(int_words):
    """
    Words that show up often such as "the", "of", and "for" don't provide much context to the nearby words.
    If we discard some of them, we can remove some of the noise from our data and in return get faster training and better representations.
    This process is called subsampling by Mikolov.
    For each word wi in the training set, we'll discard it with probability given by
    P(wi) = 1 - sqrt(t / f(wi))
    where  tt  is a threshold parameter and  f(wi) is the frequency of word  wi in the total dataset.
    :param int_words: python list representing words by int
    """
    print("Subsampling ......")
    threshold = 1e-5
    int_counter = Counter(int_words)
    total = len(int_words)
    probs = {int_word: 1-np.sqrt(threshold * total / int_counter[int_word]) for int_word in int_counter}
    subsampled_int_words = [int_word for int_word in int_words if probs[int_word] < random.random()]
    return subsampled_int_words

def get_target(words, index, window_size):
    """
    :param words: python list
    :param index: scalar
    :param window_size: scalar
    :return: python list of words in a window around the index
    """
    R = np.random.randint(1, window_size+1)
    start_idx = index-R if (index-R)>0 else 0
    end_idx = index + R
    target_words = set(words[start_idx:index] + words[index+1:end_idx+1])
    return list(target_words)

def get_batches(words, batch_size, window_size):
    n_batches = len(words) // batch_size
    for i in range(n_batches):
        batch = words[i*batch_size:(i+1)*batch_size]
        batch_x, batch_y =[],[]
        for ii in range(len(batch)):
            y = get_target(batch, ii, 5)
            batch_y.extend(y)
            batch_x.extend([batch[ii]]*len(y))
        yield batch_x, batch_y


