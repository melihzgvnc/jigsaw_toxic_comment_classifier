import pandas as pd
from pathlib import Path
import string
from collections import Counter

DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent/'data'/'train_split.csv'

class Tokenizer:
        
    def __init__(self, data_path=None):
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.load_text_data()
        unique_words = self.preprocess()
        self.build_vocab(unique_words)
        self.vocab_size = len(self.word2id)
    
    def load_text_data(self):
        data = pd.read_csv(self.data_path)
        self.raw_text = ' '.join(list(data['comment_text']))

    def preprocess(self):
        words = self.raw_text.lower().split()
        translator = str.maketrans('', '', string.punctuation)
        words_cleaned = [word.translate(translator) for word in words]

        words_with_freqs = Counter(words_cleaned)
        min_freq = 3
        # Sorted so it is deterministic. Prevent hash-randomization
        unique_words = sorted([w for w, freq in words_with_freqs.items() if freq > min_freq]) 
        unique_words = ['<NUM>' if word.isdigit() else word for word in unique_words]
        #unique_words = sorted(set(w for w in words_cleaned if w)) # Sorted so it is deterministic. Prevent hash-randomization

        return unique_words

    def build_vocab(self, unique_words):
        self.word2id = {'<PAD>': 0, '<UNK>': 1, '<NUM>': 2}
        self.id2word = {0: '<PAD>', 1: '<UNK>', 2: '<NUM>'}
        for idx, word in enumerate(unique_words, start=2):
            self.word2id[word] = idx
            self.id2word[idx] = word

    def encode(self, seq):
        encoded_seq = []
        for word in seq:
            encoded_seq.append(self.word2id.get(word, self.word2id['<UNK>']))
        
        return encoded_seq

    def decode(self, seq):
        decoded_seq = []
        for word_id in seq:
            decoded_seq.append(self.id2word.get(word_id, '<UNK>'))

        return decoded_seq

tokenizer = Tokenizer()