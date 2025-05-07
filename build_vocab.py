import json
import re
import unicodedata

import enter


def text_to_norm_words(text):
    """
    This function normalizes the input text using Unicode normalization (NFC) and then extracts words using a regular expression.
    The regex r'\w+' matches sequences of word characters (letters, digits, and underscores).
    The text is converted to lowercase before extraction.
    """
    text = unicodedata.normalize('NFC', text)
    return re.findall(r'\w+', text.lower())


def get_vocabulary(file_names):
    # file_names = ['train.json', 'dev.json', 'test.json']
    # file_names = ['train.json']
    # Load the JSON data using UTF-8 encoding
    print(f"\n- Read files {file_names}")
    words = []
    for file_path in file_names:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"- Type of {file_path}:", type(data))
            # Should show Vietnamese properly
            d = 0

        # Assuming the text data is in a specific field, e.g., 'transcriptions'
        for k, entry in data.items():
            if 'transcript' in entry:
                words.extend(text_to_norm_words(entry['transcript']))
                if d == 0:
                    print("--- One sample:", text_to_norm_words(
                        entry['transcript']))
                    d = 1
            else:
                raise ValueError("No [transcript] in file")

    # Preprocess words (e.g., remove duplicates, convert to lowercase)
    # Update this line to remove punctuation
    unique_words = set(word for word in words)
    # Use unique_words to build your vocal model
    unique_words = sorted(list(unique_words))
    print("- Len of unique words:", len(unique_words))
    return unique_words


class Vocabulary:
    def __init__(self, token_list, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        self.token_to_id = {}
        self.id_to_token = {}
        self.specials = specials
        self._build_specials()
        self._build_vocab(token_list)

    def _build_specials(self):
        for token in self.specials:
            self._add_token(token)

    def _add_token(self, token):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def _build_vocab(self, token_list):
        for token in token_list:
            self._add_token(token)

    def __len__(self):
        return len(self.token_to_id)

    def token2id(self, token):
        return self.token_to_id.get(token, self.token_to_id["<unk>"])

    def id2token(self, idx):
        return self.id_to_token.get(idx, "<unk>")

    def encode(self, tokens):
        return [self.token2id(token) for token in tokens]

    def decode(self, ids):
        return [self.id_to_token.get(i, "<unk>") for i in ids]

    def text_to_ids(self, text):
        words = [self.specials[1]] + \
            text_to_norm_words(text) + [self.specials[2]]
        return self.encode(tokens=words)

    def ids_to_text(self, ids):
        words = self.decode(ids)
        text = ""
        for w in words:
            if w not in self.specials:
                text = text + w + " "
        return text[:-1]


if __name__ == "__main__":
    # vocab = get_vocabulary()
    # vocab_train = get_vocabulary(file_names=['train.json'])
    # print(len(vocab), len(vocab_train))

    my_vocab = Vocabulary(get_vocabulary(enter.FILE_NAMES))
    print("*** Len of vocabulary:", len(my_vocab))

    case = my_vocab.text_to_ids("Tôi yêu ___ Việt    ***    Nam tôi ")
    print(case)
    print(my_vocab.ids_to_text(case))
    print(len(my_vocab))
