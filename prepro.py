import tensorflow as tf
import random
from tqdm import tqdm
# import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open
import re
import nltk

# nlp = spacy.blank('en')

# spacy tokenize the word like 'abc-def' into 'abc', '-', 'def'
# and some answer will be like this, so we don't use spacy
# as word tokenizer.
# def word_tokenize(sent):
#     doc = nlp(sent)
#     return [token.text for token in doc]


def default_tokenizer(sentence):
    _DIGIT_RE = re.compile(r"\d+")
    sentence = _DIGIT_RE.sub("0", sentence)
    sentence = " ".join(sentence.split("|"))
    return nltk.word_tokenize(sentence)


def process_file(filename, data_type, word_counter, char_counter):
    print('Generating {} examples ...'.format(data_type))
    examples = []
    total = 0
    with open(filename, 'r') as fh:
        source = json.load(fh)
        for ex in tqdm(source):
            context = ex['doc'].replace(
                "''", '" ').replace("``", '" ')
            context_tokens = context.split(' ')
            context_chars = [list(token) for token in context_tokens]

            for token in context_tokens:
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1

            ques = ex['ques'].replace(
                "''", '" ').replace("``", '" ')
            ques_tokens = ques.split(' ')
            ques_chars = [list(token) for token in ques_tokens]

            for token in ques_tokens:
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1

            ans = ex['ans']

            cans = ex['cans'].split(' ')

            if ex['ans'] not in cans:
                print('ans: {}'.format(ex['ans']))
                print('cans: {}'.format(cans))
                break


            example = {'context_tokens': context_tokens,
                       'context_chars': context_chars,
                       'ques_tokens': ques_tokens,
                       'ques_chars': ques_chars,
                       'ans': ans,
                       'cans': cans}
            examples.append(example)
            total += 1

        random.shuffle(examples)
        print('{} questions in total'.format(len(examples)))
    return examples

def convert_to_features(config, data, word2idx_dict, char2idx_dict):

    example = {}
    context, question = data

    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')

    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = config.test_para_limit
    ques_limit = config.test_ques_limit
    ans_limit = 100
    char_limit = config.char_limit

    def filter_func(example):
        return len(example['context_tokens']) > para_limit or \
               len(example['ques_tokens']) > ques_limit

    if filter_func(example):
        raise ValueError('Context/Questions lengths are over the limit')

    context_idxs = np.zeros(shape=[para_limit], dtype=np.int32)
    context_char_idxs = np.zeros(shape=[para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros(shape=[ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros(shape=[ques_limit, char_limit], dtype=np.int32)
    y1 = np.zeros([para_limit], dtype=np.float32)
    y2 = np.zeros([para_limit], dtype=np.float32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1    # OOV

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1    # OOV

    for i, token in enumerate(example['context_tokens']):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example['ques_tokens']):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example['context_chars']):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example['ques_chars']):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs

def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print('Generating {} embedding ...'.format(data_type))

    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]

    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, 'r', encoding='utf-8') as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = ''.join(array[0: -vec_size])
                vector = list(map(float, array[-vec_size: ]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print('{} / {} tokens have corresponding {} embedding vector'.format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
        print('{} tokens have corresponding embedding vector'.format(len(filtered_elements)))

    NULL = '--NULL--'
    OOV = '--OOV--'
    # leave the first two id for NULL(UNK) and OOV tokens
    token2idx_dict = {token: idx for idx, token in
                      enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]

    idx2emb_dict = {idx: embedding_dict[token] for token, idx in
                    token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict

# convert train/dev/test examples into id features, and save as TFRecord file
# and return total num examples in train/dev/test files ==
# {train/dev/test}_meta['total']
def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):

    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    char_limit = config.char_limit

    num_cans = config.num_cans  # the number of candidates, default=10.

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit or \
               len(example["cans"]) != num_cans

    print("Processing {} examples...".format(data_type))

    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1

        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

        ans = np.zeros([para_limit], dtype=np.int32)
        cans = np.zeros([num_cans, para_limit], dtype=np.int32)
        y_true = np.zeros([num_cans], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        # Mark where is the true answer in context
        for i, token in enumerate(example["context_tokens"]):
            if example["ans"] == token:
                ans[i] = 1
            else:
                ans[i] = 0

        # Mark where is the candidates in context
        for i, can in enumerate(example['cans']):
            for j, token in enumerate(example['context_tokens']):
                if token == can:
                    cans[i, j] = 1
                else:
                    cans[i, j] = 0

        # Mark where is the true answer in candidates
        for i, token in enumerate(example["cans"]):
            #cans[i] = _get_word(token)
            if token == example['ans']:
                y_true[i] = 1
            else:
                y_true[i] = 0

        record = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "context_idxs": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                    "ques_idxs": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                    "context_char_idxs": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                    "ques_char_idxs": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),

                    "ans": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans.tostring()])),
                    "cans": tf.train.Feature(bytes_list=tf.train.BytesList(value=[cans.tostring()])),
                    "y_true": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_true.tostring()]))
                }
            )
        )
        writer.write(record.SerializeToString())

    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta

def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):

    word_counter, char_counter = Counter(), Counter()

    train_examples = process_file(filename=config.train_file,
                                  data_type="train",
                                  word_counter=word_counter,
                                  char_counter=char_counter)

    dev_examples = process_file(filename=config.dev_file,
                                data_type="dev",
                                word_counter= word_counter,
                                char_counter=char_counter)

    test_examples = process_file(filename=config.test_file,
                                 data_type="test",
                                 word_counter=word_counter,
                                 char_counter=char_counter)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word_emb_mat, word2idx_dict = get_embedding(counter=word_counter,
                                                data_type="word",
                                                emb_file=word_emb_file,
                                                size=config.glove_word_size,
                                                vec_size=config.glove_dim)

    char_emb_mat, char2idx_dict = get_embedding(counter=char_counter,
                                                data_type="char",
                                                emb_file=char_emb_file,
                                                size=char_emb_size,
                                                vec_size=char_emb_dim)

    train_meta = build_features(config=config,
                                examples=train_examples,
                                data_type="train",
                                out_file=config.train_record_file,
                                word2idx_dict=word2idx_dict,
                                char2idx_dict=char2idx_dict)

    dev_meta = build_features(config=config,
                              examples=dev_examples,
                              data_type="dev",
                              out_file=config.dev_record_file,
                              word2idx_dict=word2idx_dict,
                              char2idx_dict=char2idx_dict)

    test_meta = build_features(config=config,
                               examples=test_examples,
                               data_type="test",
                               out_file=config.test_record_file,
                               word2idx_dict=word2idx_dict,
                               char2idx_dict=char2idx_dict,
                               is_test=True)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")

    save(config.train_meta, train_meta, message="train meta")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")

    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")

