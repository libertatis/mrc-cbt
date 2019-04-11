import tensorflow as tf
import random
from tqdm import tqdm
# import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open
import nltk
import re

import config

def default_tokenizer(sentence):
    _DIGIT_RE = re.compile(r"\d+")
    sentence = _DIGIT_RE.sub("0", sentence)
    sentence = " ".join(sentence.split("|"))
    return nltk.word_tokenize(sentence)

# nlp = spacy.blank('en')

# spacy tokenize the word like 'abc-def' into 'abc', '-', 'def'
# and some answer will be like this, so we don't use spacy
# as word tokenizer.
# def word_tokenize(sent):
#     doc = nlp(sent)
#     return [token.text for token in doc]


def cbt_text2json(filename):
    print('Generating examples from {}'.format(filename))
    examples = []
    total = 0
    bad = False
    num_no_ans_or_cans = 0
    num_ans_not_in_cans = 0
    num_ans_not_in_doc = 0
    totals = 0
    with open(filename, 'r') as fh:
        counter = 0
        d, q, a, A = [], [], [], []
        for line in tqdm(fh):
            counter += 1
            if counter % 10000 == 0:
                print('Reading line %d in %s' % (counter, filename))
            if counter % 22 == 21:
                q, a, _, A = line.rstrip('\n').split('\t')  # ques, ans, '\t', cans
                q = default_tokenizer(q)[1:]
                a = default_tokenizer(a)[0]
                A = default_tokenizer(A)  # [ans1, ..., ans10]

                if A == '' or A == ' ' or a == '' or a == ' ': # no candidates
                    num_no_ans_or_cans += 1
                    bad = True

                elif a not in A:  # answer not in candidates
                    num_ans_not_in_cans += 1

                    print('ans: {}'.format(a))
                    print('cans : '.format(A))
                    bad = True

                elif a not in d:
                    num_ans_not_in_doc += 1
                    bad = True

            elif counter % 22 == 0:
                if bad:
                    d, q, a, A = [], [], [], []
                    bad = False
                    totals += 1
                    continue
                example = {'doc': ' '.join([w for w in d]),
                           'ques': ' '.join([w for w in q]),
                           'ans': a,
                           'cans': ' '.join([can for can in A])}
                examples.append(example)
                d, q, a, A = [], [], [], []
                total += 1
                totals += 1
            else:
                line_tokens = default_tokenizer(line.rstrip('\n'))[1:]
                d.extend(line_tokens)

        # random.shuffle(examples)
        print('number no answer or candidates : {}'.format(num_no_ans_or_cans))
        print('number answer is not in candidates: {}'.format(num_ans_not_in_cans))
        print('number answer is not in document : {}'.format(num_ans_not_in_doc))
        print('total is {}, and {} questions in total'.format(total, len(examples)))
        print('totals is {}'.format(totals))

    return examples

def to_json(out_file, data):
    with open(out_file, 'wb', encoding='utf-8') as wf:
        json.dump(data, wf)

data_type = 'NE'

cbt_train_file = './datasets/cbt/cbtest_{}_train.txt'.format(data_type)
cbt_valid_file = './datasets/cbt/cbtest_{}_valid_2000ex.txt'.format(data_type)
cbt_test_file = './datasets/cbt/cbtest_{}_test_2500ex.txt'.format(data_type)

cbt_train_out_file = './datasets/cbt/cbtest_{}_train.json'.format(data_type)
cbt_valid_out_file = './datasets/cbt/cbtest_{}_valid_2000ex.json'.format(data_type)
cbt_test_out_file = './datasets/cbt/cbtest_{}_test_2500ex.json'.format(data_type)

to_json(cbt_train_out_file, cbt_text2json(cbt_train_file))
to_json(cbt_valid_out_file, cbt_text2json(cbt_valid_file))
to_json(cbt_test_out_file, cbt_text2json(cbt_test_file))

