import tensorflow as tf
import re
from collections import Counter
import string

# parse TFRecord files and extract features
# for train / dev / test
def get_record_parser(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit   # document max len
        ques_limit = config.test_ques_limit if is_test else config.ques_limit   # question max len
        char_limit = config.char_limit
        num_cans = config.num_cans

        features = tf.parse_single_example(
            example,
            features={
                'context_idxs': tf.FixedLenFeature([], tf.string),
                'ques_idxs': tf.FixedLenFeature([], tf.string),
                'context_char_idxs': tf.FixedLenFeature([], tf.string),
                'ques_char_idxs': tf.FixedLenFeature([], tf.string),
                'ans': tf.FixedLenFeature([], tf.string),
                'cans': tf.FixedLenFeature([], tf.string),
                'y_true': tf.FixedLenFeature([], tf.string)
            }
        )

        context_idxs = tf.reshape(tf.decode_raw(features['context_idxs'], tf.int32),
                                  shape=[para_limit])

        ques_idxs = tf.reshape(tf.decode_raw(features['ques_idxs'], tf.int32),
                               shape=[ques_limit])

        context_char_idxs = tf.reshape(tf.decode_raw(features['context_char_idxs'], tf.int32),
                                       shape=[para_limit, char_limit])

        ques_char_idxs = tf.reshape(tf.decode_raw(features['ques_char_idxs'], tf.int32),
                                    shape=[ques_limit, char_limit])

        ans = tf.reshape(tf.decode_raw(features['ans'], tf.int32),
                         shape=[para_limit])

        cans = tf.reshape(tf.decode_raw(features['cans'], tf.int32),
                          shape=[num_cans, para_limit])

        y_true = tf.reshape(tf.decode_raw(features['y_true'], tf.int32),
                            shape=[num_cans])


        return context_idxs, ques_idxs, \
               context_char_idxs, ques_char_idxs, \
               ans, cans, y_true

    return parse

def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads
    ).shuffle(config.capacity).repeat()

    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, ans, cans):
            c_len = tf.reduce_sum(tf.cast(
                tf.cast(context_idxs, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size
        )).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset

def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads
    ).repeat().batch(config.batch_size)

    return dataset

def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

















