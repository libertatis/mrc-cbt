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
