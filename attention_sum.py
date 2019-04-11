"""
attention sum operation for cloze-style Reading Comprehension Tasks.
i.e. Children's Book Test (CBT-NE, CBT-CN) Dataset.

refer to :
https://github.com/cairoHy/RC-experiments/blob/master/models/attention_sum_reader.py
"""
import tensorflow as tf

A_LEN = 10

def attention_sum(cans_ids, docoment_ids, attention_dist):
	"""Attention sum.
	
	Args:
		cans_ids:		[batch_size, num_cans]
		docoment_ids:	[batch_size, d_len]
		attention_dist:	[batch_size, d_len]
		
	Returns:
		[batch_size, num_cans]
	"""
    result = sum_probs_batch(cans_ids, 
							 docoment_ids, 
							 attention_dist)
	return result


# attention-sum process
def sum_prob_of_word(word_ix, sentence_ixs, sentence_attention_probs):
    word_ixs_in_sentence = tf.where(tf.equal(sentence_ixs, word_ix))
    return tf.reduce_sum(tf.gather(sentence_attention_probs, word_ixs_in_sentence))


# noinspection PyUnusedLocal
def sum_probs_single_sentence(prev, cur):
    candidate_indices_i, sentence_ixs_t, sentence_attention_probs_t = cur
    result = tf.scan(
        fn=lambda previous, x: sum_prob_of_word(x, sentence_ixs_t, sentence_attention_probs_t),
        elems=[candidate_indices_i],
        initializer=tf.constant(0., dtype="float32"))
    return result


def sum_probs_batch(candidate_indices_bi, sentence_ixs_bt, sentence_attention_probs_bt):
    result = tf.scan(
        fn=sum_probs_single_sentence,
        elems=[candidate_indices_bi, sentence_ixs_bt, sentence_attention_probs_bt],
        initializer=tf.Variable([0] * A_LEN, dtype="float32"))
    return result