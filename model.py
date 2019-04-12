import numpy as np
import tensorflow as tf

from attention_sum import sum_probs_batch
from layers import conv
from layers import highway
from layers import initializer
from layers import mask_logits
from layers import optimized_trilinear_for_attention
from layers import regularizer
from layers import residual_block
from layers import total_params
from layers import trilinear

from layers import BiRNNEncoder

# word : 861,376, char: 1,281,536, 1.49

# NE word : 9k steps, valid acc: 77.12, test acc: 72.05, train acc: 69.68
# CN word : 9k steps, valid acc: 71.88, test acc: 69.03, train acc: 64.25

# NE char : 9.5k steps, valid acc: 75.84, test acc: 71.44, train acc:67.40
# CN char : 9.5k steps, valid acc: 70.88, test acc: 69.60, train acc: 62.88

# ASReader Model
class Model(object):

    def __init__(self,
                 config,
                 batch,
                 word_mat=None,
                 char_mat=None,
                 trainable=True,
                 opt=False,
                 demo=False,
                 graph=None):
        self.config = config
        self.demo = demo
        self.graph = graph if graph is not None else tf.Graph()

        with self.graph.as_default():

            self.global_step = tf.get_variable(
                name='global_step', shape=[], dtype=tf.int32,
                initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(input=0.0, shape=[], name='dropout')

            # Model Input
            if self.demo:
                self.c = tf.placeholder(shape=[None, config.test_para_limit],
                                        name='context', dtype=tf.int32)

                self.q = tf.placeholder(shape=[None, config.test_ques_limit],
                                        name='question', dtype=tf.int32)

                self.ch = tf.placeholder(shape=[None, config.test_para_limit, config.char_limit],
                                         name='context_char', dtype=tf.int32)

                self.qh = tf.placeholder(shape=[None, config.test_ques_limit, config.char_limit],
                                         name='question_char', dtype=tf.int32)

                self.ans = tf.placeholder(shape=[None, config.test_para_limit],
                                          name='answer', dtype=tf.int32)

                self.cans = tf.placeholder(shape=[None, config.num_cans, config.test_para_limit],
                                           name='candidates', dtype=tf.int32)

                self.y_true = tf.placeholder(shape=[None, config.num_cans],
                                             name='y_true', dtype=tf.int32)

            else:
                self.c, self.q, self.ch, self.qh, self.ans, self.cans, self.y_true = batch.get_next()

            self.word_mat = tf.get_variable(    # pre-trained word embeddings
                name='word_mat', initializer=tf.constant(word_mat, dtype=tf.float32), trainable=False)
            self.char_mat = tf.get_variable(    # trainable char embeddings
                name='char_mat', initializer=tf.constant(char_mat, dtype=tf.float32))

            self.c_mask = tf.cast(self.c, tf.bool)  # [batch_size, c_maxlen]
            self.q_mask = tf.cast(self.q, tf.bool)  # [batch_size, q_maxlen]
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)  # [batch_size]
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)  # [batch_size]

            if opt:
                N, CL = config.batch_size if not self.demo else 1, config.char_limit
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)
                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                self.ans = tf.slice(self.ans, [0, 0], [N, self.c_maxlen])
                self.cans = tf.slice(self.cans, [0, 0, 0], [N, config.num_cans, self.c_maxlen])    # not needed
                self.y_true = tf.slice(self.y_true, [0, 0], [N, config.num_cans])
            else:
                self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

            self.ch_len = tf.reshape(
                tf.reduce_sum(tf.cast(
                    tf.cast(self.ch, tf.bool), tf.int32), axis=2),
                shape=[-1])
            self.qn_len = tf.reshape(
                tf.reduce_sum(tf.cast(
                    tf.cast(self.qh, tf.bool), tf.int32), axis=2),
                shape=[-1])

            self.forward()
            total_params()

            if trainable:
                self.lr = tf.minimum(config.learning_rate, 0.001 / tf.log(999.0) *
                                     tf.log(tf.cast(self.global_step, tf.float32) + 1))
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(
                    zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else 1, \
                                   self.c_maxlen, \
                                   self.q_maxlen, \
                                   config.char_limit, \
                                   config.hidden, \
                                   config.char_dim, \
                                   config.num_heads

        with tf.variable_scope('Input_Embedding_Layer', regularizer=regularizer):
            # ******************** char embedding *********************
            # [batch_size, seq_len, word_len] -> [batch_size x seq_len, word_len, char_dim]
            ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch),
                                shape=[N * PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh),
                                shape=[N * QL, CL, dc])
            ch_emb = tf.nn.dropout(ch_emb, keep_prob=1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, keep_prob=1.0 - 0.5 * self.dropout)

            # BiDAF style conv-highway encoder, share weights
            # [N * PL/QL, CL, d]
            ch_emb = conv(ch_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name='char_conv', reuse=None)
            qh_emb = conv(qh_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name='char_conv', reuse=True)

            # [N * CL/QL, d], reduce max along CL
            ch_emb = tf.reduce_max(ch_emb, axis=1)
            qh_emb = tf.reduce_max(qh_emb, axis=1)

            # [N, PL/QL, d]
            ch_emb = tf.reshape(ch_emb, shape=[N, PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, shape=[N, QL, ch_emb.shape[-1]])

            # *********************** Word embedding ************************
            # [N, PL/QL, dw]
            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c),
                                  keep_prob=1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q),
                                  keep_prob=1.0 - self.dropout)

            # Concat char embedding and word embedding
            # [N, PL/QL, dw + d]
            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            # share weights
            c_emb = highway(c_emb, size=d, scope='highway', dropout=self.dropout, reuse=None)
            q_emb = highway(q_emb, size=d, scope='highway', dropout=self.dropout, reuse=True)

            print('highway, q_emb.shape: {}'.format(q_emb.shape))
            print('highway, c_emb.shape: {}'.format(c_emb.shape))

        """ *************************************Encoer ****************************************"""
		with tf.variable_scope('Encoder_Layer', regularizer=regularizer):
            # [N, 2d]
            self.q_enc = BiRNNEncoder(hidden_size=d, num_layers=1,
                                      name='q_enc')(q_emb, self.q_len, is_last_states=True)

            # [N, PL, 2d]
            self.c_enc = BiRNNEncoder(hidden_size=d, num_layers=1,
                                      name='d_enc')(c_emb, self.c_len)

            print('self.q_enc shape: {}'.format(self.q_enc.shape))
            print('self.c_enc shape: {}'.format(self.c_enc.shape))


        """*************************************** Start ****************************************"""
		with tf.variable_scope('Output_Layer'):

			with tf.variable_scope('attention'):
				# [N, PL]
				res = tf.matmul(tf.expand_dims(self.q_enc, -1), self.c_enc, adjoint_a=True, adjoint_b=True)

				attn = tf.reshape(res, [-1, self.c_maxlen])
				attn_dist = tf.nn.softmax(mask_logits(attn, self.c_mask))

			# Attention sum
			# y_hat = sum_probs_batch(self.cans, self.c, attn_dist)
			with tf.variable_scope('attention_sum'):
				# [N, 10, PL]
				y_hat = tf.cast(self.cans, tf.float32) * \ 
							tf.tile(tf.expand_dims(attn_dist, axis=1), [1, config.num_cans, 1])
				y_hat = tf.reduce_sum(y_hat, axis=-1)   # [N, 10]

			with tf.variable_scope('loss'):
				# - log loss
				self.loss = -tf.reduce_mean(
					tf.log(tf.reduce_sum(tf.to_float(self.ans) * attn_dist, axis=-1) + tf.constant(0.00001))
				)
			with tf.variable_scope('correct_prediction'): 
				# correct prediction nums
				self.correct_prediction = tf.reduce_sum(
					tf.sign(tf.cast(tf.equal(tf.argmax(y_hat, 1),
											 tf.argmax(tf.cast(self.y_true, tf.float32), 1)), "float")))
				# print('y_true.shape : {}'.format(self.y_true.shape))

            """************************************** End ***************************************"""
            # add l2 normalization to loss
            if config.l2_norm is not None:
                 variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                 l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
                 self.loss += l2_loss
            
			if config.decay is not None:
                 self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
                 ema_op = self.var_ema.apply(tf.trainable_variables())
                 with tf.control_dependencies([ema_op]):
                     self.loss = tf.identity(self.loss)
            
                     self.assign_vars = []
                     for var in tf.global_variables():
                         v = self.var_ema.average(var)
                         if v:
                             self.assign_vars.append(tf.assign(var, v))

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_correct_prediction(self):
        return self.correct_prediction
