import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import ujson as json

from model import Model

from util import get_batch_dataset
from util import get_dataset
from util import get_record_parser

# for debug, print numpy array fully.
# np.set_printoptions(threshold=np.inf)


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def train(config):
    with open(config.word_emb_file, 'r') as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)    # word embedding matrix
    with open(config.char_emb_file, 'r') as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)    # char embedding matrix

    # total examples number in valid file
    with open(config.dev_meta, 'r') as fh:
        dev_meta = json.load(fh)

    dev_total = dev_meta['total']
    print('Building model...')
    parser = get_record_parser(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = Model(config, iterator, word_mat, char_mat, graph=g)

        # model = QANet4CBT(config, iterator, word_mat, char_mat, graph=g)


        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        loss_save = 10.0
        patience = 0
        best_acc = 0.

        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(config.log_dir)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())

            if os.path.exists(os.path.join(config.save_dir, 'checkpoint')):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))

            global_step = max(sess.run(model.global_step), 1)

            total_corrects = 0
            total_loss = 0.0

            # Training
            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1

                loss_in_batch, corrects_in_batch, train_op = sess.run(
                    [model.loss, model.correct_prediction, model.train_op],
                    feed_dict={
                        handle: train_handle,
                        model.dropout: config.dropout
                    })

                total_corrects += corrects_in_batch
                total_loss += loss_in_batch * config.batch_size

                if global_step % config.period == 0:
                    acc = total_corrects / (global_step * config.batch_size)
                    loss = total_loss / (global_step * config.batch_size)

                    loss_sum = tf.Summary(value=[
                        tf.Summary.Value(tag='model/loss',
                                         simple_value=loss),
                    ])
                    writer.add_summary(loss_sum, global_step)

                    acc_sum = tf.Summary(value=[
                        tf.Summary.Value(tag='model/acc',
                                         simple_value=acc),
                    ])
                    writer.add_summary(acc_sum, global_step)

                # Validation and save model
                if global_step % config.checkpoint == 0:

                    val_acc, val_loss, v_acc_sum, v_loss_sum = validate(
                        config, model, sess, dev_total, 'dev', handle, dev_handle)

                    writer.add_summary(v_acc_sum, global_step)
                    writer.add_summary(v_loss_sum, global_step)

                    # Early Stopping
                    if  val_acc < best_acc:
                        patience += 1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_acc = max(best_acc, val_acc)
                        # Save Model, keep top 5 best models.
                        filename = os.path.join(
                            config.save_dir, 'model_{}_val-acc_{}.ckpt'.format(global_step, best_acc))
                        saver.save(sess, filename)

                    writer.flush()
                    

def validate(config, model, sess, dev_total, data_type, handle, str_handle):

    v_total_loss = 0.
    v_total_corrects = 0.

    for i in tqdm(range(1, dev_total // config.batch_size + 1)):

        v_loss_in_batch, v_corrects_in_batch = sess.run([model.loss, model.correct_prediction],
                                                   feed_dict={handle: str_handle})

        v_total_loss += v_loss_in_batch
        v_total_corrects += v_corrects_in_batch

    val_acc = v_total_corrects / dev_total
    val_loss = v_total_loss / dev_total

    v_loss_sum = tf.Summary(value=[
        tf.Summary.Value(tag='{}/loss'.format(data_type),
                         simple_value=val_loss),
    ])

    v_acc_sum = tf.Summary(value=[
        tf.Summary.Value(tag='{}/acc'.format(data_type),
                         simple_value=val_acc),
    ])

    return val_acc, val_loss, v_acc_sum, v_loss_sum


def test(config):

    # Load word embedding file
    with open(config.word_emb_file, 'r') as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)

    # Load char embedding file
    with open(config.char_emb_file, 'r') as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)

    with open(config.test_meta, 'r') as fh:
        test_meta = json.load(fh)

    test_total = test_meta['total']

    graph = tf.Graph()
    print('Loading model...')
    with graph.as_default() as g:
        test_batch = get_dataset(config.test_record_file, get_record_parser(
            config, is_test=True), config).make_one_shot_iterator()

        model = Model(config, test_batch, word_mat, char_mat, trainable=False, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))

            # if config.decay < 1.0:
            #     sess.run(model.assign_vars)

            total_loss = 0.0
            total_corrects = 0

            result = {}

            for step in tqdm(range(test_total // config.batch_size + 1)):
                loss_in_batch, corrects_in_batch = sess.run([model.loss, model.correct_prediction])

                total_loss += loss_in_batch
                total_corrects += corrects_in_batch

            loss = total_loss / test_total
            acc = total_corrects / test_total

            result['loss'] = loss
            result['acc'] = acc

            with open(config.answer_file, 'w') as fh:
                json.dump(result, fh)

            print('Loss: {}, Accuracy: {}'.format(loss, acc))

