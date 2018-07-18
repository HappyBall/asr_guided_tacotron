import sys
sys.path.append('./las')
from hyperparams import Hyperparams as hp
import tensorflow as tf
from network import Listener, Speller
from module import *

def build_las(x, y, char2idx, is_training=False):
    with tf.variable_scope('encoder'):
        encoder_output = Listener(x)
    with tf.variable_scope('decoder'):
        decoder_input = tf.concat((tf.ones_like(y[:, :1])*char2idx['S'], y[:, :-1]), -1)
        decoder_input = embed(decoder_input, len(hp.vocab), hp.las_embed_size, zero_pad=True) # double check
        logits, attention_weight = Speller(decoder_input, encoder_output, is_training=is_training)

    return logits, attention_weight

def las_loss(logits, y, char2idx):
    preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
    istarget = tf.to_float(tf.not_equal(y, char2idx['P']))
    acc = tf.reduce_sum(tf.to_float(tf.equal(preds, y))*istarget)/ (tf.reduce_sum(istarget))

    # Loss
    y_smoothed = label_smoothing(tf.one_hot(y, depth=len(hp.vocab)))
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_smoothed)
    loss = tf.reduce_sum(loss*istarget) / (tf.reduce_sum(istarget))

    return loss, acc, preds

def las_train_op(loss, lr, global_step, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    ## gradient clipping
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    clipped = []
    for grad, var in gvs:
        grad = tf.clip_by_norm(grad, 5.)
        clipped.append((grad, var))
    train_op = optimizer.apply_gradients(clipped, global_step=global_step)

    return train_op

def las_summary(loss, acc, lr, mode="train_las"):
    tf.summary.scalar('{}/loss'.format(mode), loss)
    tf.summary.scalar('{}/acc'.format(mode), acc)
    tf.summary.scalar('{}/lr'.format(mode), lr)



