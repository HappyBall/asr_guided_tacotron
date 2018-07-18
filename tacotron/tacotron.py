import sys
sys.path.append('./tacotron')
from hyperparams import Hyperparams as hp
import tensorflow as tf
from modules import *
from networks import encoder, decoder1, decoder2, build_ref_encoder
from utils import *


def build_tacotron(x, y, ref, feed_previous=False, is_training=False):
    # x: texts
    # y: mels
    # ref: ref audio
    # Get encoder/decoder inputs
    encoder_inputs = embed(x, len(hp.vocab), hp.embed_size) # (N, T_x, E)
    decoder_inputs = tf.concat((tf.zeros_like(y[:, :1, :]), y[:, :-1, :]), 1) # (N, Ty/r, n_mels*r)
    decoder_inputs = decoder_inputs[:, :, -hp.n_mels:] # feed last frames only (N, Ty/r, n_mels)

    # Networks
    with tf.variable_scope("ref_encoder"):
        ref_emb = build_ref_encoder(ref, is_training)

    with tf.variable_scope("encoder"):
        # memory = [batch_size, seq_len, 2*gru_size=embed_size]
        memory = encoder(encoder_inputs, is_training=is_training) # (N, T_x, E)
        # fusing style embedding into encoder outputs for decoder's attention
        seq_len = tf.shape(x)[1]
        #memory += tf.tile(tf.expand_dims(style_emb, axis=1), [1, seq_len, 1])
        memory = tf.concat([memory, tf.tile(tf.expand_dims(ref_emb, axis=1), [1, seq_len, 1])], axis=-1)
        # double check

    with tf.variable_scope("decoder1"):  #decoder use encoder_final_state
        # Decoder1
        y_hat, alignments = decoder1(decoder_inputs,
                                memory,
                                feed_previous=feed_previous,
                                is_training=is_training) # (N, T_y//r, n_mels*r)
    with tf.variable_scope("decoder2"):
        # Decoder2 or postprocessing
        z_hat = decoder2(y_hat, is_training=is_training) # (N, T_y//r, (1+n_fft//2)*r)

    return y_hat, z_hat, alignments

def tacotron_loss(y, y_hat, z, z_hat):
    loss1 = tf.reduce_mean(tf.abs(y_hat - y))
    loss2 = tf.reduce_mean(tf.abs(z_hat - z))
    loss = loss1 + loss2
    return loss, loss1, loss2

def tacotron_train_op(loss, lr, global_step, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    ## gradient clipping
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    clipped = []
    for grad, var in gvs:
        grad = tf.clip_by_norm(grad, 5.)
        clipped.append((grad, var))
    train_op = optimizer.apply_gradients(clipped, global_step=global_step)

    return train_op

def tacotron_summary(loss1, loss2, loss, lr, y, y_hat, z, z_hat, mode='train_tacotron'):
    audio = tf.py_func(spectrogram2wav, [z_hat[0]], tf.float32)
    tf.summary.scalar('{}/loss1'.format(mode), loss1)
    tf.summary.scalar('{}/loss1'.format(mode), loss2)
    tf.summary.scalar('{}/loss'.format(mode), loss)
    tf.summary.scalar('{}/lr'.format(mode), lr)

    tf.summary.image("{}/mel_gt".format(mode), tf.expand_dims(y, -1), max_outputs=1)
    tf.summary.image("{}/mel_hat".format(mode), tf.expand_dims(y_hat, -1), max_outputs=1)
    tf.summary.image("{}/mag_gt".format(mode), tf.expand_dims(z, -1), max_outputs=1)
    tf.summary.image("{}/mag_hat".format(mode), tf.expand_dims(z_hat, -1), max_outputs=1)

    tf.summary.audio("{}/sample".format(mode), tf.expand_dims(audio, 0), hp.sr)

def diff_tacotron_summary(loss, acc, lr, z_hat, mode="train_tacotron"):
    audio = tf.py_func(spectrogram2wav, [z_hat[0]], tf.float32)
    tf.summary.scalar('{}/loss'.format(mode), loss)
    tf.summary.scalar('{}/acc'.format(mode), acc)
    tf.summary.scalar('{}/lr'.format(mode), lr)

    tf.summary.audio("{}/sample".format(mode), tf.expand_dims(audio, 0), hp.sr)