# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

import sys
sys.path.append('./tacotron')
from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf


def encoder(inputs, is_training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 2d tensor with shape of [N, T_x, E], with dtype of int32. Encoder inputs.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Encoder pre-net
        prenet_out = prenet(inputs, is_training=is_training) # (N, T_x, E/2)

        # Encoder CBHG
        ## Conv1D banks
        enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training) # (N, T_x, K*E/2)

        ## Max pooling
        enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding="same")  # (N, T_x, K*E/2)

        ## Conv1D projections
        enc = conv1d(enc, filters=hp.embed_size//2, size=3, scope="conv1d_1") # (N, T_x, E/2)
        enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

        enc = conv1d(enc, filters=hp.embed_size // 2, size=3, scope="conv1d_2")  # (N, T_x, E/2)
        enc = bn(enc, is_training=is_training, scope="conv1d_2")

        enc += prenet_out # (N, T_x, E/2) # residual connections

        ## Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            enc = highwaynet(enc, num_units=hp.embed_size//2,
                                 scope='highwaynet_{}'.format(i)) # (N, T_x, E/2)

        ## Bidirectional GRU
        memory, _ = gru(enc, num_units=hp.embed_size//2, bidirection=True) # (N, T_x, E)

    return memory

def decoder1(inputs, memory, feed_previous=False, is_training=True, scope="decoder1", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels(*r)]. Shifted log melspectrogram of sound files.
      memory: A 3d tensor with shape of [N, T_x, E].
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns
      Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = prenet(inputs, is_training=is_training, scope="decoder1_prenet")  # (N, T_y/r, E/2)
        gru_cell_1 = tf.contrib.rnn.GRUCell(hp.embed_size, name="decoder1_gru1")
        gru_cell_2 = tf.contrib.rnn.GRUCell(hp.embed_size, name="decoder1_gru2")
        gru_cell_3 = tf.contrib.rnn.GRUCell(hp.embed_size, name="decoder1_gru3")

        def step(previous_step_output, current_input):
            current_input = current_input[0]
            previous_output = previous_step_output[0][:, -hp.n_mels:]
            previous_output = prenet(previous_output, is_training=False, scope="decoder1_prenet", reuse=True) # double check
            previous_context = previous_step_output[1]
            previous_attention_weight = previous_step_output[2]
            previous_state = previous_step_output[3:6]

            if feed_previous or not is_training:
                current_input = previous_output

            decoder_input = tf.concat([current_input, previous_context], axis=-1)

            dec, state1 = gru_cell_1(decoder_input, previous_state[0]) # (N, T_y/r, E)
            context_vector, attention_weight = do_attention(dec, memory, previous_attention_weight, hp.embed_size)

            _dec, state2 = gru_cell_2(dec, previous_state[1]) # (N, T_y/r, E)
            dec = _dec + dec
            _dec, state3 = gru_cell_3(dec, previous_state[2]) # (N, T_y/r, E)
            dec = _dec + dec

            mel_hats = tf.layers.dense(dec, hp.n_mels*hp.r)

            return [mel_hats, context_vector, attention_weight, state1, state2, state3]

        batch_size = tf.shape(inputs)[0]
        init_mel = tf.zeros([batch_size, hp.n_mels*hp.r])

        init_context = tf.zeros([batch_size, memory.get_shape().as_list()[-1]])
        init_attention_weight = tf.zeros(tf.shape(memory)[:2])
        init_attention_weight = tf.concat([tf.ones_like(init_attention_weight[:,:1]), init_attention_weight[:,1:]], axis=1)
        init_state = tf.zeros([batch_size, hp.embed_size])
        init = [init_mel, init_context, init_attention_weight, init_state, init_state, init_state]

        inputs_scan = tf.transpose(inputs, [1,0,2])
        output = tf.scan(step, [inputs_scan], initializer=init)

        mel_hats = tf.transpose(output[0], [1,0,2])
        alignments = tf.transpose(output[2], [1,0,2])

    return mel_hats, alignments

"""
def decoder1(inputs, memory, is_training=True, scope="decoder1", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels(*r)]. Shifted log melspectrogram of sound files.
      memory: A 3d tensor with shape of [N, T_x, E].
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = prenet(inputs, is_training=is_training)  # (N, T_y/r, E/2)

        # Attention RNN
        dec, state = attention_decoder(inputs, memory, num_units=hp.embed_size) # (N, T_y/r, E)

        ## for attention monitoring
        alignments = tf.transpose(state.alignment_history.stack(),[1,2,0])

        # Decoder RNNs
        with tf.variable_scope("decoder1_gru1"):
            _dec, _ = gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru1") # (N, T_y/r, E)
            dec += _dec
        with tf.variable_scope("decoder1_gru2"):
            _dec, _ = gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru2") # (N, T_y/r, E)
            dec += _dec

        # Outputs => (N, T_y/r, n_mels*r)
        mel_hats = tf.layers.dense(dec, hp.n_mels*hp.r)

    return mel_hats, alignments
"""

def decoder2(inputs, is_training=True, scope="decoder2", reuse=None):
    '''Decoder Post-processing net = CBHG
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels*r]. Log magnitude spectrogram of sound files.
        It is recovered to its original shape.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      Predicted linear spectrogram tensor with shape of [N, T_y, 1+n_fft//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Restore shape -> (N, Ty, n_mels)
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

        # Conv1D bank
        dec = conv1d_banks(inputs, K=hp.decoder_num_banks, is_training=is_training) # (N, T_y, E*K/2)

        # Max pooling
        dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding="same") # (N, T_y, E*K/2)

        ## Conv1D projections
        dec = conv1d(dec, filters=hp.embed_size // 2, size=3, scope="conv1d_1")  # (N, T_x, E/2)
        dec = bn(dec, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

        dec = conv1d(dec, filters=hp.n_mels, size=3, scope="conv1d_2")  # (N, T_x, E/2)
        dec = bn(dec, is_training=is_training, scope="conv1d_2")

        # Extra affine transformation for dimensionality sync
        dec = tf.layers.dense(dec, hp.embed_size//2) # (N, T_y, E/2)

        # Highway Nets
        for i in range(4):
            dec = highwaynet(dec, num_units=hp.embed_size//2,
                                 scope='highwaynet_{}'.format(i)) # (N, T_y, E/2)

        # Bidirectional GRU
        dec, _ = gru(dec, hp.embed_size//2, bidirection=True) # (N, T_y, E)

        # Outputs => (N, T_y, 1+n_fft//2)
        outputs = tf.layers.dense(dec, 1+hp.n_fft//2)

    return outputs

def build_ref_encoder(inputs, is_training):
    # inputs = [batch_size, seq_len//r, hp.n_mels*hp.r]
    # outputs = [batch_size, hp.ref_enc_gru_size]

    # Restore shape -> [batch_size, seq_len, hp.n_mels]
    inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

    # Expand dims -> [batch_size, seq_len, hp.n_mels, 1]
    inputs = tf.expand_dims(inputs, axis=-1)

    # Six Conv2D layers follow by bn and relu activation
    # conv2d_result = [batch_size, seq_len//2^6, hp.n_mels//2^6]
    hiddens = [inputs]
    for i in range(len(hp.ref_enc_filters)):
        with tf.variable_scope('conv2d_{}'.format(i+1)):
            tmp_hiddens = conv2d(
                            hiddens[i], filters=hp.ref_enc_filters[i],
                            size=hp.ref_enc_size, strides=hp.ref_enc_strides
                        )
            tmp_hiddens = bn(tmp_hiddens, is_training=is_training, activation_fn=tf.nn.relu)
            hiddens.append(tmp_hiddens)
    conv2d_result = hiddens[-1]
    target_dim = conv2d_result.get_shape().as_list()[2] * conv2d_result.get_shape().as_list()[3]
    shape = tf.shape(conv2d_result)
    conv2d_result = tf.reshape(conv2d_result, [shape[0], shape[1], target_dim])
    conv2d_result.set_shape([None, None, target_dim])

    # Uni-dir GRU, ref_emb = the last state of gru
    # ref_emb = [batch_size, hp.ref_enc_gru_size]
    _, ref_emb = gru(conv2d_result, bidirection=False, num_units=hp.ref_enc_gru_size)

    return ref_emb

def build_STL(inputs):
    # inputs = [batch_size, hp.ref_enc_gru_size]
    # outputs = [batch_size, hp.token_emb_size]

    with tf.variable_scope('GST_emb'):
        GST = tf.get_variable(
                'global_style_tokens',
                [hp.token_num, hp.token_emb_size // hp.num_heads],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5)
              )
        # we found that applying a tanh activation to GSTs
        # before applying attention led to greater token diversity.
        GST = tf.nn.tanh(GST)
    with tf.variable_scope('multihead_attn'):
        style_emb = multi_head_attention(
                    # shape = [batch_size, 1, hp.ref_enc_gru_size]
                    tf.expand_dims(inputs, axis=1),
                    # shape = [batch_size, hp.token_num, hp.token_emb_size//hp.num_heads]
                    tf.tile(tf.expand_dims(GST, axis=0), [tf.shape(inputs)[0],1,1]),
                    num_heads=hp.num_heads,
                    num_units=hp.multihead_attn_num_unit,
                    attention_type=hp.style_att_type
                )
    return style_emb, GST
