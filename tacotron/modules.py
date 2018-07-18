# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

import sys
sys.path.append('./tacotron')
from hyperparams import Hyperparams as hp
import tensorflow as tf


def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimesionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)


def bn(inputs,
       is_training=True,
       activation_fn=None,
       scope="bn",
       reuse=None):
    '''Applies batch normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but
        the last dimension. Or if type is `ln`, the normalization is over
        the last dimension. Note that this is different from the native
        `tf.contrib.layers.batch_norm`. For this I recommend you change
        a line in ``tensorflow/contrib/layers/python/layers/layer.py`
        as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
      is_training: Whether or not the layer is in training mode.
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims

    # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
    # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
    if inputs_rank in [2, 3, 4]:
        if inputs_rank == 2:
            inputs = tf.expand_dims(inputs, axis=1)
            inputs = tf.expand_dims(inputs, axis=2)
        elif inputs_rank == 3:
            inputs = tf.expand_dims(inputs, axis=1)

        outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               fused=True,
                                               reuse=reuse)
        # restore original shape
        if inputs_rank == 2:
            outputs = tf.squeeze(outputs, axis=[1, 2])
        elif inputs_rank == 3:
            outputs = tf.squeeze(outputs, axis=1)
    else:  # fallback to naive batch norm
        outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               reuse=reuse,
                                               fused=False)
    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs

def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding="SAME",
           use_bias=False,
           activation_fn=None,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    '''
    with tf.variable_scope(scope):
        if padding.lower()=="causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list[-1]

        params = {"inputs":inputs, "filters":filters, "kernel_size":size,
                "dilation_rate":rate, "padding":padding, "activation":activation_fn,
                "use_bias":use_bias, "reuse":reuse}

        outputs = tf.layers.conv1d(**params)
    return outputs

def conv1d_banks(inputs, K=16, is_training=True, scope="conv1d_banks", reuse=None):
    '''Applies a series of conv1d separately.

    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is,
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      is_training: A boolean. This is passed to an argument of `bn`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, hp.embed_size//2, 1) # k=1
        for k in range(2, K+1): # k = 2...K
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, hp.embed_size // 2, k)
                outputs = tf.concat((outputs, output), -1)
        outputs = bn(outputs, is_training=is_training, activation_fn=tf.nn.relu)
    return outputs # (N, T, Hp.embed_size//2*K)

def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
    '''Applies a GRU.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: An int. The number of hidden units.
      bidirection: A boolean. If True, bidirectional results
        are concatenated.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
        otherwise [N, T, num_units].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        cell = tf.contrib.rnn.GRUCell(num_units)
        if bidirection:
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
            return tf.concat(outputs, 2), tf.concat(state, 1)
        else:
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            return outputs, state

def attention_decoder(inputs, memory, num_units=None, scope="attention_decoder", reuse=None):
    '''Applies a GRU to `inputs`, while attending `memory`.
    Args:
      inputs: A 3d tensor with shape of [N, T', C']. Decoder inputs.
      memory: A 3d tensor with shape of [N, T, C]. Outputs of encoder network.
      num_units: An int. Attention size.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with shape of [N, T, num_units].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units,
                                                                   memory)
        decoder_cell = tf.contrib.rnn.GRUCell(num_units)
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                                  attention_mechanism,
                                                                  num_units,
                                                                  alignment_history=True)
        outputs, state = tf.nn.dynamic_rnn(cell_with_attention, inputs, dtype=tf.float32) #( N, T', 16)

    return outputs, state

def prenet(inputs, num_units=None, is_training=True, scope="prenet", reuse=None):
    '''Prenet for Encoder and Decoder1.
    Args:
      inputs: A 2D or 3D tensor.
      num_units: A list of two integers. or None.
      is_training: A python boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3D tensor of shape [N, T, num_units/2].
    '''
    if num_units is None:
        num_units = [hp.embed_size, hp.embed_size//2]

    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name="dense1")
        outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=is_training, name="dropout1")
        outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name="dense2")
        outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=is_training, name="dropout2")
    return outputs # (N, ..., num_units[1])

def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387

    Args:
      inputs: A 3D tensor of shape [N, T, W].
      num_units: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3D tensor of shape [N, T, W].
    '''
    if not num_units:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name="dense2")
        outputs = H*T + inputs*(1.-T)
    return outputs

#for STL
def conv2d(inputs, filters=None, size=[1,1], dilation=[1,1], strides=[1,1],
           padding="SAME", use_bias=True, activation_fn=None):
    if padding.lower()=="causal":
        # pre-padding for causality
        pad_len = (size - 1) * dilation  # padding size
        inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [pad_len, 0]])
        padding = "valid"

    if filters is None:
        filters = inputs.get_shape().as_list[-1]

    params = {"inputs":inputs, "filters":filters,
              "kernel_size":size, "strides":strides,
              "dilation_rate":dilation, "padding":padding,
              "activation":None, "use_bias":use_bias}

    outputs = tf.layers.conv2d(**params)

    return outputs

def multi_head_attention(query, value, num_heads=8, attention_type='mlp_attention',
                         num_units=None, normalize=True):
    ''' ref https://github.com/syang1993/gst-tacotron/blob/master/models/multihead_attention.py '''
    def _split_last_dimension(inputs):
        static_dim = inputs.get_shape().as_list()
        dynamic_dim = tf.shape(inputs)
        assert static_dim[-1] % hp.num_heads == 0
        return tf.reshape(inputs, [dynamic_dim[0], dynamic_dim[1], hp.num_heads, static_dim[-1] // hp.num_heads])
    def _split_heads(q, k, v):
        # qs = [batch_size, num_heads, 1, num_unit//num_heads]
        # ks = [batch_size, num_heads, token_num, num_unit//num_heads]
        # vs = [batch_size, num_heads, token_num, hp.token_emb_size//num_heads]
        qs = tf.transpose(_split_last_dimension(q), [0, 2, 1, 3])
        ks = tf.transpose(_split_last_dimension(k), [0, 2, 1, 3])
        vs = tf.tile(tf.expand_dims(v, axis=1), [1, hp.num_heads, 1, 1])
        return qs, ks, vs

    def _dot_product(qs, ks, vs, num_units):
        # qk = [batch_size, num_heads, 1, token_num]
        qk = tf.matmul(qs, ks, transpose_b=True)
        scale_factor = (num_units // hp.num_heads)**-0.5
        if hp.attn_normalize:
            qk *= scale_factor
        # weights = [batch_size, num_heads, 1, token_num]
        weights = tf.nn.softmax(qk, name="dot_attention_weights")
        # context = [batch_size, num_heads, 1, hp.token_emb_size//num_heads]
        context = tf.matmul(weights, vs)
        return context
    def _mlp_attention(qs, ks, vs):
        num_units = qs.get_shape()[-1].value
        v = tf.get_variable("attention_v", [num_units], dtype=qs.dtype)
        if hp.attn_normalize:
            # Scalar used in weight normalization
            g = tf.get_variable(
                    "attention_g", dtype=qs.dtype,
                    initializer=tf.sqrt((1. / num_units))
                )
            # Bias added prior to the nonlinearity
            b = tf.get_variable(
                    "attention_b", [num_units], dtype=qs.dtype,
                    initializer=tf.zeros_initializer()
                )
            # normed_v = g * v / ||v||
            normed_v = g * v * tf.rsqrt(tf.reduce_sum(tf.square(v)))
            add = tf.reduce_sum(normed_v * tf.tanh(ks + qs + b), [-1], keep_dims=True)
        else:
            add = tf.reduce_sum(v * tf.tanh(ks + qs), [-1], keep_dims=True)

        # weights = [batch_size, num_heads, 1, token_num]
        weights = tf.nn.softmax(tf.transpose(add, [0, 1, 3, 2]), name="mlp_attention_weights")
        # context = [batch_size, num_heads, 1, hp.token_emb_size//num_heads]
        context = tf.matmul(weights, vs)

        return context

    if num_units is None:
        num_units = query.get_shape().as_list()[-1]
    if num_units % hp.num_heads != 0:
        raise ValueError("Multi head attention requires that num_units is a multiple of {}".format(num_heads))

    q = tf.layers.conv1d(query, num_units, 1)
    ### maybe duplicate value num_heads times is enough
    k = tf.layers.conv1d(value, num_units, 1)
    v = value
    qs, ks, vs = _split_heads(q, k, v)
    if attention_type == 'mlp_attention':
        style_emb = _mlp_attention(qs, ks, vs)
    elif attention_type == 'dot_attention':
        style_emb = _dot_product(qs, ks, vs, num_units)
    else:
        raise ValueError('Only mlp_attention and dot_attention are supported')

    # combine each head to one
    ### or pass through a linear?
    style_emb = tf.reshape(style_emb, [tf.shape(query)[0], hp.token_emb_size])

    return style_emb

def do_attention(state, memory, prev_weight, attention_hidden_units, memory_length=None, reuse=None):
    """
    bahdanau attention, aka, original attention
    state: [batch_size x hidden_units]
    memory: [batch_size x T x hidden_units]
    prev_weight: [batch_size x T]
    """
    state_proj = tf.layers.dense(state, attention_hidden_units, use_bias=True)
    memory_proj = tf.layers.dense(memory, attention_hidden_units, use_bias=None)
    #previous_feat = tf.layers.conv1d(inputs=tf.expand_dims(prev_weight,axis=-1), filters=10, kernel_size=50, padding='same')
    #previous_feat = tf.layers.dense(previous_feat, attention_hidden_units, use_bias=None)
    temp = tf.expand_dims(state_proj, axis=1) + memory_proj #+ previous_feat
    temp = tf.tanh(temp)
    score = tf.squeeze(tf.layers.dense(temp, 1, use_bias=None),axis=-1)

    #mask
    if memory_length is not None:
        mask = tf.sequence_mask(memory_length, tf.shape(memory)[1])
        paddings = tf.cast(tf.fill(tf.shape(score), -2**30),tf.float32)
        score = tf.where(mask, score, paddings)

    weight = tf.nn.softmax(score) #[batch x T]
    context_vector = tf.matmul(tf.expand_dims(weight,1),memory)
    context_vector = tf.squeeze(context_vector,axis=1)

    return context_vector, weight
