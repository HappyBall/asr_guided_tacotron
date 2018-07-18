# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''
from __future__ import print_function, division

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
import librosa
import copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal
import os


def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # num = np.random.randn()
    # if num < .2:
    #     y, sr = librosa.load(fpath, sr=hp.sr)
    # else:
    #     if num < .4:
    #         tempo = 1.1
    #     elif num < .6:
    #         tempo = 1.2
    #     elif num < .8:
    #         tempo = 0.9
    #     else:
    #         tempo = 0.8
    #     cmd = "ffmpeg -i {} -y ar {} -hide_banner -loglevel panic -ac 1 -filter:a atempo={} -vn temp.wav".format(fpath, hp.sr, tempo)
    #     os.system(cmd)
    #     y, sr = librosa.load('temp.wav', sr=hp.sr)

    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)


    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def plot_alignment(alignment, gs, logdir, name):
    """Plots the alignment
    alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    """
    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}_{}k.png'.format(logdir, name, gs//1000), format='png')

def learning_rate_decay(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme from tensor2tensor'''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def load_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0 # for reduction
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel.reshape((-1, hp.n_mels*hp.r)), mag

def _mask_with_zeros(mask, tensor):
    zero_mask = tf.zeros_like(tensor)
    return tf.where(mask, tensor, zero_mask)

def attention_consistency_loss(attention_tacotron, attention_las, text, char2idx):
    '''
    attention_las: [batch x text x (mel+pad)/8]
    attention_tacotron: [batch x (mel)/r x text]
    '''
    las_shape = tf.shape(attention_las)
    attention_las_expand = tf.tile(tf.expand_dims(attention_las, -1), [1,1,1,8])
    attention_las_expand = tf.reshape(attention_las_expand, [las_shape[0], las_shape[1], las_shape[2]*8])/8
    attention_las_expand = attention_las_expand[:,:,:tf.shape(attention_tacotron)[1]*hp.r]

    tacotron_shape = tf.shape(attention_tacotron)
    attention_tacotron_expand = tf.tile(tf.expand_dims(attention_tacotron, -2), [1,1,hp.r,1])
    attention_tacotron_expand = tf.reshape(attention_tacotron_expand, [tacotron_shape[0], tacotron_shape[1]*hp.r, tacotron_shape[2]])/hp.r
    end_to_end_attention = tf.matmul(attention_las_expand, attention_tacotron_expand)

    end_to_end_attention_loss = (end_to_end_attention-tf.eye(tf.shape(attention_tacotron)[-1], batch_shape=[tf.shape(attention_tacotron)[0]]))**2

    one_mask = tf.ones_like(end_to_end_attention)
    length = tf.reduce_sum(tf.to_int32(tf.not_equal(text, char2idx['P'])),-1)
    mask = tf.sequence_mask(length, tf.shape(attention_tacotron)[-1])
    vertical_mask = tf.tile(tf.expand_dims(mask, -1), [1,1,tf.shape(attention_tacotron)[-1]])
    one_mask = _mask_with_zeros(vertical_mask, one_mask)
    end_to_end_attention_loss = _mask_with_zeros(vertical_mask, end_to_end_attention_loss)

    horizontal_mask = tf.tile(tf.expand_dims(mask, -2), [1,tf.shape(attention_tacotron)[-1],1])
    one_mask = _mask_with_zeros(horizontal_mask, one_mask)
    end_to_end_attention_loss = _mask_with_zeros(horizontal_mask, end_to_end_attention_loss)

    return tf.reduce_sum(end_to_end_attention_loss)/tf.reduce_sum(one_mask)
