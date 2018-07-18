# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''
from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import re
import os
import unicodedata

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode):
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode in ("train_tacotron", "train_las", "evaluate_las"):
        # Parse
        fpaths, text_lengths, texts = [], [], []
        if 'train' in mode: #train_tacotron train_las
            print('load training data')
            transcript = os.path.join(hp.data, hp.train_data_name)
        else:
            print('load testing data')
            transcript = os.path.join(hp.data, hp.test_data_name)
        lines = open(transcript, 'r').readlines()
        total_hours = 0
        """
        if mode=="train":
            lines = lines[1:]
        else: # We attack only one sample!
            lines = lines[:1]
        """

        for line in lines:
            #fname, _, text = line.strip().split("|")
            fname = line.split()[0]
            text = " ".join(line.split()[1:])
            fpath = os.path.join(hp.data, "wavs", fname + ".wav")
            fpaths.append(fpath)

            text = text_normalize(text) + "E"  # E: EOS
            text = [char2idx[char] for char in text]
            text_lengths.append(len(text))
            if 'train' in mode:
                texts.append(np.array(text, np.int32).tostring())
            else: # evaluate_las or synthesize
                texts.append(np.array(text, np.int32))

        return fpaths, text_lengths, texts
    else: # mode=synthesize
        # Parse
        lines = open(hp.test_data, 'r').readlines()
        sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
        lengths = [len(sent) for sent in sents]
        maxlen = sorted(lengths, reverse=True)[0]
        texts = np.zeros((len(sents), maxlen), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts

def get_batch(mode):
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data(mode) # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.batch_size

        fpaths = tf.convert_to_tensor(fpaths)
        text_lengths = tf.convert_to_tensor(text_lengths)
        texts = tf.convert_to_tensor(texts)

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        if hp.prepro:
            def _load_spectrograms(fpath):
                if mode == 'train_tacotron':
                    fname = os.path.basename(fpath.decode())
                    mel = hp.prepro_path + "/mels/{}".format(fname.replace("wav", "npy"))
                    mag = hp.prepro_path + "/mags/{}".format(fname.replace("wav", "npy"))
                    return fname, np.load(mel), np.load(mag)
                elif mode == 'train_las':
                    fname = os.path.basename(fpath.decode())
                    mel = hp.prepro_path + "/mels/{}".format(fname.replace("wav", "npy"))
                    mag = hp.prepro_path + "/mags/{}".format(fname.replace("wav", "npy"))
                    mel = np.load(mel)
                    mel = np.reshape(mel, [-1, hp.n_mels])
                    t = mel.shape[0]
                    num_paddings = 8 - (t % 8) if t % 8 != 0 else 0
                    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
                    return fname, mel, np.load(mag)
                else:
                    sys.exit(-1)


            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        if mode == 'train_tacotron':
            mel.set_shape((None, hp.n_mels*hp.r))
        elif mode == 'train_las':
            mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.n_fft//2+1))

        # Batching
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=hp.batch_size,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=16,
                                            capacity=hp.batch_size * 4,
                                            dynamic_pad=True)

    return texts, mels, mags, fnames, num_batch

def get_random_texts(mode):
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data(mode) # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.batch_size

        fpaths = tf.convert_to_tensor(fpaths)
        text_lengths = tf.convert_to_tensor(text_lengths)
        texts = tf.convert_to_tensor(texts)

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        # Add shape information
        text.set_shape((None,))

        # Batching
        texts = tf.train.batch(
                                tensors=[text],
                                batch_size=hp.batch_size,
                                num_threads=16,
                                capacity=hp.batch_size * 4,
                                dynamic_pad=True)

    return texts

"""
def get_ref_batch(mode):
    #Loads training data and put them in queues
    assert (mode == 'train_tacotron')
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data(mode) # list

        # Calc total batch count
        num_batch = len(fpaths) // hp.batch_size

        fpaths = tf.convert_to_tensor(fpaths)
        # Create Queues
        fpath = tf.train.slice_input_producer([fpaths], shuffle=True)

        if hp.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath.decode())
                mel = hp.prepro_path + "/mels/{}".format(fname.replace("wav", "npy"))
                return fname, np.load(mel)

            fname, mel = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32])
        else:
            fname, mel = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        mel.set_shape((None, hp.n_mels*hp.r))

        # Batching
        mels, fnames = tf.train.batch(
                                tensors=[mel, fname],
                                batch_size=hp.batch_size,
                                num_threads=16,
                                capacity=hp.batch_size * 4,
                                dynamic_pad=True)

    return mels
"""