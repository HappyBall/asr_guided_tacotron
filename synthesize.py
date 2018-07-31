'''
modified from
https://www.github.com/kyubyong/tacotron
'''

from hyperparams import Hyperparams as hp
import tqdm
from data_load import load_data
import tensorflow as tf
from train import Graph
from utils import spectrogram2wav
from scipy.io.wavfile import write
import os
import numpy as np


def synthesize():
    if not os.path.exists(hp.taco_sampledir): os.mkdir(hp.taco_sampledir)

    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    # Load data
    texts = load_data(mode="synthesize")
    _, mel_ref, _ = load_spectrograms(hp.ref_wavfile)
    mel_ref = np.tile(mel_ref, (texts.shape[0], 1, 1))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hp.taco_logdir)); print("Restored!")

        # Feed Forward
        ## mel
        _y_hat = sess.run(g.diff_mels_taco_hat, {g.random_texts_taco: texts, g.mels_taco: mel_ref})
        y_hat = _y_hat # we can plot spectrogram

        mags = sess.run(g.diff_mags_taco_hat, {g.diff_mels_taco_hat: y_hat})
        for i, mag in enumerate(mags):
            print("File {}.wav is being generated ...".format(i+1))
            audio = spectrogram2wav(mag)
            write(os.path.join(hp.taco_sampledir, '{}.wav'.format(i+1)), hp.sr, audio)

if __name__ == '__main__':
    synthesize()
    print("Done")

