'''
modified from
https://www.github.com/kyubyong/tacotron
'''
from hyperparams import Hyperparams as hp
import tqdm
from data_load import load_data, load_vocab
import tensorflow as tf
from graph import Graph
import os
import numpy as np

def load_pre_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel = hp.prepro_path + "/mels/{}".format(fname.replace("wav", "npy"))
    mag = hp.prepro_path + "/mags/{}".format(fname.replace("wav", "npy"))
    mel = np.load(mel)
    mel = np.reshape(mel, [-1, hp.n_mels])
    t = mel.shape[0]
    num_paddings = 8 - (t % 8) if t % 8 != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel, np.load(mag)

def get_sent(idx2char, sent_idx):
    #sent_idx: 1-d array
    sent = ''
    for idx in sent_idx:
        if idx2char[idx] == 'E':
            break
        sent += idx2char[idx]
    return sent

def wer(r, h):
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def evaluate():
    # Load graph
    g = Graph(mode="evaluate_las"); print("Graph loaded")

    # Load data
    _, idx2char = load_vocab()
    fpaths, _, texts = load_data(mode="evaluate_las")
    all_mel_spec = [load_pre_spectrograms(fpath)[1] for fpath in fpaths]
    maxlen = max([len(m) for m in all_mel_spec])
    new_mel_spec = np.zeros((len(all_mel_spec), maxlen, hp.n_mels), np.float)
    for i, m_spec in enumerate(all_mel_spec):
        new_mel_spec[i, :len(m_spec), :] = m_spec

    saver_las = tf.train.Saver(var_list=g.las_variable)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver_las.restore(sess, tf.train.latest_checkpoint(hp.las_logdir))
        total_y_hat = np.zeros((len(texts), 100), np.float32)
        batch_idx = list(range(0,len(texts),hp.las_inference_batch_size))
        batch_idx.append(len(texts))
        for i in tqdm.tqdm(range(len(batch_idx)-1)):
            y_hat = total_y_hat[batch_idx[i]:batch_idx[i+1]]
            mel_spec = new_mel_spec[batch_idx[i]:batch_idx[i+1]]
            for j in range(100):
                _y_hat = sess.run(g.preds, {g.mels_las: mel_spec, g.texts_las: y_hat})
                y_hat[:, j] = _y_hat[:, j]
            total_y_hat[batch_idx[i]:batch_idx[i+1]] = y_hat

    all_we = 0
    all_wrd = 0
    opf = open(os.path.join(hp.las_logdir,"Inference_text_seqs.txt"), "w") #inference output

    for i, idx_inf in enumerate(total_y_hat):
        fname = os.path.basename(fpaths[i])

        idx_gt = texts[i]
        str_gt = get_sent(idx2char, idx_gt)

        str_inf = get_sent(idx2char, idx_inf)

        all_we += wer(list(str_inf), list(str_gt))
        all_wrd += len(str_gt)
        #all_we += float(wer(list(str_inf), list(str_gt)))/float(len(str_gt))

        final_str = fname + '\n' + str_gt + '\n' + str_inf + '\n'*2
        opf.write(final_str)
    print('cer: ' + str(all_we/all_wrd))
    opf.write('cer: ' + str(all_we/all_wrd))
    #print('cer: ' + str(all_we/float(len(texts))))
    #opf.write('cer: ' + str(all_we/float(len(texts))))

if __name__ == '__main__':
    evaluate()
    print("Done")
