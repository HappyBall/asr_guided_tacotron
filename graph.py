'''
modified from
https://www.github.com/kyubyong/tacotron
'''
from hyperparams import Hyperparams as hp
import tensorflow as tf
from tacotron.tacotron import build_tacotron, tacotron_loss, tacotron_train_op, tacotron_summary, diff_tacotron_summary
from las.las import build_las, las_loss, las_train_op, las_summary
from data_load import load_vocab, get_batch, get_random_texts
from utils import *

class Graph:
    def __init__(self, mode): #mode: train_las, evaluate_las, train_tacotron, synthesize
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        #tacotron
        #init tacotron variable
        if mode == "train_tacotron":
            self.texts_taco, self.mels_taco, self.mags_taco, _, self.num_batch = get_batch(mode=mode)
            self.random_texts_taco = get_random_texts(mode=mode)
            #self.random_texts_taco = tf.concat([self.texts_taco[1:], self.texts_taco[:1]], axis=0)
            is_training = True
        else: #train las or synthesize # evaluate tacotron still not implemented
            self.texts_taco = tf.placeholder(tf.int32, shape=(None, None))
            self.mels_taco = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))
            self.random_texts_taco  = tf.placeholder(tf.int32, shape=(None, None))
            is_training = False
        #build tacotron graph
        with tf.variable_scope("tacotron") as scope:
            self.mels_taco_hat, self.mags_taco_hat, self.alignments_taco = build_tacotron(self.texts_taco, self.mels_taco, self.mels_taco, is_training=is_training)
            scope.reuse_variables()
            mel_input = tf.zeros([tf.shape(self.texts_taco)[0],50,hp.n_mels*hp.r])
            self.diff_mels_taco_hat, self.diff_mags_taco_hat, self.diff_alignments_taco = build_tacotron(self.random_texts_taco, mel_input, self.mels_taco, feed_previous=True, is_training=is_training)
        with tf.variable_scope("pad") as scope: # transform shape to [batch_size x length x hp.n_mels]
            self.diff_mels_taco_hat_pad = tf.reshape(self.diff_mels_taco_hat, [tf.shape(self.diff_mels_taco_hat)[0], -1, hp.n_mels])
            pad_num = 7-tf.mod(tf.shape(self.diff_mels_taco_hat_pad)[1]-1,8) # pad the sequence length to the multiple of 8
            self.diff_mels_taco_hat_pad = tf.concat([self.diff_mels_taco_hat_pad, tf.zeros([tf.shape(self.diff_mels_taco_hat_pad)[0], pad_num, hp.n_mels])], 1)

        #las
        #init las variable
        if mode=="train_las":
            self.texts_las, self.mels_las, _, _, self.num_batch = get_batch(mode=mode)
            is_training = True
        else:
            self.mels_las = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))
            self.texts_las = tf.placeholder(tf.int32, shape=(None, None))
            is_training = False
        #build las graph
        with tf.variable_scope("las") as scope:
            self.texts_las_logits, self.alignments_las = build_las(self.mels_las, self.texts_las, self.char2idx, is_training=is_training)
            scope.reuse_variables()
            #feed tacotron output to las
            self.diff_texts_las_logits, self.diff_alignments_las = build_las(self.diff_mels_taco_hat_pad, self.random_texts_taco, self.char2idx, is_training=False)

        #define trainable variables
        self.tacotron_variable = [v for v in tf.trainable_variables() if v.name.startswith("tacotron")]
        self.tacotron_to_las_variable = [v for v in tf.trainable_variables() if v.name.startswith("tacotron") and not v.name.startswith("tacotron/decoder2")] # for asr error do not train decoder2
        self.las_variable      = [v for v in tf.trainable_variables() if v.name.startswith("las")]

        if mode == 'train_tacotron':
            #define loss
            self.loss, self.loss1, self.loss2 = tacotron_loss(self.mels_taco, self.mels_taco_hat,
                                                              self.mags_taco, self.mags_taco_hat)
            #define train op
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.lr = learning_rate_decay(hp.taco_lr, global_step=self.global_step)
            self.train_op = tacotron_train_op(self.loss, self.lr, self.global_step, self.tacotron_variable)

            #write summary
            tacotron_summary(self.loss1, self.loss2, self.loss, self.lr,
                            self.mels_taco, self.mels_taco_hat,
                            self.mags_taco, self.mags_taco_hat)

            #when ref signal and input signal are not the same
            self.diff_loss, self.diff_acc, self.diff_preds = las_loss(self.diff_texts_las_logits, self.random_texts_taco, self.char2idx)

            if hp.taco_consis_weight != 0:
                self.diff_loss = self.diff_loss + hp.taco_consis_weight*attention_consistency_loss(self.diff_alignments_taco, self.diff_alignments_las, self.random_texts_taco, self.char2idx)

            #define train op
            self.diff_global_step = tf.Variable(0, name='diff_global_step', trainable=False)
            self.diff_lr = learning_rate_decay(hp.taco_diff_lr, global_step=self.diff_global_step)
            self.diff_train_op = las_train_op(self.diff_loss, self.diff_lr, self.diff_global_step, self.tacotron_to_las_variable)
            diff_tacotron_summary(self.diff_loss, self.diff_acc, self.diff_lr, self.diff_mags_taco_hat)

            self.merged = tf.summary.merge_all()

        if mode == 'train_las' or mode == 'evaluate_las':
            self.loss, self.acc, self.preds = las_loss(self.texts_las_logits, self.texts_las, self.char2idx)

            #define train op
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            #self.lr = tf.placeholder(tf.float32, shape=())
            self.lr = hp.las_lr
            self.train_op = las_train_op(self.loss, self.lr, self.global_step, self.las_variable)
            las_summary(self.loss, self.acc, self.lr)

            self.merged = tf.summary.merge_all()

#TODO
# add length to ref encoder
#get batch should condition on mode
#hyperpatameters
#get ref(shape)  tf.train.batch
#load data mode
