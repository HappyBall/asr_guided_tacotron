'''
modified from:
https://www.github.com/kyubyong/tacotron
'''

import os
import sys
import numpy as np
from hyperparams import Hyperparams as hp
import tensorflow as tf
from tqdm import tqdm
from utils import *
from graph import Graph

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("keep_train", "False", "keep training from existed model or not")

if __name__ == '__main__':
    keep_train = FLAGS.keep_train
    g = Graph(mode='train_las'); print("Training Graph loaded")
    if not os.path.isdir(hp.las_logdir):
        os.makedirs(hp.las_logdir)
    logfile = open(hp.las_logfile, "a")
    saver = tf.train.Saver(max_to_keep=10)
    saver_las = tf.train.Saver(var_list=g.las_variable)
    init = tf.global_variables_initializer()
    #sv = tf.train.Supervisor(las_logdir=hp.las_logdir, save_summaries_secs=60, save_model_secs=0)
    with tf.Session() as sess:
        #while 1:
        writer = tf.summary.FileWriter(hp.las_logdir, graph = sess.graph)
        sess.run(init)
        print('finish init model')

        if keep_train == "True":
            saver_las.restore(sess, tf.train.latest_checkpoint(hp.las_logdir))
            print("Continue training from existed latest model...")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        #lr = hp.las_lr
        previous_total_loss = np.inf
        for epoch in range(1, hp.las_num_epochs + 1):
            total_loss = 0.0
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                #_, gs = sess.run([g.train_op, g.global_step])
                #_, gs, l = sess.run([g.train_op, g.global_step, g.loss], feed_dict={g.lr:lr})
                _, gs, l = sess.run([g.train_op, g.global_step, g.loss])

                total_loss += l

                # Write checkpoint files
                if gs % 1000 == 0:
                    #sv.saver.save(sess, hp.las_logdir + '/model_gs_{}k'.format(gs//1000))
                    # plot the first alignment for logging
                    al = sess.run(g.alignments_las)
                    plot_alignment(al[0], gs, hp.las_logdir, name='las')

            #if total_loss > previous_total_loss:
            #    lr = lr*hp.las_lr_decay
            #    print('decay learning rate by:', hp.las_lr_decay, 'now lr:', lr)
            #previous_total_loss = total_loss

            print("Epoch " + str(epoch) + " average loss:  " + str(total_loss/float(g.num_batch)) + "\n")
            sys.stdout.flush()
            logfile.write("Epoch " + str(epoch) + " average loss:  " + str(total_loss/float(g.num_batch)) + "\n")

            # Write checkpoint files
            if epoch % 10 == 0:
                #sv.saver.save(sess, hp.las_logdir + '/model_gs_{}k'.format(gs//1000))
                saver.save(sess, hp.las_logdir + '/model_epoch_{}.ckpt'.format(epoch))
                #result = sess.run(g.merged, feed_dict={g.lr:lr})
                result = sess.run(g.merged)
                writer.add_summary(result, epoch)

        coord.request_stop()
        coord.join(threads)

    print("Done")

# add dropout
# use diff attention


