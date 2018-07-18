'''
modified from
https://www.github.com/kyubyong/tacotron
'''
import os
import sys
from hyperparams import Hyperparams as hp
import tensorflow as tf
from tqdm import tqdm
from utils import *
from graph import Graph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("keep_train", "False", "keep training from existed model or not")

if __name__ == '__main__':
    keep_train = FLAGS.keep_train
    g = Graph(mode='train_tacotron'); print("Training Graph loaded")
    if keep_train == "True":
        logfile = open(hp.taco_logfile, "a")
    else:
        if not os.path.exists(hp.taco_logdir):
            os.makedirs(hp.taco_logdir)
        logfile = open(hp.taco_logfile, "w")
    saver = tf.train.Saver(max_to_keep=10)
    saver_las = tf.train.Saver(var_list=g.las_variable) # used to restore only las variable
    init = tf.global_variables_initializer()
    #sv = tf.train.Supervisor(taco_logdir=hp.taco_logdir, save_summaries_secs=60, save_model_secs=0)
    with tf.Session() as sess:
        #while 1:
        writer = tf.summary.FileWriter(hp.taco_logdir, graph = sess.graph)

        if keep_train == "True":
            saver.restore(sess, tf.train.latest_checkpoint(hp.taco_logdir))
            print("Continue training from existed latest model...")
        else:
            sess.run(init)
            print("Initial new training...")

        # load graph from las_logdir
        saver_las.restore(sess, tf.train.latest_checkpoint(hp.las_logdir))
        print('finish loading las varaible')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(1, hp.taco_num_epochs + 1):
            total_loss, total_mel_loss, total_linear_loss, total_diff_loss, total_diff_acc = 0.0, 0.0, 0.0, 0.0, 0.0
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                _, _, gs, diff_gs, l, l_mel, l_linear, l_diff_loss, l_diff_acc = sess.run([g.train_op, g.diff_train_op, g.global_step, g.diff_global_step, g.loss, g.loss1, g.loss2, g.diff_loss, g.diff_acc])

                total_loss += l
                total_mel_loss += l_mel
                total_linear_loss += l_linear
                total_diff_loss += l_diff_loss
                total_diff_acc += l_diff_acc

                # Write checkpoint files
                if gs % 1000 == 0:
                    al = sess.run(g.alignments_taco)
                    plot_alignment(al[0], gs, hp.taco_logdir, name='taco')
                if diff_gs % 1000 == 0:
                    al = sess.run(g.diff_alignments_taco)
                    plot_alignment(al[0], diff_gs, hp.taco_logdir, name='taco_diff')
                    al = sess.run(g.diff_alignments_las)
                    plot_alignment(al[0], diff_gs, hp.taco_logdir, name='taco_diff_las')


            log = "Epoch " + str(epoch) + " average loss:  " + str(total_loss/float(g.num_batch)) + ", average mel loss: " + str(total_mel_loss/float(g.num_batch)) + ", average linear loss: " + str(total_linear_loss/float(g.num_batch)) + "\n"
            log = log + "average diff loss: " + str(total_diff_loss/g.num_batch) + " average diff acc: " + str(total_diff_acc/g.num_batch) + "\n"
            print(log)
            sys.stdout.flush()
            logfile.write(log)

            # Write checkpoint files
            if epoch % 10 == 0:
                #sv.saver.save(sess, hp.taco_logdir + '/model_gs_{}k'.format(gs//1000))
                saver.save(sess, hp.taco_logdir + '/model_epoch_{}.ckpt'.format(epoch))
                result = sess.run(g.merged)
                writer.add_summary(result, epoch)

        coord.request_stop()
        coord.join(threads)

    print("Done")

