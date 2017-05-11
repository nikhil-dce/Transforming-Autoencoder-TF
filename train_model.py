import tensorflow as tf
import numpy as np
import sys
import os
import time

from trans_ae import TransformingAutoencoder

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/media/nikhil/hdv/event_summary', """Directory where we write logs and checkpoints""")
tf.app.flags.DEFINE_integer('num_epochs', 20, "Number of epochs to train")
tf.app.flags.DEFINE_integer('num_gpus', 1, "Number of gpus to use")
tf.app.flags.DEFINE_integer('batch_size', 20, "Batch size")
tf.app.flags.DEFINE_integer('save_checkpoint_every', 2, "Save prediction after save_checkpoint_every epochs")
tf.app.flags.DEFINE_integer('save_pred_every', 1, "Save prediction after save_pred_every epochs")

TOWER_NAME = 'tower'
LEARNING_RATE_ADAM = 1e-4
MOVING_AVERAGE_DECAY = 0.9999

class Model_Train:

    def __init__(self, X_trans, trans, X_original, num_capsules, r_dimen, g_dimen, in_dimen):

        self.g_dimen = g_dimen
        self.r_dimen = r_dimen
        self.num_capsules = num_capsules
        self.in_dimen = in_dimen

        self.items = len(X_original)
        self.steps_per_epoch = self.items / FLAGS.batch_size

        self.X_trans = X_trans
        self.trans = trans
        self.X_original = X_original

        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

        print ("TRAIN Directory is %s" % (FLAGS.train_dir))

    def batch_for_step(self, step):
        return (self.X_trans[step*FLAGS.batch_size:(step+1)*FLAGS.batch_size], self.trans[step*FLAGS.batch_size:(step+1)*FLAGS.batch_size], self.X_original[step*FLAGS.batch_size:(step+1)*FLAGS.batch_size])
    
    def train(self):

        with tf.Graph().as_default(), tf.device('/cpu:0'):

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            opt = tf.train.AdamOptimizer(LEARNING_RATE_ADAM)

            tower_grads = []
            encoder = TransformingAutoencoder(self.in_dimen, self.r_dimen, self.g_dimen, self.num_capsules, FLAGS.batch_size)

            # Input placeholders for each step
            X_batch_in = tf.placeholder(tf.float32, shape=[None, 784])
            X_batch_out = tf.placeholder(tf.float32, shape=[None, 784])
            extra_in = tf.placeholder(tf.float32, shape=[None, 2])

            # Only 1 GPU currently
            with tf.device('/gpu:0'):
                with tf.name_scope('%s_%d' % (TOWER_NAME, 0)) as scope:
                    X_batch_pred = encoder.forward_pass(X_batch_in, extra_in)
                    batch_loss = encoder.loss(X_batch_pred, X_batch_out)
                    grads = opt.compute_gradients(batch_loss)

                    tf.summary.scalar('loss', batch_loss)
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    # TODO 
                    # Apply gradients in cpu when multiple gpu

                    for grad, var in grads:
                        if grad is not None:
                            summaries.append(tf.summary.histogram(var.op.name + '\gradients', grad))

                    with tf.name_scope('gradients_apply'):
                        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

                    # Using exponential moving average => Check if this works
                    with tf.name_scope('exp_moving_average'):
                        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
                        variable_average_op = variable_averages.apply(tf.trainable_variables())


            train_op = tf.group(apply_gradient_op, variable_average_op)
            summary_op = tf.summary.merge(summaries)
            
            saver = tf.train.Saver(tf.global_variables())
            init = tf.global_variables_initializer()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement = True
            sess = tf.Session(config=config)

            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value

                total_parameters += variable_parameters

            print 'Total Training Parameters: %d' % (total_parameters)

            sess.run(init)
            print ('Variables Initialized')

            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            meta_graph_def = tf.train.export_meta_graph(filename=FLAGS.train_dir+'/my-model.meta')

            print ('GRAPH is Saved!!')
                        
            for epoch in range(FLAGS.num_epochs):
                start_time = time.time()
                epoch_loss = []
                saveSummary = False
                if epoch % FLAGS.save_pred_every == 0:
                    saveSummary = True
                    
                for step in range(self.steps_per_epoch):
                    
                    x_batch, trans_batch, x_orig_batch = self.batch_for_step(step)
                    feed_dict = {X_batch_in:x_orig_batch, extra_in:trans_batch, X_batch_out:x_batch} 

                    if saveSummary:
                        step_loss, _, summary = sess.run([batch_loss, train_op, summary_op], feed_dict=feed_dict)
                        summary_writer.add_summary(summary, epoch*self.steps_per_epoch + step)
                    else:
                        step_loss, _ = sess.run([batch_loss, train_op], feed_dict=feed_dict)
                    epoch_loss.append(step_loss)
                    
                epoch_loss = sum(epoch_loss)
                duration_time = time.time() - start_time
                print ('Epoch {:d} with loss {:.3f}, ({:.3f} sec/step)'.format(epoch+1, epoch_loss, duration_time))

                # Save model checkpoint
                if epoch % FLAGS.save_checkpoint_every == 0:
                    print 'Saving model checkpoint'
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=epoch)
                
            print "Training Complete"
            sess.close()
            sys.stdout.flush()
        
        
        
