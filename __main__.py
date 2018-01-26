
# !/usr/bin/env python
import utils
from config import Config
from parser import Parser
from dataset import DataSet
from network import Network
import eval_performance as perf

import sys
import time
import pickle
import threading
import numpy as np
import tensorflow as tf
from os import path
from copy import deepcopy
from tabulate import tabulate


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

class DCI(object):
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.arch = Network(config)
        self.change = 0
        self.attn_values = 0

        self.rnn_outputs = self.arch.predict(self.data_placeholder, self.data_placeholder2,
                                             self.keep_prob_in, self.keep_prob_out, self.inp_lengths)
        self.outputs     = self.arch.projection(self.rnn_outputs)
        self.loss        = self.arch.loss(self.outputs, self.label_placeholder, self.wce_placeholder, self.mask)
        self.optimizer   = self.config.solver._optimizer(self.learning_placeholder)
        self.train       = self.arch.training(self.loss, self.optimizer)

        self.saver        = tf.train.Saver()
        self.summary      = tf.summary.merge_all()
        self.step_incr_op = self.arch.global_step.assign(self.arch.global_step + 1)
        self.init         = tf.global_variables_initializer()

    def bootstrap(self, sess, data, update=True):
        for step, (input_batch, input_batch2, seq, label_batch, tot, lengths, mask) in enumerate(
                self.dataset.next_batch(data, batch_size=self.config.batch_size, shuffle=False)):
            feed_dict = self.create_feed_dict(input_batch, input_batch2, label_batch)
            feed_dict[self.keep_prob_in] = 1
            feed_dict[self.keep_prob_out] = 1
            feed_dict[self.inp_lengths] = lengths

            attn_values, pred_labels = sess.run([self.arch.attn_vals, self.arch.label_preds], feed_dict=feed_dict)
            self.dataset.accumulate_label_cache(pred_labels, seq)

            if step%100 == 0:
                print("{}/{}".format(step, tot), end='\r')

        if update:
            self.dataset.update_label_cache()

    def predict_results(self, sess, data, preds=None):
        if preds == None:
            preds = self.dataset.label_cache

        # nodes = np.where(self.dataset.get_nodes(data))[0]
        # labels_pred = preds[nodes]
        # labels_orig = preds[nodes]

        labels_orig, labels_pred = [], []
        for node in np.where(self.dataset.get_nodes(data))[0]:
            labels_orig.append(self.dataset.all_labels[node])
            labels_pred.append(preds[node])

        return perf.evaluate(labels_pred, labels_orig)

    def load_data(self):
        # Get the 'encoded data'
        self.dataset = DataSet(self.config)

        self.config.data_sets._len_vocab = self.dataset.all_features.shape[0]
        self.config.data_sets._len_labels = self.dataset.all_labels.shape[1]
        self.config.data_sets._len_features = self.dataset.all_features.shape[1]
        self.config.data_sets._multi_label = (np.sum(self.dataset.all_labels, axis=1) > 1).any()
        self.config.num_steps = 20#self.dataset.all_walks.shape[1]

        print('--------- Project Path: ' + self.config.codebase_root_path + self.config.project_name)
        print('--------- Total number of nodes: ' + str(self.config.data_sets._len_vocab))
        print('--------- Walk length: ' + str(self.config.num_steps))
        print('--------- Multi-Label: ' + str(self.config.data_sets._multi_label))
        print('--------- Label Length: ' + str(self.config.data_sets._len_labels))
        print('--------- Feature Length: ' + str(self.config.data_sets._len_features))
        print('--------- Train nodes: ' + str(np.sum(self.dataset.train_nodes)))
        print('--------- Val nodes: ' + str(np.sum(self.dataset.val_nodes)))
        print('--------- Test nodes: ' + str(np.sum(self.dataset.test_nodes)))

    def add_placeholders(self):
        self.data_placeholder = tf.placeholder(tf.float32,
                                               #shape=[self.config.num_steps, None, self.config.data_sets._len_features],
                                               shape=[None, self.config.batch_size, self.config.data_sets._len_features],
                                               name='Input')
        self.data_placeholder2 = tf.placeholder(tf.float32,
                                                #shape=[self.config.num_steps, None, self.config.data_sets._len_labels],
                                                shape=[None, self.config.batch_size, self.config.data_sets._len_labels],
                                                name='label_inputs')
        self.label_placeholder = tf.placeholder(tf.float32, shape=[self.config.batch_size, self.config.data_sets._len_labels],
                                                name='Target')
        self.keep_prob_in = tf.placeholder(tf.float32, name='keep_prob_in')
        self.keep_prob_out = tf.placeholder(tf.float32, name='keep_prob_out')
        self.wce_placeholder = tf.placeholder(tf.float32, shape=[self.config.data_sets._len_labels], name='Cross_entropy_weights')
        self.inp_lengths = tf.placeholder(tf.int32, shape=[self.config.batch_size], name='input_lengths')
        self.mask = tf.placeholder(tf.float32, shape=[self.config.batch_size], name='Mask_for_dummy_entries')
        self.learning_placeholder = tf.placeholder(tf.float32,name='learning_Rate')

    def create_feed_dict(self, input_batch, input_batch2, label_batch):
        feed_dict = {
            self.data_placeholder: input_batch,
            self.data_placeholder2: input_batch2,
            self.label_placeholder: label_batch,
            self.learning_placeholder: self.learning_rate
        }
        return feed_dict

    def add_metrics(self, metrics):
        """assign and add summary to a metric tensor"""
        for i, metric in enumerate(self.config.metrics):
            tf.summary.scalar(metric, metrics[i])

    def print_metrics(self, inp):
        for item, val in inp.items():
            print(item, ": ", val)

    def add_summaries(self, sess):
        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer_train = tf.summary.FileWriter(self.config.logs_dir + "train", sess.graph)

    def write_summary(self, sess, summary_writer, metric_values, step, feed_dict):
        summary = self.summary.merged_summary
        # feed_dict[self.loss]=loss
        feed_dict[self.metrics] = metric_values
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

    def run_epoch(self, sess, data, train_op=None, summary_writer=None, verbose=1):
        #Optimize the objective for one entire epoch via mini-batches
        
        if not train_op:
            train_op = tf.no_op()
            keep_prob_in = 1
            keep_prob_out = 1
        else:
            keep_prob_in = self.config.mRNN._keep_prob_in
            keep_prob_out = self.config.mRNN._keep_prob_out

        total_loss, label_loss = [], []
        f1_micro, f1_macro, accuracy, bae = [], [], [], []
        for step, (input_batch, input_batch2, seq, label_batch, tot, lengths, mask) in enumerate(
                self.dataset.next_batch(data, self.config.batch_size, shuffle=True)):
            # print("\n\n\nActualLabelCount: ", np.shape(input_batch), np.shape(input_batch2), np.shape(label_batch), np.shape(seq))
            feed_dict = self.create_feed_dict(input_batch, input_batch2, label_batch)
            feed_dict[self.keep_prob_in] = keep_prob_in
            feed_dict[self.keep_prob_out] = keep_prob_out
            feed_dict[self.wce_placeholder] = self.dataset.wce
            feed_dict[self.mask] = mask
            feed_dict[self.inp_lengths] = lengths

            # Writes loss summary @last step of the epoch
            if (step + 1) < tot:
                _, loss_value, pred_labels = sess.run([train_op, self.loss, self.arch.label_preds],
                                                      feed_dict=feed_dict)
            else:
                _, loss_value, summary, pred_labels = sess.run(
                    [train_op, self.loss, self.summary, self.arch.label_preds], feed_dict=feed_dict)
                if summary_writer != None:
                    summary_writer.add_summary(summary, self.arch.global_step.eval(session=sess))
                    summary_writer.flush()

            total_loss.append(loss_value[0])
            label_loss.append(loss_value[1])

            if verbose and step % verbose == 0:
                metrics = [0] * 10
                if self.config.solver._curr_label_loss:
                    metrics = perf.evaluate(pred_labels, label_batch)
                    f1_micro.append(metrics['micro_f1'])
                    f1_macro.append(metrics['macro_f1'])
                    accuracy.append(metrics['accuracy'])
                    bae.append(metrics['bae'])
                print('%d/%d : label = %0.4f : micro-F1 = %0.3f : accuracy = %0.3f : bae = %0.3f'
                      % (step, tot, np.mean(label_loss), np.mean(f1_micro),np.mean(accuracy), np.mean(bae)), end="\r")
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
        return np.mean(total_loss), np.mean(f1_micro), np.mean(f1_macro), np.mean(accuracy), np.mean(bae)

    def fit(self, sess, epoch, patience, validation_loss):
        # Controls how many time to optimize the function before making next label prediction
        patience_increase = self.config.patience_increase  # wait this much longer when a new best is found
        improvement_threshold = self.config.improvement_threshold  # a relative improvement of this much is considered significant

        for i in range(self.config.max_inner_epochs): #change this
            start_time = time.time()
            average_loss, tr_micro, tr_macro, tr_accuracy, bae = self.run_epoch(sess, data='train', train_op=self.train,
                                                                           summary_writer=self.summary_writer_train)
            duration = time.time() - start_time

            print("Tr_micro: %f :: Tr_macro: %f :: Tr_accuracy: %f :: Tr_bae: %f  :: Time: %f"%(tr_micro, tr_macro, tr_accuracy, bae, duration))
            if (epoch % self.config.val_epochs_freq == 0):
                s = time.time()
                self.dataset.update_cache = {}
                self.bootstrap(sess, data='all', update=False)
                print('Bootstrap time: ', time.time() - s)

                pred_labels = self.dataset.get_update_cache()
                metrics = self.predict_results(sess, data='val', preds=pred_labels)  # evaluate performance for validation set
                val_micro, val_macro, val_accuracy, bae = metrics['micro_f1'], metrics['macro_f1'], metrics['accuracy'], metrics['bae']
                val_loss = bae

                print('Epoch %d: tr_loss = %.2f, val_loss %.2f || tr_micro = %.2f, val_micro = %.2f || tr_acc = %.2f, val_acc = %.2f  (%.3f sec)'
                        %(epoch, average_loss, val_loss, tr_micro, val_micro, tr_accuracy, val_accuracy, duration))

                if val_loss < validation_loss:
                    validation_loss = val_loss
                    self.saver.save(sess, self.config.ckpt_dir + 'last-best')
                    np.save(self.config.ckpt_dir + 'last-best_labels.npy', pred_labels)

                    patience = patience_increase
                    print('best step %d\n' % (epoch))

                else:
                    patience -= 1
                    if patience == 0:
                        break

            epoch +=1

        return epoch, validation_loss

    def fit_outer(self, sess):
        # define parametrs for early stopping early stopping
        max_epochs = self.config.max_outer_epochs
        patience = self.config.patience  # look as this many examples regardless
        done_looping = False
        epoch = 1
        best_step = -1
        flag = self.config.boot_reset
        outer_epoch =1
        self.learning_rate = self.config.solver.learning_rate
        validation_loss = 1e6

        while (epoch <= max_epochs) and (not done_looping):
            # sess.run([self.step_incr_op])
            # self.arch.global_step.eval(session=sess)
            if outer_epoch == 2 and flag: #reset after first bootstrap
                print("------ Graph Reset | Firdt bootstrap done -----\n\n\n")
                sess.run(self.init)  # reset all weights
                flag = False
                validation_loss = 1e6
                #IMP: under assumption that we can always do better by adding pseudo-labels,
                # otherwise val_loss of first prediction needs to be considered as well
            print([v.name for v in tf.trainable_variables()])  # Just to monitor the trainable variables in tf graph

            # Fit the model to predict best possible labels given the current estimates of unlabeled values
            epoch, new_loss = self.fit(sess, epoch, patience, validation_loss)
            outer_epoch +=1
            if new_loss >= validation_loss:
                self.learning_rate = self.learning_rate / 10
                patience = epoch + max(self.config.val_epochs_freq, self.config.patience_increase)
                print('--------- Learning rate dropped to: %f' % (self.learning_rate))

                self.saver.restore(sess, tf.train.latest_checkpoint(self.config.ckpt_dir))
                self.dataset.label_cache = np.load(self.config.ckpt_dir + 'last-best_labels.npy').item()

                if self.learning_rate <= 0.000001:
                    print('Stopping by patience method')
                    done_looping = True

            else:
                self.dataset.update_label_cache()
                print("========== Label updated ============= \n")
                # Get predictions for test nodes
                self.print_metrics(self.predict_results(sess, data='test'))
                validation_loss = new_loss

        # End of Training
        self.saver.restore(sess, tf.train.latest_checkpoint(self.config.ckpt_dir))  # restore the best parameters
        self.dataset.label_cache = np.load(self.config.ckpt_dir + 'last-best_labels.npy').item()

        #self.bootstrap(sess, data='all')  # Get new estimates of unlabeled nodes
        metrics = self.predict_results(sess, data='test')
        self.print_metrics(metrics)  # Get predictions for test nodes
        metrics['val_loss'] = validation_loss

        return metrics


########END OF CLASS MODEL#####################################

def init_Model(config):
    tf.reset_default_graph()
    np.random.seed(1234)
    tf.set_random_seed(1234)
    with tf.variable_scope('RNNLM', reuse=None) as scope:
        model = DCI(config)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    load_ckpt_dir = ''
    print('--------- Training from scratch')
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir, config=tfconfig)
    return model, sess


def train_model(cfg):
    print('############## Training Module ')
    config = deepcopy(cfg)
    model, sess = init_Model(config)
    with sess:
        model.add_summaries(sess)
        metrics = model.fit_outer(sess)
        return metrics



def main():
    args = Parser().get_parser().parse_args()
    print("=====Configurations=====\n", args)
    cfg = Config(args)
    train_percents = args.percents.split('_')
    folds = args.folds.split('_')

    outer_loop_stats = {}
    attention = {}
    results = {}
    nodes = {}

    # Create Main directories
    path_prefixes = [cfg.dataset_name, cfg.folder_suffix, cfg.data_sets.label_type]
    utils.create_directory_tree(path_prefixes)

    for train_percent in train_percents:
        cfg.train_percent = train_percent
        path_prefix = path.join(path.join(*path_prefixes), cfg.train_percent)
        utils.check_n_create(path_prefix)

        attention[train_percent] = {}
        results[train_percent] = {}
        outer_loop_stats[train_percent] = {}
        nodes[train_percent] = {}

        for fold in folds:
            print('Training percent: ', train_percent, ' Fold: ', fold, '---Running')
            cfg.train_fold = fold
            utils.check_n_create(path.join(path_prefix, cfg.train_fold))
            cfg.create_directories(path.join(path_prefix, cfg.train_fold))
            results[train_percent][fold] = train_model(deepcopy(cfg))

            print('Training percent: ', train_percent, ' Fold: ', fold, '---completed')

        utils.remove_directory(path_prefix)
        path_prefixes = [cfg.dataset_name, cfg.folder_suffix, cfg.data_sets.label_type]
        np.save(path.join(*path_prefixes, 'results.npy'), results)


if __name__ == "__main__":
    main()