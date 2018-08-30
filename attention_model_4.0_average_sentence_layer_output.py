from __future__ import print_function, division

import tensorflow as tf
import pandas
import time
from data_utils import *
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Q_A pair: Q_A+_A+_A-_A- one question with four answers
NUM_ANSWER_FOR_PER_QUESTION = 2
# threshold t for loss function: func = t - (cos_1 - cos_2)
THRESHOLD = 1.0  # -=-=-=-=--THRESHOLD=1.5; func = t - (cos_1 - parameter*cos_2); per epoch with different A-

tf.flags.DEFINE_string("log_dir", "./attention_model_4.0_average_rnn_output/log/", "log_dir")
tf.flags.DEFINE_string("log_dir_emb", "./attention_model_4.0_average_rnn_output/log/word_embedding/",
                       "visualize_embeddings_log_dir")
tf.flags.DEFINE_string("checkpoint_dir", "./attention_model_4.0_average_rnn_output/checkpoint/", "checkpoint_dir")
tf.flags.DEFINE_string("train_data_dir", "./data/", "train_data_dir")
tf.flags.DEFINE_string("output_dir", "./attention_model_4.0_average_rnn_output/output/", "output_dir")
tf.flags.DEFINE_integer("word_embedding_size", 500, "word_embedding_size")
tf.flags.DEFINE_integer("sentence_embedding_size", 500, "sentence_embedding_size")
tf.flags.DEFINE_integer("batch_size", 64, "batch_size//2 = num_training_question, "
                                          "and NUM_ANSWER_FOR_PER_QUESTION*batch_size=num_answer")
tf.flags.DEFINE_float("embedding_keep_prob", 0.5, "keep_prob for emb_input")
tf.flags.DEFINE_float("word_gru_keep_prob", 0.5, "keep_prob for word_GRU network output")
tf.flags.DEFINE_float("sentence_gru_keep_prob", 1.0, "keep_prob for sentence_GRU network output")
tf.flags.DEFINE_string("optimizer", "tf.train.MomentumOptimizer(learning_rate=1.0, momentum=0.9)",
                       "optimizer for training, Default SGD+Momentum")
tf.flags.DEFINE_float("learning_rate", 0.0027, "learning_rate for optimizer. Default 0.03")
tf.flags.DEFINE_float("decrease_rate", 0.3, "decrease_rate for learning_rate. Default 0.3")
tf.flags.DEFINE_integer("epoch_size", 20, "epoch_size. Default 10")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "max_grad_norm for clip_gradient. Default 40.0")
tf.flags.DEFINE_integer("num_layer", 1, "num_layer of sentence representation GRU. Default 1")
tf.flags.DEFINE_boolean("share_emb_and_softmax", True, "if emb_matrix and softmax_layer share the same variables.")
tf.flags.DEFINE_integer("evaluation_interval", 1, "evaluation_interval for save model")
FLAGS = tf.flags.FLAGS


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    """
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


class AttentionModel(object):
    def __init__(self, question_vocab_size, answer_vocab_size,
                 batch_size, word_embedding_size, sentence_embedding_size, num_layer,
                 max_grad_norm=5.0, initializer=tf.random_normal_initializer(stddev=0.1),
                 share_emb_and_softmax=True,
                 name="attention_model"):
        """
        Creates an End-To-End Q-A matching attention_model
        :param question_vocab_size: The size of the question-vocabulary (should include the nil word)
        :param answer_vocab_size: The size of the answer-vocabulary
        :param batch_size: batch_size when training
        :param word_embedding_size: The size of the word embedding.
        :param sentence_embedding_size: The size of the sentence embedding.
        :param num_layer: the num_layer of sentence representation GRU
        :param max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`
        :param initializer: initializer for variables
        :param share_emb_and_softmax: if word_emb_matrix and softmax_weight share the same variables
        :param name: name for this model
        """

        # hyperparameter
        self._question_vocab_size = question_vocab_size
        self._answer_vocab_size = answer_vocab_size
        self._batch_size = batch_size
        self._word_embedding_size = word_embedding_size
        self._sentence_embedding_size = sentence_embedding_size
        self._max_grad_norm = max_grad_norm
        self._initializer = initializer
        self._share_emb_and_softmax = share_emb_and_softmax
        self._name = name

        # input
        # question_length: the real sequence length of each question in a batch
        # answer_length: real sequence length of each answer in a batch
        self.question = tf.placeholder(tf.int32, [NUM_ANSWER_FOR_PER_QUESTION * self._batch_size, None], "question")
        self.answer = tf.placeholder(tf.int32, [NUM_ANSWER_FOR_PER_QUESTION * self._batch_size, None], "answer")
        self.question_length = tf.placeholder(tf.int32, [NUM_ANSWER_FOR_PER_QUESTION * self._batch_size],
                                              "question_length")
        self.answer_length = tf.placeholder(tf.int32, [NUM_ANSWER_FOR_PER_QUESTION * self._batch_size], "answer_length")
        self.label = tf.placeholder(tf.int32, [self._batch_size, NUM_ANSWER_FOR_PER_QUESTION], "label")
        self.embedding_keep_prob = tf.placeholder(tf.float32, [], "embedding_keep_prob")
        self.word_gru_keep_prob = tf.placeholder(tf.float32, [], "word_gru_keep_prob")
        self.sentence_gru_keep_prob = tf.placeholder(tf.float32, [], "sentence_gru_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, [], "learning_rate")

        # define word_level_embedding_matrix for question and answer
        # self._question_word_emb_matrix = tf.get_variable("question_word_emb_matrix",
        #                                                  [self._question_vocab_size, self._word_embedding_size],
        #                                                  tf.float32, initializer=self._initializer)
        # self._answer_word_emb_matrix = tf.get_variable("answer_word_emb_matrix",
        #                                                [self._answer_vocab_size, self._word_embedding_size],
        #                                                tf.float32, initializer=self._initializer)
        # use pre_train_vec (word2vec, 500d)
        emb_q = build_pre_train_emb_matrix(FLAGS.train_data_dir + "Question_vec_500.txt",
                                           FLAGS.train_data_dir + "filtered_top_30000_Question_word_dict.csv")
        emb_a = build_pre_train_emb_matrix(FLAGS.train_data_dir + "Answer_vec_500.txt",
                                           FLAGS.train_data_dir + "filtered_top_50000_Answer_word_dict.csv")
        self._question_word_emb_matrix = tf.get_variable("question_word_emb_matrix", shape=emb_q.shape,
                                                         initializer=tf.constant_initializer(emb_q), trainable=True)
        self._answer_word_emb_matrix = tf.get_variable("answer_word_emb_matrix", shape=emb_a.shape,
                                                       initializer=tf.constant_initializer(emb_a), trainable=True)

        # # define bi-GRU
        # self._question_word_cell_fw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self._word_embedding_size),
        #                                                             output_keep_prob=self.word_gru_keep_prob)
        # self._question_word_cell_bw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self._word_embedding_size),
        #                                                             output_keep_prob=self.word_gru_keep_prob)
        # self._answer_word_cell_fw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self._word_embedding_size),
        #                                                           output_keep_prob=self.word_gru_keep_prob)
        # self._answer_word_cell_bw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self._word_embedding_size),
        #                                                           output_keep_prob=self.word_gru_keep_prob)

        # define sentence representation GRU
        self._num_layer = num_layer
        self._question_sentence_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.GRUCell(self._sentence_embedding_size),
            output_keep_prob=self.sentence_gru_keep_prob) for _ in
            range(self._num_layer)])
        self._answer_sentence_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.GRUCell(self._sentence_embedding_size),
            output_keep_prob=self.sentence_gru_keep_prob) for _ in
            range(self._num_layer)])

        # # softmax_layer
        # if (self._word_embedding_size == self._sentence_embedding_size) and self._share_emb_and_softmax:
        #     self._question_softmax_weight = tf.transpose(self._question_word_emb_matrix)
        #     self._question_softmax_bias = tf.get_variable("question_softmax_bias", [self._question_vocab_size])
        #     self._answer_softmax_weight = tf.transpose(self._answer_word_emb_matrix)
        #     self._answer_softmax_bias = tf.get_variable("answer_softmax_bias", [self._answer_vocab_size])
        # else:
        #     self._share_emb_and_softmax = False

        # self._opt = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-8)
        self._opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)

    def forward(self):
        with tf.variable_scope("attention_model"):
            print('self.question ', self.question)
            print('self.question_length ', self.question_length)
            print('self.answer ', self.answer)
            print('self.answer_length ', self.answer_length)
            # word_embedding and dropout for input
            with tf.device('/cpu:0'):
                question_word_emb = tf.nn.embedding_lookup(self._question_word_emb_matrix, self.question)
                answer_word_emb = tf.nn.embedding_lookup(self._answer_word_emb_matrix, self.answer)

            question_word_emb = tf.nn.dropout(question_word_emb, self.embedding_keep_prob)
            answer_word_emb = tf.nn.dropout(answer_word_emb, self.embedding_keep_prob)

            # # word representation: build word_gru network for question and answer
            # with tf.variable_scope("question_word_gru"):
            #     # question_word_gru_outputs shape[NUM_ANSWER_FOR_PER_QUE*batch_size, max_time, word_embedding_size*2]
            #     question_word_gru_outputs, question_word_gru_state = tf.nn.bidirectional_dynamic_rnn(
            #         self._question_word_cell_fw, self._question_word_cell_bw, question_word_emb, self.question_length,
            #         dtype=tf.float32)
            #     question_word_gru_outputs = tf.concat([question_word_gru_outputs[0], question_word_gru_outputs[1]],
            #                                           -1)
            # with tf.variable_scope("answer_word_gru"):
            #     # answer_word_gru_outputs shape[NUM_ANSWER_FOR_PER_QUESTION*batch_size, max_time, word_embedding_size*2]
            #     answer_word_gru_outputs, answer_word_gru_state = tf.nn.bidirectional_dynamic_rnn(
            #         self._answer_word_cell_fw, self._answer_word_cell_bw, answer_word_emb, self.answer_length,
            #         dtype=tf.float32)
            #     answer_word_gru_outputs = tf.concat([answer_word_gru_outputs[0], answer_word_gru_outputs[1]], -1)

            question_mask = tf.sequence_mask(lengths=self.question_length, maxlen=tf.reduce_max(self.question_length),
                                             dtype=tf.float32, name="question_mask")
            print("--question_mask-- ", question_mask)
            answer_mask = tf.sequence_mask(lengths=self.answer_length, maxlen=tf.reduce_max(self.answer_length),
                                           dtype=tf.float32, name="answer_mask")
            print("--answer_mask-- ", answer_mask)  # ----------
            # sentence representation: build sentence_gru network for question and answer
            with tf.variable_scope("question_sentence_gru"):
                # question_sentence_gru_state shape=[num_layer, true_batch_size, sentence_embedding_size]
                question_sentence_gru_outputs, question_sentence_gru_state = tf.nn.dynamic_rnn(
                    self._question_sentence_cell, question_word_emb, self.question_length,
                    dtype=tf.float32)
                question_sentence_output = question_sentence_gru_outputs
            with tf.variable_scope("answer_sentence_gru_attention"):
                answer_sentence_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self._sentence_embedding_size, question_sentence_gru_outputs,
                    memory_sequence_length=self.question_length)
                answer_sentence_attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                    self._answer_sentence_cell, answer_sentence_attention_mechanism,
                    attention_layer_size=self._sentence_embedding_size, alignment_history=True)
                zero_state = answer_sentence_attention_cell.zero_state(
                    batch_size=NUM_ANSWER_FOR_PER_QUESTION * self._batch_size, dtype=tf.float32)
                answer_sentence_train_helper = tf.contrib.seq2seq.TrainingHelper(answer_word_emb,
                                                                                 self.answer_length, time_major=False)
                answer_sentence_decoder = tf.contrib.seq2seq.BasicDecoder(answer_sentence_attention_cell,
                                                                          answer_sentence_train_helper, zero_state)
                answer_sentence_gru_outputs, answer_sentence_gru_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    answer_sentence_decoder, output_time_major=False, impute_finished=True)
                answer_sentence_output = answer_sentence_gru_outputs.rnn_output
                # print("answer_sentence_gru_outputs ", answer_sentence_gru_outputs)
                # print('answer_sentence_gru_state ', answer_sentence_gru_state)
                # attention_matrices = answer_sentence_gru_state.alignment_history.stack(name="attention_matrix")
                # print("attention_matrices ", attention_matrices)

            # choose final_state of the last layer as sentence representation: Context_q and Context_a
            # calculate similarity between Context_q and Context_a
            # get reduce_mean among a batch. Add to summary
            context_q = tf.reduce_sum(question_sentence_output * tf.expand_dims(question_mask, axis=2),
                                      axis=1) / tf.reduce_sum(question_mask, axis=1, keep_dims=True)
            print("--context_q-- ", context_q)
            context_a = tf.reduce_sum(answer_sentence_output * tf.expand_dims(answer_mask, axis=2),
                                      axis=1) / tf.reduce_sum(answer_mask, axis=1, keep_dims=True)
            print("--context_a-- ", context_a)
            norm_q = tf.norm(context_q, axis=1)
            context_q = context_q / tf.expand_dims(norm_q, axis=1)
            print("--after normal context_q-- ", context_q)
            norm_a = tf.norm(context_a, axis=1)
            context_a = context_a / tf.expand_dims(norm_a, axis=1)
            print("--after normal context_a-- ", context_a)
            # print('context_q ', context_q, ' context_a ', context_a)
            sentence_representation_q = tf.reshape(context_q, [-1, 4, self._sentence_embedding_size])
            sentence_representation_a = context_a
            print('sentence_representation_q ', sentence_representation_q)
            print('sentence_representation_a ', sentence_representation_a)
            context_q = tf.reshape(context_q, [-1, NUM_ANSWER_FOR_PER_QUESTION, self._sentence_embedding_size])
            context_a = tf.reshape(context_a, [-1, NUM_ANSWER_FOR_PER_QUESTION, self._sentence_embedding_size])
            print('context_q ', context_q, ' context_a ', context_a)
            similarity_per_batch = tf.reduce_sum(context_q * context_a, axis=-1)
            print('similarity_per_batch ', similarity_per_batch)
            # logits = tf.square(similarity_per_batch - self.label)  # point-wise
            logits = tf.add(THRESHOLD, tf.reduce_sum(
                similarity_per_batch * tf.cast(self.label, tf.float32), axis=-1) * (-1))  # pair-wise
            logits = tf.map_fn(lambda x: tf.cond(x > 0.0, lambda: x, lambda: 0.0), logits, dtype=tf.float32,
                               name="logits")
            print('logits ', logits)
            loss_op = tf.reduce_mean(logits, name="loss_op")
            tf.summary.scalar("similarity_loss", loss_op)
            # prob_op: the indexes of one answer with the largest prob
            prob_op = tf.cast(tf.argmax(similarity_per_batch, axis=1), tf.int32)  # shape=[batch_size]
            precise_op = self._batch_size - tf.reduce_sum(prob_op)
            print('precise_op ', precise_op)

            # update gradient. (clip gradient and add noise)
            grads_and_vars = self._opt.compute_gradients(loss_op)
            # print('grad_var: ', grads_and_vars)  # ----------
            grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
            grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
            train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")
            return loss_op, precise_op, train_op, similarity_per_batch, logits, sentence_representation_q, sentence_representation_a

    def show_emb_matrix(self):
        return self._question_word_emb_matrix, self._answer_word_emb_matrix


VOCAB_SIZE_QUESTION = 30000  # 33344
VOCAB_SIZE_ANSWER = 50000  # 94268


def main():
    with tf.variable_scope("attention_model"):
        model = AttentionModel(VOCAB_SIZE_QUESTION, VOCAB_SIZE_ANSWER, FLAGS.batch_size, FLAGS.word_embedding_size,
                               FLAGS.sentence_embedding_size, FLAGS.num_layer)

    training_dataset_loader = TrainingDatasetLoader(NUM_ANSWER_FOR_PER_QUESTION, FLAGS.batch_size,
                                                    "./split_training_Question.dat", "./split_training_Answer.dat")
    print('training_dataset_loader.num_batch ', training_dataset_loader.num_batch)
    # question_emb_matrix, answer_emb_matrix = model.show_emb_matrix()
    batch_loss, precise_indices, train_op, similarity_per_batch_, logits_, sentence_q, sentence_a = model.forward()
    testing_set_loader = TestingDatasetLoader(NUM_ANSWER_FOR_PER_QUESTION, FLAGS.batch_size,
                                              "./split_testing_Question.dat", "./split_testing_Answer.dat")
    print('testing_set_loader.num_batch ', testing_set_loader.num_batch)

    saver = tf.train.Saver(max_to_keep=20)
    global_step = 118991  # ----------
    summary_merge_op = tf.summary.merge_all()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  # ----------
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # tf.global_variables_initializer().run()
        saver.restore(sess, FLAGS.checkpoint_dir + "attention_model_ckpt-118980")  # ----------
        # 11898 23796 35694 47592 59490 71388 83286 95184 107082 118980
        # 130878 142776 154674 166572 178470 190368 202266 214164 226062

        logdir_i = FLAGS.log_dir
        summary_writer = tf.summary.FileWriter(logdir=logdir_i, graph=sess.graph)
        save_path = FLAGS.checkpoint_dir
        print("version_4.0_attention_model_4.0 opt=Momentum(0.03, 0.9). decrease_rate=0.3 with projector")
        print("UPDATE: learning_rate decrease only when abs(precise_test - precises_test[-2]) < 1e-2")
        print("UPDATE: sentence_representation = average_sentence_layer_output")
        print("lr=0.009 when epoch=1, lr=0.0027 when epoch=10")
        print("dropout=0.5, max_grad_norm=5.0, no word_GRU_layer.")
        print("loss_function=max(0, 1-(cos_1-cos_2)), sentence_representation=average_sentence_layer_output")
        print("Train_set checkpoint. save_path: ", save_path)  # ----------

        attention_model_learning_rate = FLAGS.learning_rate
        precises_train = [0]  # ----------
        precises_test = [0]  # ----------
        for epoch in range(10, FLAGS.epoch_size):  # ----------
            print('**************epoch************************', epoch)
            print("learning_rate is ", attention_model_learning_rate)
            start_time = time.time()
            precise_num_train = 0
            for batch in range(training_dataset_loader.num_batch):  # num_batch 11898
                question_batch, question_batch_length, answer_batch, answer_batch_length \
                    = training_dataset_loader.next_batch()
                # print("question_batch.len ", len(question_batch))
                # print("question_batch_length.len ", len(question_batch_length))
                # print("answer_batch.len ", len(answer_batch))
                # print("answer_batch_length.len ", len(answer_batch_length))
                loss_mean_train, precise_num_batch_train, _, summary_merge, similarity_per_batch_train, logits_train = \
                    sess.run([batch_loss, precise_indices, train_op, summary_merge_op, similarity_per_batch_, logits_],
                             feed_dict={model.question: question_batch,
                                        model.question_length: question_batch_length,
                                        model.answer: answer_batch,
                                        model.answer_length: answer_batch_length,
                                        model.label: np.asarray([[1, -1] for _ in range(FLAGS.batch_size)], np.int32),
                                        model.embedding_keep_prob: FLAGS.embedding_keep_prob,
                                        model.word_gru_keep_prob: FLAGS.word_gru_keep_prob,
                                        model.sentence_gru_keep_prob: FLAGS.sentence_gru_keep_prob,
                                        model.learning_rate: attention_model_learning_rate})
                summary_writer.add_summary(summary_merge, global_step)
                precise_num_train += precise_num_batch_train
                print('similarity_per_batch_train ', similarity_per_batch_train)
                print('logits_train ', logits_train)
                print('precise_num_batch_train ', precise_num_batch_train)
                print('in the ', epoch, ' epoch and ', batch, ' batch, the loss is ', loss_mean_train)
                if (global_step % training_dataset_loader.num_batch) == 0:  #
                    precise_train = float(precise_num_train) / (FLAGS.batch_size * training_dataset_loader.num_batch)
                    print("epoch: ", epoch, " step: ", global_step, " precise_num_train: ", precise_num_train,
                          " precise_train: ", precise_train)
                    precises_train.append(precise_train)
                    with codecs.open(FLAGS.output_dir + "precise_temp.txt", "a", "utf-8") as temp_file_train:
                        temp_file_train.write("precise_train=" + str(precise_train))

                    if not os.path.exists(save_path):
                        print('no dir')
                        os.makedirs(save_path)
                    else:
                        print("dir exists")
                    saver.save(sess, save_path + "attention_model_ckpt", global_step=global_step)
                    print('**************save********step=**********', global_step)
                del question_batch
                del question_batch_length
                del answer_batch
                del answer_batch_length
                del similarity_per_batch_train
                del logits_train
                # print('Epoch: {} . epoch_time: {}'.format(epoch, (time.time() - start_time)))

                if global_step % training_dataset_loader.num_batch == 0:  # epoch % FLAGS.evaluation_interval == 0:
                    # question_matrix, answer_matrix = sess.run([question_emb_matrix, answer_emb_matrix])
                    # print("question_matrix[:2, :5] ", question_matrix[:2, :5])
                    # del question_matrix
                    # del answer_matrix

                    precise_num_test = 0
                    record_precise_batch = []
                    record_state_q = []
                    record_state_a = []
                    for batch_test in range(testing_set_loader.num_batch):  #
                        question_batch_test, question_batch_length_test, answer_batch_test, answer_batch_length_test \
                            = testing_set_loader.next_batch()
                        loss_mean_test, precise_num_batch_test, similarity_per_batch_test, logits_test, state_q, state_a = \
                            sess.run(
                                [batch_loss, precise_indices, similarity_per_batch_, logits_, sentence_q, sentence_a],
                                feed_dict={model.question: question_batch_test,
                                           model.question_length: question_batch_length_test,
                                           model.answer: answer_batch_test,
                                           model.answer_length: answer_batch_length_test,
                                           model.label: np.asarray(
                                               [[1, -1] for _ in range(FLAGS.batch_size)],
                                               np.int32),
                                           model.embedding_keep_prob: 1.0,
                                           model.word_gru_keep_prob: 1.0,
                                           model.sentence_gru_keep_prob: 1.0})
                        print('similarity_per_batch_test ', similarity_per_batch_test)
                        print('logits_test ', logits_test)
                        print('precise_num_batch_test ', precise_num_batch_test)
                        print('in the ', epoch, ' epoch test and ', batch_test, ' batch, the loss is ', loss_mean_test)
                        record_precise_batch.append(precise_num_batch_test)
                        precise_num_test += precise_num_batch_test
                        record_state_q.append(np.reshape(state_q[:, 0, :], [-1, FLAGS.sentence_embedding_size]))
                        record_state_a.append(state_a)
                        del question_batch_test
                        del question_batch_length_test
                        del answer_batch_test
                        del answer_batch_length_test
                        del similarity_per_batch_test
                        del logits_test
                        del state_q
                        del state_a
                    precise_test = float(precise_num_test) / (FLAGS.batch_size * testing_set_loader.num_batch)
                    print("Test. epoch: ", epoch, " step: ", global_step, " precise_num_test: ", precise_num_test,
                          " precise_test: ", precise_test)
                    precises_test.append(precise_test)
                    with codecs.open(FLAGS.output_dir + "precise_temp.txt", "a", "utf-8") as temp_file:
                        temp_file.write(" , precise_test=" + str(precise_test) + " , learning_rate=" +
                                        str(attention_model_learning_rate) + "\n")

                    if (precise_test - precises_test[-2]) < 0:
                        print('this epoch gets worse result.')
                    if ((precise_test - precises_test[-2] < 1e-3) and
                            (0.001 < attention_model_learning_rate)):
                        attention_model_learning_rate = attention_model_learning_rate * FLAGS.decrease_rate
                        print('epoch ', epoch, ' done. update model.learning_rate to ', attention_model_learning_rate)
                    print('record_state_q.shape ', np.shape(record_state_q))
                    print('record_state_a.shape ', np.shape(record_state_a))
                    record_state_q_a = list((chain.from_iterable(
                        [chain.from_iterable(record_state_q), chain.from_iterable(record_state_a)])))
                    print('record_state_q_a.shape ', np.shape(record_state_q_a))
                    record_state_q_a_numpy = [list(map(str, record_state_q_a_i)) for record_state_q_a_i in
                                              record_state_q_a]
                    print('record_state_q_a_numpy.shape ', np.shape(record_state_q_a_numpy))
                    with codecs.open(FLAGS.log_dir_emb + "sentence_representation_embedding_epoch_{}.txt".format(epoch),
                                     "w", "utf-8") as embedding_inputs:
                        for i in record_state_q_a_numpy:
                            embedding_inputs.write(" ".join(i) + '\n')
                    print(
                        'epoch: {} . step: {} . epoch_time: {}'.format(epoch, global_step, (time.time() - start_time)))

                    print("epoch {} attention_model done. Saving csv.".format(epoch))
                    df_precises_batch = pandas.DataFrame(data={'precise_batch': record_precise_batch},
                                                         index=range(1, testing_set_loader.num_batch + 1),
                                                         columns=['precise_batch'])
                    df_precises_batch.index.name = 'test_batch_id'
                    df_precises_batch.to_csv(
                        FLAGS.output_dir + "precises_batch_epoch_{}_attention_model.csv".format(epoch))
                    print("csv save done.")
                    del record_precise_batch
                    del record_state_q
                    del record_state_a
                    del record_state_q_a
                    del record_state_q_a_numpy

                global_step += 1
        summary_writer.close()
        df_precises = pandas.DataFrame(
            data={'precises_train': precises_train, 'precises_test': precises_test},
            index=range(-1, FLAGS.epoch_size),
            columns=['precises_train', 'precises_test'])
        df_precises.index.name = 'epoch'
        df_precises.to_csv(FLAGS.output_dir + "precises_attention_model_2" + ".csv")
        print('ALL DONE.')
        del training_dataset_loader
        del testing_set_loader
        del precises_test


if __name__ == '__main__':
    main()
