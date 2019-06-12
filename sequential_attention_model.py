from __future__ import print_function, division

import tensorflow as tf
import pandas
import time
from data_utils import *
from custom_attention_wrapper import CustomBahdanauAttentionV2
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Q_A pair: Q_A-_A-_A-_A-_A+ one question with four fake answers and one true answer
NUM_ANSWER_FOR_PER_QUESTION = 2  # for per Q: (Q, A+, A-)
NUM_FAKE_ANSWERS_FOR_PER_QUESTION = 4
NUM_ALL_ANSWERS_FOR_PER_QUESTION = 5
# threshold t for loss function
THRESHOLD = 0.1  
VERSION = 'SAKM_yahoo'

tf.flags.DEFINE_string("log_dir",
                       "./sequential_attention_model_yahoo" +
                       "/log/",
                       "log_dir")
tf.flags.DEFINE_string("log_dir_emb",
                       "./sequential_attention_model_yahoo" +
                       "/log/word_embedding/",
                       "visualize_embeddings_log_dir")
tf.flags.DEFINE_string("checkpoint_dir",
                       "./sequential_attention_model_yahoo" +
                       "/checkpoint/",
                       "checkpoint_dir")
tf.flags.DEFINE_string("train_data_dir", "./data_yahoo/", "train_data_dir")
tf.flags.DEFINE_string("output_dir",
                       "./sequential_attention_model_yahoo" +
                       "/output/",
                       "output_dir")
tf.flags.DEFINE_integer("word_embedding_size", 300, "word_embedding_size")
tf.flags.DEFINE_integer("sentence_embedding_size", 300, "sentence_embedding_size")
tf.flags.DEFINE_integer("batch_size", 20, "batch_size//4 = num_training_question, "
                                          "and NUM_ANSWER_FOR_PER_QUESTION*batch_size=num_answer")
tf.flags.DEFINE_float("embedding_keep_prob", 0.5, "keep_prob for emb_input")
tf.flags.DEFINE_float("word_gru_keep_prob", 0.5, "keep_prob for word_GRU network output")
tf.flags.DEFINE_float("sentence_gru_keep_prob", 1.0, "keep_prob for sentence_GRU network output")
tf.flags.DEFINE_float("sentence_output_keep_prob", 1.0, "keep_prob for sentence_layer_output")
tf.flags.DEFINE_string("optimizer", "tf.train.MomentumOptimizer(learning_rate=1.0, momentum=0.9)",
                       "optimizer for training, Default SGD+Momentum")
tf.flags.DEFINE_float("learning_rate", 0.003, "learning_rate for optimizer. Default 0.01") 
tf.flags.DEFINE_float("decrease_rate", 0.3, "decrease_rate for learning_rate. Default 0.3")
tf.flags.DEFINE_integer("attention_hops", 3, "num_hops for sequential_attention") 
tf.flags.DEFINE_integer("epoch_size", 15, "epoch_size. Default 10")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "max_grad_norm for clip_gradient. Default 40.0")
tf.flags.DEFINE_integer("num_layer", 1, "num_layer of sentence representation GRU. Default 1")
tf.flags.DEFINE_boolean("share_emb_and_softmax", True, "if emb_matrix and softmax_layer share the same variables.")
tf.flags.DEFINE_integer("evaluation_interval", 1, "evaluation_interval for save model")
tf.flags.DEFINE_float("q_top_alignment_percent", 40, "q_top_alignment_percent")
tf.flags.DEFINE_float("a_top_alignment_percent", 14, "a_top_alignment_percent")
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


class SequentialAttentionModel(object):
    def __init__(self, question_vocab_size, answer_vocab_size,
                 batch_size, word_embedding_size, sentence_embedding_size, num_layer,
                 max_grad_norm=5.0, initializer=tf.random_normal_initializer(stddev=0.1),
                 share_emb_and_softmax=True,
                 name="sequential_attention_model"):
        """
        Creates an End-To-End Q-A matching sequential_attention_model
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
        self._batch_answer_size = NUM_ANSWER_FOR_PER_QUESTION * self._batch_size  # 160
        self._word_embedding_size = word_embedding_size
        self._sentence_embedding_size = sentence_embedding_size
        self._max_grad_norm = max_grad_norm
        self._initializer = initializer
        self._share_emb_and_softmax = share_emb_and_softmax
        self._name = name
        self._rate = tf.constant([[0.2], [0.3], [0.5]], dtype=tf.float32,
                                 shape=[FLAGS.attention_hops, 1], name="attention_hops_rate")
        # self._q_top_alignment_percent = FLAGS.q_top_alignment_percent
        # self._a_top_alignment_percent = FLAGS.a_top_alignment_percent

        # input
        # question_length: the real sequence length of each question in a batch
        # answer_length: real sequence length of each answer in a batch
        self.question = tf.placeholder(tf.int32, [NUM_ANSWER_FOR_PER_QUESTION * self._batch_size, None], "question")
        self.answer = tf.placeholder(tf.int32, [NUM_ANSWER_FOR_PER_QUESTION * self._batch_size, None], "answer")
        self.question_length = tf.placeholder(tf.int32, [NUM_ANSWER_FOR_PER_QUESTION * self._batch_size],
                                              "question_length")
        self.question_keyword_num = tf.placeholder(tf.int32, [self._batch_answer_size],
                                                   "question_keyword_num")
        self.answer_length = tf.placeholder(tf.int32, [NUM_ANSWER_FOR_PER_QUESTION * self._batch_size], "answer_length")
        self.answer_keyword_num = tf.placeholder(tf.int32, [self._batch_answer_size],
                                                 "answer_keyword_num")
        self.label = tf.placeholder(tf.int32, [self._batch_size, NUM_ANSWER_FOR_PER_QUESTION], "label")
        self.embedding_keep_prob = tf.placeholder(tf.float32, [], "embedding_keep_prob")
        self.word_gru_keep_prob = tf.placeholder(tf.float32, [], "word_gru_keep_prob")
        self.sentence_gru_keep_prob = tf.placeholder(tf.float32, [], "sentence_gru_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, [], "learning_rate")

        emb_q = build_pre_train_emb_matrix(FLAGS.train_data_dir + "yahoo_Question_vec_300.txt",
                                           FLAGS.train_data_dir + "yahoo_10000_Question_word_dict.csv")
        emb_a = build_pre_train_emb_matrix(FLAGS.train_data_dir + "yahoo_Answer_vec_300.txt",
                                           FLAGS.train_data_dir + "yahoo_20000_Answer_word_dict.csv")
        self._question_word_emb_matrix = tf.get_variable("question_word_emb_matrix", shape=emb_q.shape,
                                                         initializer=tf.constant_initializer(emb_q), trainable=True)
        self._answer_word_emb_matrix = tf.get_variable("answer_word_emb_matrix", shape=emb_a.shape,
                                                       initializer=tf.constant_initializer(emb_a), trainable=True)

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

        # self._opt = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-8)
        self._opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)

    def forward(self):
        with tf.variable_scope("sequential_attention_model"):
            # word_embedding and dropout for input
            with tf.device('/cpu:0'):
                question_word_emb_without_dropout = tf.nn.embedding_lookup(self._question_word_emb_matrix,
                                                                           self.question)
                answer_word_emb_without_dropout = tf.nn.embedding_lookup(self._answer_word_emb_matrix, self.answer)

            # sentence representation: build sentence_gru network for question and answer
            logits_batch_hops = []
            answer_sentence_output_hops = []
            question_sentence_output_hops = []
            answer_sentence_output_hops.append(
                tf.zeros([self._batch_answer_size, tf.reduce_max(self.answer_length), self._sentence_embedding_size],
                         dtype=tf.float32, name="answer_sentence_output_hop-1"))
            context_q_hops = []
            context_a_hops = []
            question_mask = tf.sequence_mask(lengths=self.question_length, maxlen=tf.reduce_max(self.question_length),
                                             dtype=tf.float32, name="question_mask")
            print("--question_mask-- ", question_mask)
            answer_mask = tf.sequence_mask(lengths=self.answer_length, maxlen=tf.reduce_max(self.answer_length),
                                           dtype=tf.float32, name="answer_mask")
            print("--answer_mask-- ", answer_mask)  
            for hop in range(FLAGS.attention_hops):
                with tf.variable_scope("question_sentence_gru_attention"):
                    question_word_emb = tf.nn.dropout(question_word_emb_without_dropout, self.embedding_keep_prob)

                    if hop > 0:
                        tf.get_variable_scope().reuse_variables()
                    # question_sentence_gru_state shape=[num_layer, batch_size, sentence_embedding_size]
                    question_sentence_attention_mechanism = CustomBahdanauAttentionV2(
                        self._sentence_embedding_size, answer_sentence_output_hops[-1],
                        top_alignment_number=self.answer_keyword_num,
                        memory_sequence_length=self.answer_length
                    )
                    question_sentence_attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                        self._question_sentence_cell, question_sentence_attention_mechanism,
                        attention_layer_size=self._sentence_embedding_size, alignment_history=True
                    )
                    question_sentence_zero_state = question_sentence_attention_cell.zero_state(
                        batch_size=NUM_ANSWER_FOR_PER_QUESTION * self._batch_size, dtype=tf.float32
                    )
                    question_sentence_train_helper = tf.contrib.seq2seq.TrainingHelper(
                        question_word_emb, self.question_length, time_major=False
                    )
                    question_sentence_decoder = tf.contrib.seq2seq.BasicDecoder(
                        question_sentence_attention_cell, question_sentence_train_helper,
                        question_sentence_zero_state
                    )
                    question_sentence_gru_outputs, question_sentence_gru_state, __ = \
                        tf.contrib.seq2seq.dynamic_decode(question_sentence_decoder,
                                                          output_time_major=False, impute_finished=True)
                    question_attention_matrices = question_sentence_gru_state.alignment_history.stack(
                        name="question_attention_matrices")
                    question_sentence_output = tf.identity(question_sentence_gru_outputs.rnn_output,  
                                                           name="question_sentence_output_hop{}".format(hop))
                    question_sentence_output_hops.append(question_sentence_output)
                    
                with tf.variable_scope("answer_sentence_gru_attention"):
                    answer_word_emb = tf.nn.dropout(answer_word_emb_without_dropout, self.embedding_keep_prob)
                    
                    if hop > 0:
                        tf.get_variable_scope().reuse_variables()
                    answer_sentence_attention_mechanism = CustomBahdanauAttentionV2(
                        self._sentence_embedding_size, question_sentence_output_hops[-1],
                        top_alignment_number=self.question_keyword_num,
                        memory_sequence_length=self.question_length)
                    answer_sentence_attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                        self._answer_sentence_cell, answer_sentence_attention_mechanism,
                        attention_layer_size=self._sentence_embedding_size, alignment_history=True)
                    answer_sentence_zero_state = answer_sentence_attention_cell.zero_state(
                        batch_size=NUM_ANSWER_FOR_PER_QUESTION * self._batch_size, dtype=tf.float32)
                    answer_sentence_train_helper = tf.contrib.seq2seq.TrainingHelper(
                        answer_word_emb, self.answer_length, time_major=False)
                    answer_sentence_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=answer_sentence_attention_cell, helper=answer_sentence_train_helper,
                        initial_state=answer_sentence_zero_state)
                    answer_sentence_gru_outputs, answer_sentence_gru_state, _ = tf.contrib.seq2seq.dynamic_decode(
                        answer_sentence_decoder, output_time_major=False, impute_finished=True)
                    answer_sentence_output = tf.identity(answer_sentence_gru_outputs.rnn_output,  
                                                         name="answer_sentence_output_hop{}".format(hop))
                    answer_sentence_output_hops.append(answer_sentence_output)
                    
                context_q = tf.reduce_sum(
                    tf.reduce_mean(question_sentence_output_hops, axis=0) * tf.expand_dims(question_mask, axis=2),
                    axis=1) / tf.reduce_sum(question_mask, axis=1, keep_dims=True)
                context_a = tf.reduce_sum(
                    tf.reduce_mean(answer_sentence_output_hops, axis=0) * tf.expand_dims(answer_mask, axis=2),
                    axis=1) / tf.reduce_sum(answer_mask, axis=1, keep_dims=True)
                norm_q = tf.norm(context_q, axis=1)
                context_q = context_q / tf.expand_dims(norm_q, axis=1)
                norm_a = tf.norm(context_a, axis=1)
                context_a = context_a / tf.expand_dims(norm_a, axis=1)

                context_q = tf.reshape(context_q, [-1, NUM_ANSWER_FOR_PER_QUESTION, self._sentence_embedding_size],
                                       name="context_q_hop{}".format(hop))  
                context_a = tf.reshape(context_a, [-1, NUM_ANSWER_FOR_PER_QUESTION, self._sentence_embedding_size],
                                       name="context_a_hop{}".format(hop))  
                context_q_hops.append(context_q)
                context_a_hops.append(context_a)

                similarity_per_batch = tf.reduce_sum(context_q * context_a, axis=-1,
                                                     name="similarity_per_batch_hop{}".format(hop))  
                logits = tf.add(THRESHOLD, tf.reduce_sum(
                    similarity_per_batch * tf.cast(self.label, tf.float32), axis=-1) * (-1))  # pair-wise
                logits = tf.map_fn(lambda x: tf.cond(x > 0.0, lambda: x, lambda: 0.0), logits, dtype=tf.float32,
                                   name="logits_hop{}".format(hop))  
                logits_batch_hops.append(logits)  
                if hop == (FLAGS.attention_hops - 1):
                    context_rate = [0.2 ** 0.5, 0.3 ** 0.5, 0.5 ** 0.5]
                    context_q_hops_weighted_mean = [context_q_hops[i] * context_rate[i] for i in
                                                    range(FLAGS.attention_hops)]
                    context_a_hops_weighted_mean = [context_a_hops[i] * context_rate[i] for i in
                                                    range(FLAGS.attention_hops)]
                    context_q_for_predict = tf.concat(context_q_hops_weighted_mean, axis=-1)
                    context_a_for_predict = tf.concat(context_a_hops_weighted_mean, axis=-1)
                    similarity_per_batch_all_hops = tf.reduce_sum(context_q_for_predict * context_a_for_predict,
                                                                  axis=-1, name="similarity_per_batch_all_hops")
                    logits_for_predict = tf.add(THRESHOLD, tf.reduce_sum(
                        similarity_per_batch_all_hops * tf.cast(self.label, tf.float32), axis=-1) * (-1))  # pair-wise
                    logits_for_predict = tf.map_fn(lambda x: tf.cond(x > 0.0, lambda: x, lambda: 0.0),
                                                   logits_for_predict, dtype=tf.float32,
                                                   name="logits_for_predict")  
                    prob_op = tf.cast(tf.argmax(similarity_per_batch_all_hops, axis=1), tf.int32)  # shape=[batch_size]
                    precise_op = self._batch_size - tf.reduce_sum(prob_op)

            logits_batch = logits_batch_hops * self._rate
            logits_batch = tf.reduce_sum(logits_batch, axis=0, name="logits_batch")  # shape=[batch_size]
            loss_op = tf.reduce_mean(logits_batch, name="loss_op")  # over batch
            tf.summary.scalar("similarity_loss", loss_op)

            # update gradient. (clip gradient and add noise)
            grads_and_vars = self._opt.compute_gradients(loss_op)
            grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
            grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
            grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
            train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")
            return loss_op, precise_op, train_op, question_attention_matrices

    def inference(self):
        with tf.variable_scope("sequential_attention_model"):
            with tf.device('/cpu:0'):
                question_word_emb_without_dropout = tf.nn.embedding_lookup(self._question_word_emb_matrix,
                                                                           self.question)
                answer_word_emb_without_dropout = tf.nn.embedding_lookup(self._answer_word_emb_matrix, self.answer)

            # sentence representation: build sentence_gru network for question and answer

            answer_sentence_output_hops = []
            question_sentence_output_hops = []
            answer_sentence_output_hops.append(
                tf.zeros([self._batch_answer_size, tf.reduce_max(self.answer_length), self._sentence_embedding_size],
                         dtype=tf.float32, name="answer_sentence_output_hop-1"))
            context_q_hops = []
            context_a_hops = []
            question_mask = tf.sequence_mask(lengths=self.question_length, maxlen=tf.reduce_max(self.question_length),
                                             dtype=tf.float32, name="question_mask")
            print("--question_mask-- ", question_mask)
            answer_mask = tf.sequence_mask(lengths=self.answer_length, maxlen=tf.reduce_max(self.answer_length),
                                           dtype=tf.float32, name="answer_mask")
            print("--answer_mask-- ", answer_mask)  
            for hop in range(FLAGS.attention_hops):
                with tf.variable_scope("question_sentence_gru_attention"):
                    question_word_emb = tf.nn.dropout(question_word_emb_without_dropout, self.embedding_keep_prob)

                    tf.get_variable_scope().reuse_variables()
                    question_sentence_attention_mechanism = CustomBahdanauAttentionV2(
                        self._sentence_embedding_size, answer_sentence_output_hops[-1],
                        top_alignment_number=self.answer_keyword_num,
                        memory_sequence_length=self.answer_length
                    )
                    question_sentence_attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                        self._question_sentence_cell, question_sentence_attention_mechanism,
                        attention_layer_size=self._sentence_embedding_size, alignment_history=True
                    )
                    question_sentence_zero_state = question_sentence_attention_cell.zero_state(
                        batch_size=NUM_ANSWER_FOR_PER_QUESTION * self._batch_size, dtype=tf.float32
                    )
                    question_sentence_train_helper = tf.contrib.seq2seq.TrainingHelper(
                        question_word_emb, self.question_length, time_major=False
                    )
                    question_sentence_decoder = tf.contrib.seq2seq.BasicDecoder(
                        question_sentence_attention_cell, question_sentence_train_helper,
                        question_sentence_zero_state
                    )
                    question_sentence_gru_outputs, question_sentence_gru_state, __ = \
                        tf.contrib.seq2seq.dynamic_decode(question_sentence_decoder,
                                                          output_time_major=False, impute_finished=True)
                    question_attention_matrices = question_sentence_gru_state.alignment_history.stack(
                        name="question_attention_matrices")
                    question_sentence_output = tf.identity(question_sentence_gru_outputs.rnn_output,  
                                                           name="question_sentence_output_hop{}".format(hop))
                    question_sentence_output_hops.append(question_sentence_output)
                    
                with tf.variable_scope("answer_sentence_gru_attention"):
                    answer_word_emb = tf.nn.dropout(answer_word_emb_without_dropout, self.embedding_keep_prob)
                    
                    tf.get_variable_scope().reuse_variables()
                    answer_sentence_attention_mechanism = CustomBahdanauAttentionV2(
                        self._sentence_embedding_size, question_sentence_output_hops[-1],
                        top_alignment_number=self.question_keyword_num,
                        memory_sequence_length=self.question_length)
                    answer_sentence_attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                        self._answer_sentence_cell, answer_sentence_attention_mechanism,
                        attention_layer_size=self._sentence_embedding_size, alignment_history=True)
                    answer_sentence_zero_state = answer_sentence_attention_cell.zero_state(
                        batch_size=NUM_ANSWER_FOR_PER_QUESTION * self._batch_size, dtype=tf.float32)
                    answer_sentence_train_helper = tf.contrib.seq2seq.TrainingHelper(
                        answer_word_emb, self.answer_length, time_major=False)
                    answer_sentence_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=answer_sentence_attention_cell, helper=answer_sentence_train_helper,
                        initial_state=answer_sentence_zero_state)
                    answer_sentence_gru_outputs, answer_sentence_gru_state, _ = tf.contrib.seq2seq.dynamic_decode(
                        answer_sentence_decoder, output_time_major=False, impute_finished=True)
                    answer_sentence_output = tf.identity(answer_sentence_gru_outputs.rnn_output,  
                                                         name="answer_sentence_output_hop{}".format(hop))
                    answer_sentence_output_hops.append(answer_sentence_output)
                    
                context_q = tf.reduce_sum(
                    tf.reduce_mean(question_sentence_output_hops, axis=0) * tf.expand_dims(question_mask, axis=2),
                    axis=1) / tf.reduce_sum(question_mask, axis=1, keep_dims=True)
                context_a = tf.reduce_sum(
                    tf.reduce_mean(answer_sentence_output_hops, axis=0) * tf.expand_dims(answer_mask, axis=2),
                    axis=1) / tf.reduce_sum(answer_mask, axis=1, keep_dims=True)
                norm_q = tf.norm(context_q, axis=1)
                context_q = context_q / tf.expand_dims(norm_q, axis=1)
                norm_a = tf.norm(context_a, axis=1)
                context_a = context_a / tf.expand_dims(norm_a, axis=1)
                context_q = tf.reshape(context_q, [-1, NUM_ALL_ANSWERS_FOR_PER_QUESTION, self._sentence_embedding_size],
                                       name="context_q_hop{}".format(hop))  
                context_a = tf.reshape(context_a, [-1, NUM_ALL_ANSWERS_FOR_PER_QUESTION, self._sentence_embedding_size],
                                       name="context_a_hop{}".format(hop))  
                context_q_hops.append(context_q)
                context_a_hops.append(context_a)
                if hop == (FLAGS.attention_hops - 1):
                    context_rate = [0.2 ** 0.5, 0.3 ** 0.5, 0.5 ** 0.5]
                    context_q_hops_weighted_mean = [context_q_hops[i] * context_rate[i] for i in
                                                    range(FLAGS.attention_hops)]
                    context_a_hops_weighted_mean = [context_a_hops[i] * context_rate[i] for i in
                                                    range(FLAGS.attention_hops)]
                    context_q_for_predict = tf.concat(context_q_hops_weighted_mean, axis=-1)
                    context_a_for_predict = tf.concat(context_a_hops_weighted_mean, axis=-1)
                    
                    score_batch = tf.reduce_sum(context_q_for_predict * context_a_for_predict, axis=-1)  
                    score_batch_ranked, rank_index = tf.nn.top_k(score_batch,
                                                                 k=NUM_ALL_ANSWERS_FOR_PER_QUESTION,
                                                                 sorted=True)  
                    
                    score_batch_hop_0 = tf.reduce_sum(context_q_hops[0] * context_a_hops[0], axis=-1)  
                    score_batch_ranked_hop_0, rank_index_hop_0 = tf.nn.top_k(score_batch_hop_0,
                                                                             k=NUM_ALL_ANSWERS_FOR_PER_QUESTION,
                                                                             sorted=True)  
                    score_batch_hop_last = tf.reduce_sum(context_q_hops[-1] * context_a_hops[-1], axis=-1)  
                    score_batch_ranked_hop_last, rank_index_hop_last = tf.nn.top_k(score_batch_hop_last,
                                                                                   k=NUM_ALL_ANSWERS_FOR_PER_QUESTION,
                                                                                   sorted=True)  
                    return rank_index, rank_index_hop_0, rank_index_hop_last


VOCAB_SIZE_QUESTION = 10000  
VOCAB_SIZE_ANSWER = 20000  


def main():
    with tf.variable_scope("sequential_attention_model"):
        model = SequentialAttentionModel(VOCAB_SIZE_QUESTION, VOCAB_SIZE_ANSWER, FLAGS.batch_size, FLAGS.word_embedding_size,
                                    FLAGS.sentence_embedding_size, FLAGS.num_layer)

    training_dataset_loader = TrainingDatasetLoader(NUM_ANSWER_FOR_PER_QUESTION, FLAGS.batch_size,
                                                    "./train_data.dat")
    print('training_dataset_loader.num_batch ', training_dataset_loader.num_batch)

    batch_loss, precise_indices, train_op, q_attention_matrices = model.forward()
    rank_indices, rank_indices_hop_0, rank_indices_hop_last = model.inference()

    saver = tf.train.Saver(max_to_keep=15)
    global_step = 1  
    summary_merge_op = tf.summary.merge_all()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        
        logdir_i = FLAGS.log_dir
        summary_writer = tf.summary.FileWriter(logdir=logdir_i, graph=sess.graph)
        save_path = FLAGS.checkpoint_dir
        
        sequential_attention_model_learning_rate = FLAGS.learning_rate
        precises_train = []

        for epoch in range(0, FLAGS.epoch_size):
            print('**************epoch************************', epoch)
            print("learning_rate is ", sequential_attention_model_learning_rate)
            start_time = time.time()
            precise_num_train = 0
            q_len_train = []
            a_len_train = []
            for batch in range(training_dataset_loader.num_batch):  
                question_batch, question_batch_length, question_keyword_num, answer_batch, answer_batch_length, answer_keyword_num \
                    = training_dataset_loader.next_batch()
                q_len_train.append(max(question_batch_length))
                a_len_train.append(max(answer_batch_length))
                loss_mean_train, precise_num_batch_train, _, summary_merge, q_attention_matrices_train = \
                    sess.run(
                        [batch_loss, precise_indices, train_op, summary_merge_op, q_attention_matrices],
                        feed_dict={model.question: question_batch,
                                   model.question_length: question_batch_length,
                                   model.question_keyword_num: question_keyword_num,
                                   model.answer: answer_batch,
                                   model.answer_length: answer_batch_length,
                                   model.answer_keyword_num: answer_keyword_num,
                                   model.label: np.asarray([[1, -1] for _ in range(FLAGS.batch_size)], np.int32),
                                   model.embedding_keep_prob: FLAGS.embedding_keep_prob,
                                   model.word_gru_keep_prob: FLAGS.word_gru_keep_prob,
                                   model.sentence_gru_keep_prob: FLAGS.sentence_gru_keep_prob,
                                   model.learning_rate: sequential_attention_model_learning_rate})
                summary_writer.add_summary(summary_merge, global_step)
                precise_num_train += precise_num_batch_train
                print('precise_num_batch_train ', precise_num_batch_train)
                print('in the ', epoch, ' epoch and ', batch, ' batch, the loss is ', loss_mean_train)
                if (global_step % training_dataset_loader.num_batch) == 0:  
                    precise_train = float(precise_num_train) / (
                            FLAGS.batch_size * training_dataset_loader.num_batch)
                    print("epoch: ", epoch, " step: ", global_step, " precise_num_train: ", precise_num_train,
                          " precise_train: ", precise_train)
                    precises_train.append(precise_train)
                    if not os.path.exists(FLAGS.output_dir):
                        print('no dir')
                        os.makedirs(FLAGS.output_dir)
                    with codecs.open(FLAGS.output_dir + "precise_temp.txt", "a", "utf-8") as temp_file_train:
                        temp_file_train.write("precise_train=" + str(precise_train) + " , learning_rate=" +
                                              str(sequential_attention_model_learning_rate) + "\n")

                    if not os.path.exists(save_path):
                        print('no dir')
                        os.makedirs(save_path)
                    else:
                        print("dir exists")
                    saver.save(sess, save_path + "sequential_attention_model_ckpt", global_step=global_step)
                    print('**************save********step=**********', global_step)
                del question_batch
                del question_batch_length
                del answer_batch
                del answer_batch_length
                # print('Epoch: {} . epoch_time: {}'.format(epoch, (time.time() - start_time)))

                global_step += 1
            print('q_len_train: ', pd.Series(q_len_train).describe())
            print('a_len_train: ', pd.Series(a_len_train).describe())
            del q_len_train
            del a_len_train
        summary_writer.close()
        
        del training_dataset_loader


if __name__ == '__main__':
    main()
