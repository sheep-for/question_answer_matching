# encoding:utf-8

import numpy as np
import codecs
import re
import os
import pandas as pd
import collections
from operator import itemgetter
from itertools import chain
from functools import reduce
import random
import gensim.models.word2vec as word2vec


import platform
if platform.system() == 'Windows':
    import pickle
else:
    import cPickle as pickle


def build_pre_train_emb_matrix(emb_vec_file, vocab_file):
    embedding = []
    temp_emb = dict()
    question_vec_file = codecs.open(emb_vec_file, 'r', 'utf-8')
    for index, line in enumerate(question_vec_file.readlines()):
        if index != 0:
            key = line.strip().split(' ')[0]
            vector = list(line.strip().split(' ')[1:])
            temp_emb[key] = vector
    question_vec_file.close()

    vocab_word = list(chain.from_iterable(pd.DataFrame(
        pd.read_csv(vocab_file, encoding='utf_8_sig'), columns=['word']).values))
    count = 0

    for i, item in enumerate(vocab_word):
        if item in temp_emb:
            vec = temp_emb[item]
            embedding.append(vec)
            count += 1
        else:
            print("new word.", item, "idx ", i, " initialized as <unk>")
            vec = temp_emb['\0']
            embedding.append(vec)
    
    del temp_emb
    return np.array(embedding, np.float32)


def get_keyword_num_log_tfidf(sentence_length, input_type="question"):
    # for q: keyword_num = min(10 * ln(x), x)
    # for a: keyword_num = min(x[logx] / lg(2x), x) 
    """
    :param sentence_length: numpy.int32, sequence_length of inputs
    :param input_type: question or answer
    :return: the same shape and type with sentence_length,
    the keyword_num for attention_layer and representation_layer
    """

    sentence_length = np.array(sentence_length, np.int32)
    if input_type == "question":
        keyword_num = np.minimum(
            np.int32(10 * np.log(sentence_length)), sentence_length)
    elif input_type == "answer":
        keyword_num = np.minimum(
            np.int32(sentence_length * np.int32(np.log10(sentence_length)) / np.log10(2.0 * sentence_length)),
            sentence_length)
    else:
        keyword_num = sentence_length
    if 0 in keyword_num:
        keyword_num = np.where(keyword_num > 0, keyword_num, np.ones_like(sentence_length))
    return keyword_num


class TrainingDatasetLoader(object):

    def __init__(self, num_answer_for_per_question, batch_size, train_file):
        # self.num_answer_for_per_question = num_answer_for_per_question
        self.question_batch_size = batch_size * num_answer_for_per_question // 8 
        self.train_file = train_file
        self.question = []
        self.question_length = []
        self.answer = []
        self.answer_length = []
        self.pointer = 0

        self.create_batches()

    def create_batches(self):
        question_input = codecs.open(self.train_file, 'r', 'utf-8')
        question_data = question_input.readlines()
        for index_q, line_q in enumerate(question_data):
            line_str = line_q.strip().split('\t')
            line_int = [int(x) for x in line_str[0].split()]
            self.question.append(line_int)
            self.question_length.append(len(line_int))
            for i in range(1, 6):
                answer_i_str = line_str[i].split()
                answer_i_int = [int(x) for x in answer_i_str]
                self.answer.append(answer_i_int)
                self.answer_length.append(len(answer_i_int))
        del question_data
        question_input.close()

        self.num_batch = len(self.question_length) // self.question_batch_size
        self.question = self.question[: self.num_batch * self.question_batch_size]
        self.question_length = self.question_length[: self.num_batch * self.question_batch_size]
        self.answer = self.answer[: self.num_batch * 5 * self.question_batch_size]
        self.answer_length = self.answer_length[: self.num_batch * 5 * self.question_batch_size]
        
        self.point = list(range(self.num_batch))
        random.shuffle(self.point)
        
    def next_batch(self):
        question_batch_length_before_tile = self.question_length[
            self.point[self.pointer] * self.question_batch_size: (self.point[self.pointer] + 1) * self.question_batch_size]
        max_question_batch_length = max(question_batch_length_before_tile)
        question_batch_before_tile = self.question[
                                     self.point[self.pointer] * self.question_batch_size: (self.point[self.pointer] + 1) * self.question_batch_size]
        for idx_q, q in enumerate(question_batch_before_tile):
            if len(q) < max_question_batch_length:
                question_batch_before_tile[idx_q] = q + [0] * (max_question_batch_length - len(q))
        question_batch = []
        question_batch_length = []
        for question_before_tile in question_batch_before_tile:
            for _ in range(8):
                question_batch.append(question_before_tile)
        for question_length_before_tile in question_batch_length_before_tile:
            for _ in range(8):
                question_batch_length.append(question_length_before_tile)
        del question_batch_before_tile
        del question_batch_length_before_tile

        answer_batch_length_before_tile = self.answer_length[
                                          self.point[self.pointer] * 5 * self.question_batch_size: (self.point[self.pointer] + 1) * 5 * self.question_batch_size]
        max_answer_batch_length = max(answer_batch_length_before_tile)
        answer_batch_before_tile = self.answer[self.point[self.pointer] * 5 * self.question_batch_size: (self.point[self.pointer] + 1) * 5 * self.question_batch_size]
        for idx_a, a in enumerate(answer_batch_before_tile):
            if len(a) < max_answer_batch_length:
                answer_batch_before_tile[idx_a] = a + [0] * (max_answer_batch_length - len(a))
        answer_batch = []
        answer_batch_length = []
        for i in range(self.question_batch_size):
            for j in range(4):
                answer_batch.append(answer_batch_before_tile[(5*i+4)])
                answer_batch_length.append(answer_batch_length_before_tile[(5*i+4)])
                answer_batch.append(answer_batch_before_tile[(5*i+j)])
                answer_batch_length.append(answer_batch_length_before_tile[(5*i+j)])
        question_keyword_num = get_keyword_num_log_tfidf(question_batch_length, "question")
        answer_keyword_num = get_keyword_num_log_tfidf(answer_batch_length, "answer")

        self.pointer = (self.pointer + 1) % self.num_batch
        if self.pointer == 0:
            random.shuffle(self.point)
            print('data shuffle. point: ', self.point[:5])
        return np.array(question_batch, np.int32), np.array(question_batch_length, np.int32), question_keyword_num, np.array(
            answer_batch, np.int32), np.array(answer_batch_length, np.int32), answer_keyword_num

    def reset_pointer(self):
        self.pointer = 0
        random.shuffle(self.point)


class TestingDatasetLoader(object):

    def __init__(self, num_answer_for_per_question, batch_size, test_file):
        # self.num_answer_for_per_question = num_answer_for_per_question
        self.question_batch_size = batch_size * num_answer_for_per_question // 5  # 160/5=32
        self.test_file = test_file
        self.question = []
        self.question_length = []
        self.answer = []
        self.answer_length = []
        self.pointer = 0

        self.create_batches()

    def create_batches(self):
        question_input = codecs.open(self.test_file, 'r', 'utf-8')
        question_data = question_input.readlines()
        for index_q, line_q in enumerate(question_data):
            line_str = line_q.strip().split('\t')
            line_int = [int(x) for x in line_str[0].split()]
            self.question.append(line_int)
            self.question_length.append(len(line_int))
            for i in range(1, 6):
                answer_i_str = line_str[i].split()
                answer_i_int = [int(x) for x in answer_i_str]
                self.answer.append(answer_i_int)
                self.answer_length.append(len(answer_i_int))
        del question_data
        question_input.close()

        self.num_batch = len(self.question_length) // self.question_batch_size
        self.question = self.question[: self.num_batch * self.question_batch_size]
        self.question_length = self.question_length[: self.num_batch * self.question_batch_size]
        self.answer = self.answer[: self.num_batch * 5 * self.question_batch_size]
        self.answer_length = self.answer_length[: self.num_batch * 5 * self.question_batch_size]
        
        self.point = list(range(self.num_batch))
        
    def next_batch(self):
        question_batch_length_before_tile = self.question_length[
            self.point[self.pointer] * self.question_batch_size: (self.point[self.pointer] + 1) * self.question_batch_size]
        max_question_batch_length = max(question_batch_length_before_tile)
        question_batch_before_tile = self.question[
                                     self.point[self.pointer] * self.question_batch_size: (self.point[self.pointer] + 1) * self.question_batch_size]
        for idx_q, q in enumerate(question_batch_before_tile):
            if len(q) < max_question_batch_length:
                question_batch_before_tile[idx_q] = q + [0] * (max_question_batch_length - len(q))
        question_batch = []
        question_batch_length = []
        for question_before_tile in question_batch_before_tile:
            for _ in range(5):
                question_batch.append(question_before_tile)
        for question_length_before_tile in question_batch_length_before_tile:
            for _ in range(5):
                question_batch_length.append(question_length_before_tile)
        del question_batch_before_tile
        del question_batch_length_before_tile

        answer_batch_length = self.answer_length[
                                          self.point[self.pointer] * 5 * self.question_batch_size: (self.point[self.pointer] + 1) * 5 * self.question_batch_size]
        max_answer_batch_length = max(answer_batch_length)
        answer_batch = self.answer[self.point[self.pointer] * 5 * self.question_batch_size: (self.point[self.pointer] + 1) * 5 * self.question_batch_size]
        for idx_a, a in enumerate(answer_batch):
            if len(a) < max_answer_batch_length:
                answer_batch[idx_a] = a + [0] * (max_answer_batch_length - len(a))
        question_keyword_num = get_keyword_num_log_tfidf(question_batch_length, "question")
        answer_keyword_num = get_keyword_num_log_tfidf(answer_batch_length, "answer")

        self.pointer = (self.pointer + 1) % self.num_batch
        
        return np.array(question_batch, np.int32), np.array(question_batch_length, np.int32), question_keyword_num, np.array(
            answer_batch, np.int32), np.array(answer_batch_length, np.int32), answer_keyword_num

    def reset_pointer(self):
        self.pointer = 0
