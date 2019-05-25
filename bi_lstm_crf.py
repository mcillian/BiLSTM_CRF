import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers

class BiLSTM_CRF():
    def __init__(self, embeddings, lstm_dim_=100, num_tags_=4, lr_=0.001):
        self.lstm_dim = lstm_dim_
        self.num_tags = num_tags_
        self.lr = lr_
        self.initializer = initializers.xavier_initializer()
        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')
        self.max_steps = tf.placeholder(dtype=tf.int32, shape=[None, ], name='seq_length')
        self.x_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_input')
        self.y_input = tf.placeholder(dytepe=tf.int32, shape=[None, None], name='y_input')
        self.num_steps = tf.shape(self.x_input)[-1]
        with tf.variable_scope("char_embedding"):
            self.embeddings = tf.get_variable(name='embedding')
        self.logits = self.project_layer_single(self.bigru_layer())
        with tf.variable_scope("crf_loss"):
            self.trans = tf.get_variable(name="transistions", shape=[self.num_tags, self.num_tags],
                                         initializers=self.initializer)
            self.loss = self.loss_layer(self.logits)
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def bilstm_layer(self):
        embed_ = tf.nn.embedding_lookup(self.embeddings, self.x_input)
        with tf.variable_scope("char_bilstm"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True)
            lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True)
            outputs, outputs_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                     embed_, dtype=tf.float32)
        x_in_ = tf.concat(outputs, axis=2)
        return x_in_

    def bigru_layer(self):
        embed_ = tf.nn.embedding_lookup(self.embeddings, self.x_input)
        with tf.variable_scope("char_bigru"):
            lstm_fw_cell = rnn.GRUBlockCell(self.lstm_dim)
            lstm_bw_cell = rnn.GRUBlockCell(self.lstm_dim)
            outputs, outputs_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                     embed_, dtype=tf.float32)
        x_in_ = tf.concat(outputs, axis=2)
        return x_in_

    def project_layer(self):
        with tf.variable_scope("project"):
            with tf.variable_scope("hidden"):
                w_tanh = tf.get_variable(name="w_tanh",shape=[self.lstm_dim*2, self.lstm_dim],
                                         initializers=self.initializer,
                                         regularizer=tf.contrib.layers.l2_regularizer(0.01))
                b_tanh = tf.get_variable(name="b_tanh", shape=[self.lstm_dim],
                                         initializers=tf.zeros_initializer())
                x_in_ = tf.reshape(tesor=x_in_, shape=[-1, self.lstm_dim*2])
                output = tf.tanh(tf.add(tf.matmul(x_in_, w_tanh), b_tanh))

            with tf.variable_scope("output"):
                w_out = tf.get_variable(name="w_out", shape=[self.lstm_dim, self.num_tags],
                                        initializers=self.initializer,
                                        regularizer=tf.contrib.layers.l2_regularizer(0.01))
                b_out = tf.get_variable(name="b_out", shape=[self.num_tags],
                                        initializers=tf.zeros_initializer())
                pred_ = tf.add(tf.matmul(output, w_out), b_out)
                logits_ = tf.reshape(tensor=pred_, shape=[-1, self.num_tags, self.num_tags], name='logits')
        return logits_

    def peoject_layer_single(self):
        with tf.variable_scope("output"):
            w_out = tf.get_variable(name="w_out", shape=[self.lstm_dim, self.num_tags],
                                    initializers=self.initializer,
                                    regularizer=tf.contrib.layers.l2_regularizer(0.01))
            b_out = tf.get_variable(name="b_out", shape=[self.num_tags],
                                    initializers=tf.zeros_initializer())
            x_in_ = tf.reshape(tesor=x_in_, shape=[-1, self.lstm_dim * 2])
            pred_ = tf.add(tf.matmul(x_in_, w_out), b_out)
            logits_ = tf.reshape(tensor=pred_, shape=[-1, self.num_tags, self.num_tags], name='logits')
        return logits_

    def loss_layer(self):
        with tf.variable_scope("crf_loss"):
            log_likelihood, trans = crf_log_likelihood(input=project_logits, tag_indices=self.y_input,
                                                       transition_params=self.trans,
                                                       sequence_lengths=self.max_steps)
        return tf.reduce_mean(-log_likelihood)

