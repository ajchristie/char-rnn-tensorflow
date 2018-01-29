import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
        ## Main network block is assembled (output weights and bias are below).

        ## below, shapes are given for defaults.
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length]) ## 50 x 50
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length]) ## 50 x 50
        self.initial_state = cell.zero_state(args.batch_size, tf.float32) ## 50

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size]) ## 128 x 65
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])   ## 65
        ## Output connections are wired up

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size]) ## 65 x 128
        inputs = tf.nn.embedding_lookup(embedding, self.input_data) ## 50 x 50 x 128
        ## Per docs, this is shape(ids) + shape(params)[1:], which is an odd way to write it.

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        ## cut the rows of input to fit the unrolled network.
        inputs = tf.split(inputs, args.seq_length, 1) ## list of fifty 50 x 1 x 128 tensors
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs] ## list of fifty 50 x 128 tensors
        ## ok, so, now the list covers the 50 time steps in the sequence. The first dimension of the tensor spans the batch and second dimension is embedding/rnn size.

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)
            ## it is not altogether clear to me what this loop is doing here or that it is ever actually invoked: when not training, sequence and batch are set to 1, which makes this loop unecessary. It seems like, in general, this is here to make it possible to produce a different state from the cell output, but all the cells we use either have hidden and output equal or produce the different hidden state themselves. Surely, part of the weirdness is due to using a legacy decoder rather than something more current?
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        ## concatenation gives a 50 x 6400 tensor, after reshape you get 2500 x 128

        ## now produce the final output on which loss will be calculated
        self.logits = tf.matmul(output, softmax_w) + softmax_b ## 2500 x 65
        self.probs = tf.nn.softmax(self.logits) ## distributions for the (2500) characters in the fifty 50-character-long sequences
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False) ## this is an initialization; it gets assigned a nonzero value in train.py; also, this could happen up at the top.
        tvars = tf.trainable_variables() ## returns list of all trainable variables,fed to gradients
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)
            ## run through the prime text to get the state set up for prediction

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
