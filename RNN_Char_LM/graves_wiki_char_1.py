import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

data_file = 'data/graves_wiki_data.txt'
x_file = 'data/graves_wiki_x.txt'
y_file = 'data/graves_wiki_y.txt'

def unique(filename = data_file):
    input_file = open(filename, 'rb')
    inp_con = input_file.read()
    vocab = np.sort(list(set(inp_con)))
    return vocab

def make_seq():
    input_file = open(data_file, 'rb')
    inp_con = input_file.readlines()
    input_file.close()

    input = open(x_file, 'w')
    output = open(y_file, 'w')

    n = 0
    for i in range(len(inp_con)):
        inp_con[i] = inp_con[i][:-1]
        if len(inp_con[i]) >= 10:
            for j in range(len(inp_con[i]) - 9):
                inp = inp_con[i][j: j+9]
                out = inp_con[i][j+9]

                if len(inp) == 9:
                    input.write(inp + "\n")
                    output.write(out + "\n")
                    n = n+1

    print("The number of sequences generated :", n)
    input.close()
    output.close()


def word2vec(_string, alphabet= unique()):
    vector = [[0 if char != letter else 1 for char in alphabet]
                  for letter in _string]
    return vector

def corpus2vec(inp_con, op_con):
    i = 0
    for word in inp_con:
        #print(word)
        word = word[:-2]
        #print(word)
        temp = np.array(word2vec(word))
        #print("temp_x_shape", temp.shape)
        temp = temp.reshape((1, temp.shape[0], temp.shape[1]))
        if i == 0:
            vec_ip = temp
        else:
            vec_ip = np.append(vec_ip, temp, axis= 0 )
        i = i+1

    i = 0
    for word in op_con:
        #print(word)
        word = word[:-2]
        #print(word)
        temp = np.array(word2vec(word))
        #print("temp_y_shape", temp.shape)
        temp = temp.reshape((1, temp.shape[1]))
        if i == 0:
            vec_op = temp
        else:
            vec_op = np.append(vec_op, temp, axis=0)
        i = i + 1
    return vec_ip, vec_op


def train_test():
    input = open(x_file, 'rb')
    inp_con = input.readlines()
    input.close()

    output = open(y_file, 'rb')
    op_con = output.readlines()
    output.close()

    inp_con, op_con = shuffle(inp_con, op_con)
    inp_train, inp_test, op_train, op_test = train_test_split(inp_con, op_con, test_size=0.15)

    return inp_train, op_train, inp_test,  op_test


n_timesteps = 9
n_classes = len(unique())


class Wiki:
    def __init__(self):
        self.num_units = 32

    def create_model(self):
        self.x = tf.placeholder(tf.float32, [None, n_timesteps, n_classes])
        self.y = tf.placeholder(tf.float32, [None, n_classes])

        with tf.variable_scope("LSTM_1"):
            cell = tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True)
            val_1, state_1 = tf.nn.dynamic_rnn(cell, self.x, dtype= tf.float32)

        with tf.variable_scope("LSTM_2"):
            cell = tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True)
            val_2, state_2 = tf.nn.dynamic_rnn(cell, val_1, dtype=tf.float32)

        with tf.variable_scope("LSTM_3"):
            cell = tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True)
            val_3, state_3 = tf.nn.dynamic_rnn(cell, val_2 + val_1 , dtype=tf.float32)

        lstm_op = val_3 + val_2 + val_1

        time_major = tf.transpose(lstm_op, [1, 0, 2])
        last_step = time_major[-1]

        with tf.variable_scope("Linear"):
            w = tf.Variable(tf.truncated_normal([self.num_units,n_classes]), name= "weights")
            b = tf.Variable(tf.truncated_normal([n_classes]), name="bias")

            output = tf.matmul(last_step, w) + b

        output_softmax = tf.nn.softmax(output)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels= self.y, logits= output)
        self.mean_loss = tf.reduce_mean(loss)

        self.optimizer = tf.train.RMSPropOptimizer(0.0001).minimize(self.mean_loss)

        true = tf.argmax(self.y, axis=1)
        pred = tf.argmax(output_softmax, axis=1)
        corrects = tf.cast(tf.equal(true, pred), tf.float32)
        self.acc = tf.reduce_mean(corrects)

        print("RNN Ready")

    def train(self, mode, epochs = 200, batch_size = 4 ):
        with tf.Session() as sess:
            if (mode == "Load"):
                loader = tf.train.import_meta_graph("model/rnn_char_1.meta")
                loader.restore(sess, "model/rnn_char_1.ckpt")
                print("Model restored")

            elif (mode == "New"):
                init = tf.global_variables_initializer()
                sess.run([init])
                print("Weights are initialized")

            self.saver = tf.train.Saver()

            inp_train, op_train, inp_test, op_test = train_test()

            train_batches = len(inp_train)//batch_size

            for e in range(epochs):
                total_loss = 0
                total_acc = 0
                start = 0
                for batch in range(train_batches):
                    end = min(len(inp_train) - 1, start + batch_size)
                    if batch%500 == 0:
                        #print("epoch", e, "of", epochs, "\tbatch", batch, "of", train_batches)
                        pass
                    x_temp = inp_train[start:end]
                    y_temp = op_train[start:end]

                    x_train, y_train = corpus2vec(x_temp, y_temp)

                    feed = {self.x: x_train, self.y: y_train}

                    _, bl, ba = sess.run([self.optimizer, self.mean_loss, self.acc], feed_dict=feed)
                    total_loss += bl
                    total_acc += ba
                    start = end

                total_loss = total_loss / train_batches
                total_acc = total_acc / train_batches
                print("Train Epoch %d loss %0.4f acc %0.3f" % (e, total_loss, total_acc))

                if e%1==0:                
                    test_batches = len(inp_test)//batch_size
                    total_loss = 0
                    total_acc = 0
                    start = 0
                    for batch in range(test_batches):
                        end = min(len(inp_test) - 1, start + batch_size)

                        x_temp = inp_test[start:end]
                        y_temp = op_test[start:end]

                        x_test, y_test = corpus2vec(x_temp, y_temp)

                        feed_t = {self.x: x_test, self.y: y_test}

                        bl, ba = sess.run([ self.mean_loss, self.acc], feed_dict=feed_t)
                        total_loss += bl
                        total_acc += ba
                        start = end
                total_loss = total_loss / test_batches
                total_acc = total_acc / test_batches
                print("Test Epoch %d loss %0.4f acc %0.3f \n" % (e, total_loss, total_acc))
                self.saver.save(sess, "models/rnn_char_1.ckpt")

                
if __name__ == "__main__":
    # make_seq()

    # MODE = New or Load
    mode = "New"
    obj = Wiki()
    if (mode == 'New'):
        obj.create_model()
    obj.train(mode, epochs = 500, batch_size = 8)
