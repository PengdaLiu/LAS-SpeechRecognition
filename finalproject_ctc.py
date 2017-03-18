import tensorflow as tf
from model_ctc import Model
import numpy as np
from utils.general_utils import get_minibatches
import load_data as ld
import time
class Config:#Several +drop out
    n_classes = 28
    batch_size = 4
    #U = 16  
    n_mfcc = 120
    n_features = 40
    pblstm_hidden = 30
    aslstm_state =  40
    hidden_atten1 = 50
    hidden_atten2 = 5
    hidden_dist= 4
    context_final = 6
    model_output= "model_weights"
    #max_time = 128
    #timestep = 82
    n_epochs = 360
    drop = 0.9
    lr = 2e-3 

    def __init__(self, timestep=82, max_time=128):
        self.timestep=timestep
        self.max_time=max_time
        self.U=max_time/8

"""
def create_mask(sequence, timestep):
    return [s.index(s[-1]) * [True] + (timestep - s.index(s[-1])) * [False]]
"""

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
    	l = seq.tolist().index(seq[-1]) - 1
        indices.extend(zip([n]*l, range(l)))
        values.extend(seq[1:l+1]-1)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    # print(indices, values, shape)
    return indices, values, shape

class LASmodel(Model):
    def mlp(self,s,w1,b1,w2,b2):
        l1 = tf.add(tf.matmul(s,w1),b1)
        l2 = tf.add(tf.matmul(tf.nn.sigmoid(l1),w2),b2)
        return tf.nn.softmax(l2)
    
    def mlp3(self,s,w1,b1,w2,b2):
        l1 = tf.add(tf.matmul(s,w1),b1)
        l2 = tf.add(tf.matmul(tf.nn.sigmoid(l1),w2),b2)
        return l2
    
    def mlp2(self,h,w1,b1,w2,b2):
        s = tf.reshape(h, [-1, self.config.n_mfcc])
        l1 = tf.add(tf.matmul(s,w1),b1)
        l2 = tf.add(tf.matmul(tf.nn.sigmoid(l1),w2),b2)
        l2 = tf.reshape(l2, [-1, self.config.U,self.config.context_final ])
        return tf.nn.softmax(l2)
    
    def chadist(self,s,c,w1_dist,b1_dist,w2_dist,b2_dist):
        return self.mlp3(tf.concat(1,[s,c]),w1_dist,b1_dist,w2_dist,b2_dist)

    def to_onehot(self,i):
        onehot=np.zeros(self.config.n_classes)
        onehot[i]=1
        return tf.constant(onehot)

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32    , \
            shape=[None, self.config.max_time, self.config.n_features])
        self.lb_placeholder =tf.sparse_placeholder(tf.int32)
        #self.labels_placeholder =tf.placeholder(tf.int32, shape=[None, self.config.timestep])
        self.seq_len_placeholder = tf.placeholder(tf.int32, shape=[None,])
        self.mask_placeholder = tf.placeholder(tf.bool, shape=[None, self.config.timestep])

    def create_feed_dict(self, inputs_batch, labels_batch=None,seq_batch=None):
        feed_dict={self.input_placeholder:inputs_batch, self.seq_len_placeholder:seq_batch, \
                self.lb_placeholder:labels_batch}#,self.labels_placeholder:labels_batch}
        return feed_dict

    def attentioncontext(self,si,h,w1,b1,w2,b2,w11,b11,w22,b22):
        sii = self.mlp(si,w1,b1,w2,b2)
        e = tf.reduce_sum(tf.mul(tf.reshape(sii,[-1,1,self.config.context_final]),self.mlp2(h,w11,b11,w22,b22)),axis=2)
        e = tf.nn.softmax(e)
        return tf.reduce_sum(tf.mul(tf.reshape(e,[-1,self.config.U,1]),h),axis=1)
    
    def add_prediction_op(self):
        preds=[]
        init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32    )
        first=tf.one_hot([0], self.config.n_classes) 
        first=tf.tile(first,[ 1,tf.shape(self.input_placeholder)[0] ])
        preds.append(tf.reshape(first,[-1,self.config.n_classes]))
        ##Initialize variable
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.config.aslstm_state, state_is_tuple=True)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2,state_is_tuple=True)
        initial_state = state = stacked_lstm.zero_state(tf.shape(self.input_placeholder)[0], tf.float32)        #Variable initialization
        w1_atten=tf.Variable(init([self.config.aslstm_state, self.config.hidden_atten1]))
        b1_atten=tf.Variable(tf.zeros([ self.config.hidden_atten1, ]))
        w2_atten=tf.Variable(init([ self.config.hidden_atten1, self.config.context_final]))
        b2_atten=tf.Variable(tf.zeros([self.config.context_final , ]))
        w11_atten=tf.Variable(init([ self.config.n_mfcc, self.config.hidden_atten2]))
        b11_atten=tf.Variable(tf.zeros([ self.config.hidden_atten2, ]))
        w22_atten=tf.Variable(init([self.config.hidden_atten2 ,self.config.context_final ]))
        b22_atten=tf.Variable(tf.zeros([ self.config.context_final, ]))
        w1_dist=tf.Variable(init([ self.config.aslstm_state+self.config.n_mfcc, self.config.hidden_dist]))
        b1_dist=tf.Variable(tf.zeros([ self.config.hidden_dist, ]))
        w2_dist=tf.Variable(init([ self.config.hidden_dist, self.config.n_classes]))
        b2_dist=tf.Variable(tf.zeros([ self.config.n_classes, ]))

        hidden_size=self.config.pblstm_hidden
        max_time=self.config.max_time
        fcell_l1 = tf.nn.rnn_cell.LSTMCell(hidden_size)
        bcell_l1 = tf.nn.rnn_cell.LSTMCell(hidden_size)

        fcell_l2 = tf.nn.rnn_cell.LSTMCell(hidden_size)
        bcell_l2 = tf.nn.rnn_cell.LSTMCell(hidden_size)

        fcell_l3 = tf.nn.rnn_cell.LSTMCell(hidden_size)
        bcell_l3 = tf.nn.rnn_cell.LSTMCell(hidden_size)

        outputs_l1, state_l1 = tf.nn.bidirectional_dynamic_rnn(fcell_l1, bcell_l1, self.input_placeholder, \
            dtype=tf.float32    , sequence_length=self.seq_len_placeholder, scope="L1")
       # concat outputs_l1
        outputs_l1 = tf.reshape(tf.concat(2, tf.nn.dropout(outputs_l1, self.config.drop)), [-1, max_time / 2, hidden_size * 4])

        outputs_l2, state_l2 = tf.nn.bidirectional_dynamic_rnn(fcell_l2, bcell_l2, outputs_l1, \
            dtype=tf.float32    , sequence_length=self.seq_len_placeholder / 2, scope="L2")
    # concat outputs_l2
        outputs_l2 = tf.reshape(tf.concat(2, tf.nn.dropout(outputs_l2, self.config.drop)), [-1, max_time / 4, hidden_size * 4])

        outputs_l3, state_l3 = tf.nn.bidirectional_dynamic_rnn(fcell_l3, bcell_l3, outputs_l2, \
            dtype=tf.float32    , sequence_length=self.seq_len_placeholder / 4, scope="L3")
        # concat'ed outputs_l3
        h= tf.reshape(tf.concat(2, outputs_l3), [-1, max_time / 8, hidden_size * 4])

        h = tf.reshape(h, [-1, self.config.n_mfcc]) # =hidden * 4
        Wc = tf.Variable(tf.truncated_normal([self.config.n_mfcc, self.config.n_classes], stddev=0.1))
        bc = tf.Variable(tf.constant(0., shape=[self.config.n_classes]))
        return tf.reshape(tf.matmul(h, Wc) + bc, [-1, max_time / 8, self.config.n_classes])

        """
        initial_s = output = tf.zeros([tf.shape(self.input_placeholder)[0],self.config.aslstm_state], tf.float32)
        c=self.attentioncontext(output,h,w1_atten,b1_atten,w2_atten,b2_atten,w11_atten,b11_atten,w22_atten,b22_atten)
        
        for i in range(self.config.timestep-1):
            if i > 0: tf.get_variable_scope().reuse_variables()
            r=np.random.binomial(1,0.9)
            r=tf.constant(r,dtype=tf.float32)
            random_sample=tf.reshape(preds[i],[-1,self.config.n_classes])*r+(1-r)*tf.one_hot(self.labels_placeholder[:,i], self.config.n_classes, axis=-1,dtype=tf.float32)
            concatenated=tf.concat(1,[random_sample,c]) 
            output, state = stacked_lstm(concatenated, state)
            c = self.attentioncontext(output, h, w1_atten,b1_atten,w2_atten,b2_atten,w11_atten,b11_atten,w22_atten,b22_atten)
            ith_pred=tf.reshape(self.chadist(output,c,w1_dist,b1_dist,w2_dist,b2_dist),[-1,self.config.n_classes])
            preds.append(ith_pred)
        preds = tf.pack(preds, 1)        
        return preds
        """

    def add_decode_op(self, preds):
        preds_t = tf.transpose(preds, (1,0,2))
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=preds_t, sequence_length=self.seq_len_placeholder / 8)
        #decoded, log_prob = tf.nn.ctc_greedy_decoder(inputs=preds_t, sequence_length=self.seq_len_placeholder / 8)
        return preds, decoded
    
    def add_loss_op(self, preds):
    	preds = tf.transpose(preds, (1,0,2))
    	ctcloss = tf.nn.ctc_loss(preds, self.lb_placeholder, self.seq_len_placeholder / 8)
        return tf.reduce_mean(ctcloss)
        '''
        loss=0.0
        '''
        '''
        for i in range(self.config.timestep):
            if i>0 :
                embedding=tf.one_hot(self.labels_placeholder[:,i], self.config.n_classes, axis=-1,dtype=tf.float32)
                a= tf.reduce_mean(tf.log(tf.mul(tf.reshape(preds[i],[-1,self.config.n_classes]),embedding)))
                loss+=a
        return -loss
        '''
        '''
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=preds)
        loss=tf.reduce_sum(loss,axis=1)
        loss = tf.reduce_mean(loss)
        return loss
        '''

    def add_training_op(self, loss):
        #opt=tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)
        opt = tf.train.MomentumOptimizer(self.config.lr, 0.9)
        train_op=opt.minimize(loss)
        return train_op
        
    def __init__(self, config):
        self.config=config
        self.build()
    
    def run_epoch(self, sess, inputs, labels,seqs):
        """Runs an epoch of training.

        Args:
            sess: tf.Session() object
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch ,seq_batch in get_minibatches([inputs, labels,seqs], self.config.batch_size):
            n_minibatches += 1
            # print(sparse_tuple_from(labels_batch))
            total_loss += self.train_on_batch(sess, input_batch, sparse_tuple_from(labels_batch),seq_batch)
        return total_loss / n_minibatches
    
    #deleted 03/15 GZ
    #def run_test(self,sess,inputs,labels,seqs):
        #return self.test_on_batch(sess,inputs,labels,seqs)
    def predict_on_batch(self, sess, inputs_batch, labels_batch, seq_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        print(labels_batch)
        feed = self.create_feed_dict(inputs_batch, labels_batch=sparse_tuple_from(labels_batch), seq_batch=seq_batch)
        predictions = sess.run(self.decode, feed_dict=feed)
        return predictions

    def fit(self, saver,sess, inputs, labels,seqs):
        """Fit model on provided data.

        Args:
            sess: tf.Session()
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            losses: list of loss per epoch
        """
        losses = []
        for epoch in range(self.config.n_epochs):
            # if epoch==0: print(labels)
            start_time = time.time()
            average_loss = self.run_epoch(sess, inputs, labels,seqs)
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
            losses.append(average_loss)
        return losses

def test_LAS_model():
    train, max_time, max_chars = ld.load_data("../data_debug_13500_b1_nml.npz")
    # voxforge/ae-20100821-aov/wav/a0351.wav: IT WAS MORE LIKE SUGAR. using datan.npz
    # voxforge/aaa-20150128-fak/wav/a0557.wav: THE LAST REFUGEE HAD PASSED.
    inputs=train[0][:4]
    labels=train[1][:4]
    seqs=((train[2][:4]-1)/8+1)*8
    #seqs=train[2][:8]
    #test_inputs=test[0]
    #test_labels=test[1]
    #test_seqs=test[2]##For loop tune hyper
    config=Config(max_time = max_time, timestep = max_chars + 2)
    #"""
    with tf.Graph().as_default():
        model = LASmodel(config)
        init = tf.global_variables_initializer()
        saver=tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            losses = model.fit(saver,sess, inputs[:3], labels[:3],seqs[:3])
            saver.save(sess, model.config.model_output)
    #"""
    with tf.Graph().as_default():
        model=LASmodel(config)
        saver = tf.train.Saver()
        with tf.Session() as session:
            init = tf.global_variables_initializer()
            session.run(init)
            new_saver = tf.train.import_meta_graph(model.config.model_output+'.meta')
            new_saver.restore(session, tf.train.latest_checkpoint('./'))
            # modified 03/15 GZ, also model.py:test_on_batch, pred_on_batch
            """
            start_time = time.time()
            loss= model.test_on_batch(session, test_inputs, test_labels, test_seqs)
            duration = time.time() - start_time
            print 'Test loss = {:.2f} ({:.3f} sec)'.format(loss, duration)
            # new stuff 03/15 GZ
            """
            start_time = time.time()
            pred, dec = model.predict_on_batch(session, inputs, labels, seqs)#test_inputs[:10], test_labels[:10], test_seqs[:10])
            duration = time.time() - start_time
            print 'Predictions below: ({:.3f} sec)'.format(duration)
            print pred[0].shape, pred.shape
            np.savez("samples", pred=pred)
            print dec[0]
            #"""

if __name__ == "__main__":
    test_LAS_model()

