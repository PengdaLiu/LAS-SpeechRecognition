import tensorflow as tf
from model_dnn import Model
import numpy as np
from utils.general_utils import get_minibatches
import load_data as ld
import time
EOS=28
num_train=7
num_dev=2
DEBUG = False
all_data_prefix = "/mnt/Data/data_resized_8000_b"
all_data_suffix = ".npz"
dbg_data_prefix = "/mnt/Data/data_debug_5000_b"
dbg_data_suffix = "_nml.npz"
dbg_train = [dbg_data_prefix+str(i)+dbg_data_suffix for i in range(num_train)]
fll_train=["/mnt/Data/data_resized_8000_b0.npz","/mnt/Data/data_resized_8000_b1.npz","/mnt/Data/data_resized_8000_b2.npz", \
	"/mnt/Data/data_resized_8000_b3.npz","/mnt/Data/data_resized_8000_b4.npz","/mnt/Data/data_resized_8000_b5.npz","/mnt/Data/data_resized_8000_b6.npz"]
dbg_dev = [dbg_data_prefix+str(i)+dbg_data_suffix for i in range(num_train, num_train+num_dev)]
fll_dev=["/mnt/Data/data_resized_8000_b7.npz","/mnt/Data/data_resized_8000_b8.npz"]
all_train = dbg_train if DEBUG else fll_train
all_dev = dbg_dev if DEBUG else fll_dev
TEST=(dbg_data_prefix + str(9) + dbg_data_suffix) if DEBUG else "/mnt/Data/data_resized_8000_b9.npz"

class Config:#Several +drop out
	n_classes = 29
	batch_size = 400
	#U = 16   
	n_mfcc = 80#Must be 4 times pblstm_hidden
	pblstm_hidden = 20
	n_features = 40
	dropout = 0.9
	p_use_last_pred = 0.1
	aslstm_state =  300
	hidden_atten1 = 500
	hidden_atten2 = 100
	hidden_dist= 100
	dnn_hidden1 = 80
	dnn_hidden2 = 80
	context_final = 200
	model_output= "model_weights"
	#max_time = 128
	#timestep = 82
	n_epochs = 500
	blocksz = 10
	lr = 1e-3 

	def set_p_use_last_pred(self, p=0.1):
		self.p_use_last_pred = p

	def __init__(self, timestep=82, max_time=128):
		self.timestep = timestep
		self.max_time = max_time
		self.blockdim = self.n_features * self.blocksz
		self.pblstm_nfeatures = self.blockdim / 2
		self.nblocks = max_time / self.blocksz
		self.U = self.nblocks / 8

class LASmodel(Model):
	def DNN(self, x, W1, W2, W3, b1, b2, b3):

	    """
	    x: batch_size * nblocks * blockdim ??? I'm guessing here
	    W1: blockdim * hidden1
	    W2: hidden1 * hidden2
	    W3: hidden2 * pblstm_nfeatures
	    y : batch_sisze * time * blockdim ??? Same above
	    """

	    h0 = tf.reshape(x, [-1, self.config.blockdim])
	    h1 = tf.nn.relu(tf.matmul(h0, W1) + b1)
	    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
	    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
	    y = tf.reshape(h3, [-1, self.config.nblocks, self.config.pblstm_nfeatures])
	    return y

	def set_p_use_last_pred(self, p):
		self.config.set_p_use_last_pred(p)
	
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
		self.input_placeholder = tf.placeholder(tf.float32	, \
			shape=[None, self.config.nblocks, self.config.blockdim])
		self.labels_placeholder =tf.placeholder(tf.int32, shape=[None, self.config.timestep])
		self.seq_len_placeholder = tf.placeholder(tf.int32, shape=[None,])
		self.mask_placeholder=tf.placeholder(tf.bool,shape=[None,self.config.timestep])
		self.dropout_placeholder =tf.placeholder(tf.float32)
		self.p_last_placeholder =tf.placeholder(tf.float32)

	def create_feed_dict(self, inputs_batch, labels_batch=None,seq_batch=None,mask_batch=None,dropout=1, p_last=0.1):
		feed_dict={self.input_placeholder:inputs_batch, self.labels_placeholder:labels_batch,self.seq_len_placeholder:seq_batch,self.mask_placeholder : mask_batch, self.dropout_placeholder:dropout, self.p_last_placeholder:p_last}
		return feed_dict

	def attentioncontext(self,si,h,w1,b1,w2,b2,w11,b11,w22,b22):
		sii = self.mlp(si,w1,b1,w2,b2)
		e = tf.reduce_sum(tf.mul(tf.reshape(sii,[-1,1,self.config.context_final]),self.mlp2(h,w11,b11,w22,b22)),axis=2)
		e = tf.nn.softmax(e)
		return tf.reduce_sum(tf.mul(tf.reshape(e,[-1,self.config.U,1]),h),axis=1)
	
	def add_prediction_op(self):
		preds=[]
		init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32	)
		first=tf.one_hot([0], self.config.n_classes) 
		first=tf.tile(first,[ 1,tf.shape(self.input_placeholder)[0] ])
		preds.append(tf.reshape(first,[-1,self.config.n_classes]))
		##Initialize variable
		lstm = tf.nn.rnn_cell.BasicLSTMCell(self.config.aslstm_state, state_is_tuple=True)
		stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2,state_is_tuple=True)
		initial_state = state = stacked_lstm.zero_state(tf.shape(self.input_placeholder)[0], tf.float32)		#Variable initialization
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
		w1_dnn = tf.Variable(init([self.config.blockdim, self.config.dnn_hidden1]))
		b1_dnn = tf.Variable(tf.zeros([ self.config.dnn_hidden1, ]))
		w2_dnn = tf.Variable(init([self.config.dnn_hidden1, self.config.dnn_hidden2]))
		b2_dnn = tf.Variable(tf.zeros([ self.config.dnn_hidden2, ]))
		w3_dnn = tf.Variable(init([self.config.dnn_hidden2, self.config.pblstm_nfeatures]))
		b3_dnn = tf.Variable(tf.zeros([ self.config.pblstm_nfeatures, ]))

		hidden_size=self.config.pblstm_hidden
		max_time=self.config.nblocks
		fcell_l1 = tf.nn.rnn_cell.LSTMCell(hidden_size)
		bcell_l1 = tf.nn.rnn_cell.LSTMCell(hidden_size)

		fcell_l2 = tf.nn.rnn_cell.LSTMCell(hidden_size)
		bcell_l2 = tf.nn.rnn_cell.LSTMCell(hidden_size)

		fcell_l3 = tf.nn.rnn_cell.LSTMCell(hidden_size)
		bcell_l3 = tf.nn.rnn_cell.LSTMCell(hidden_size)

		transformed_input = self.DNN(self.input_placeholder, w1_dnn, w2_dnn, w3_dnn, b1_dnn, b2_dnn, b3_dnn)

		outputs_l1, state_l1 = tf.nn.bidirectional_dynamic_rnn(fcell_l1, bcell_l1, transformed_input, \
			dtype=tf.float32	, sequence_length=self.seq_len_placeholder, scope="L1")
   	# concat outputs_l1
		outputs_l1 = tf.reshape(tf.concat(2, outputs_l1), [-1, max_time / 2, hidden_size * 4])

		outputs_l2, state_l2 = tf.nn.bidirectional_dynamic_rnn(fcell_l2, bcell_l2, outputs_l1, \
			dtype=tf.float32	, sequence_length=self.seq_len_placeholder / 2, scope="L2")
	# concat outputs_l2
		outputs_l2 = tf.reshape(tf.concat(2, outputs_l2), [-1, max_time / 4, hidden_size * 4])

		outputs_l3, state_l3 = tf.nn.bidirectional_dynamic_rnn(fcell_l3, bcell_l3, outputs_l2, \
			dtype=tf.float32	, sequence_length=self.seq_len_placeholder / 4, scope="L3")
		# concat'ed outputs_l3
		h= tf.reshape(tf.concat(2, outputs_l3), [-1, max_time / 8, hidden_size * 4])
		initial_s = output = tf.zeros([tf.shape(self.input_placeholder)[0],self.config.aslstm_state], tf.float32)
		c=self.attentioncontext(output,h,w1_atten,b1_atten,w2_atten,b2_atten,w11_atten,b11_atten,w22_atten,b22_atten)		
		for i in range(self.config.timestep-1):
			if i>0:
				tf.get_variable_scope().reuse_variables()
            #r=np.random.binomial(1, float(self.p_last_placeholder))
			r=tf.constant(np.random.binomial(1, self.config.p_use_last_pred), tf.float32)
            #r=tf.constant(r,dtype=tf.float32)
			random_sample=tf.reshape(tf.nn.softmax(preds[i]),[-1,self.config.n_classes])*r+(1-r)*tf.one_hot(self.labels_placeholder[:,i], self.config.n_classes, axis=-1,dtype=tf.float32)
			concatenated=tf.concat(1,[random_sample,c]) 
			output, state = stacked_lstm(concatenated, state)
			output=tf.nn.dropout(output,self.dropout_placeholder)
			c = self.attentioncontext(output, h, w1_atten,b1_atten,w2_atten,b2_atten,w11_atten,b11_atten,w22_atten,b22_atten)
			ith_pred=tf.reshape(self.chadist(output,c,w1_dist,b1_dist,w2_dist,b2_dist),[-1,self.config.n_classes])
			preds.append(ith_pred)
		preds = tf.pack(preds, 1)		
		return preds
	
	def add_correct(self,preds):
		predictions =tf.argmax(preds, axis=2)
		num_correct=self.tf_count(tf.boolean_mask(tf.cast(predictions,tf.int32)-self.labels_placeholder,self.mask_placeholder),0);
		a=tf.ones(tf.shape(self.mask_placeholder),tf.float32)
		return num_correct/tf.reduce_sum(tf.boolean_mask(a,self.mask_placeholder));
	
	def tf_count(self,t, val):
		elements_equal_to_value = tf.equal(t, val)
		as_ints = tf.cast(elements_equal_to_value, tf.float32)
		count = tf.reduce_sum(as_ints)
		return count

	def add_loss_op(self, preds):
		loss=0.0
		'''
		for i in range(self.config.timestep):
			if i>0 :
				embedding=tf.one_hot(self.labels_placeholder[:,i], self.config.n_classes, axis=-1,dtype=tf.float32)
				a= tf.reduce_mean(tf.log(tf.mul(tf.reshape(preds[i],[-1,self.config.n_classes]),embedding)))
				loss+=a
		return -loss
		'''
		loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=preds)
		loss = tf.reduce_mean(tf.boolean_mask(loss,self.mask_placeholder))
		return loss

	def add_training_op(self, loss):
		#opt=tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)
		opt = tf.train.AdamOptimizer(self.config.lr)
		train_op=opt.minimize(loss,)
		return train_op
		
	def __init__(self, config):
		self.config=config
		self.build()
	
	def add_softmax_pred_op(self, pred):
		return tf.nn.softmax(pred)
	
	def run_epoch(self, sess, inputs, labels,seqs,mask):
		"""Runs an epoch of training.

		Args:
			sess: tf.Session() object
			inputs: np.ndarray of shape (n_samples, n_features)
			labels: np.ndarray of shape (n_samples, n_classes)
		Returns:
			average_loss: scalar. Average minibatch loss of model on epoch.
		"""
		n_minibatches, total_loss = 0.0,0.0
		for input_batch, labels_batch ,seq_batch ,mask_batch in get_minibatches([inputs, labels,seqs,mask], self.config.batch_size):
			n_minibatches += 1
			batchloss= self.train_on_batch(sess, input_batch, labels_batch,seq_batch,mask_batch)
			total_loss+=batchloss
		return total_loss / n_minibatches

	def dev_model(self,sess,inputs,labels,seqs,mask):
		n_minibatches, total_loss = 0.0,0.0
		for input_batch, labels_batch ,seq_batch ,mask_batch in get_minibatches([inputs, labels,seqs,mask], self.config.batch_size):
			n_minibatches += 1
			batchloss= self.loss_on_batch(sess, input_batch, labels_batch,seq_batch,mask_batch)
			total_loss+=batchloss
		return total_loss / n_minibatches

	def test_model(self,sess,inputs,labels,seqs,mask):
		n_minibatches, total_loss,total_correct = 0.0,0.0,0.0
		pred = None
		lbls = None
		for input_batch, labels_batch ,seq_batch ,mask_batch in get_minibatches([inputs, labels,seqs,mask], self.config.batch_size):
			n_minibatches += 1
			lbls, pred,batchloss,batchcorrect= self.test_on_batch(sess, input_batch, labels_batch,seq_batch,mask_batch)
			total_loss+=batchloss
			total_correct+=batchcorrect
		return lbls, pred, total_loss / n_minibatches ,total_correct/n_minibatches
	

	def fit(self, saver,sess, inputs, labels,seqs,mask):
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
			start_time = time.time()
			average_loss,correct_rate = self.run_epoch(sess, inputs, labels,seqs,mask)
			duration = time.time() - start_time
			print 'Epoch {:}: loss = {:.2f} correct rate={:.2f} ({:.3f} sec)'.format(epoch, average_loss,correct_rate ,duration)
			losses.append(average_loss)
		return losses

def to_mask(data, thresh):
	return np.concatenate([np.array([[True]]*data.shape[0]),data[:,:-1]!=thresh],axis=1)
	"""
	m=np.shape(data)[0]
	n=np.shape(data)[1]
	mask=np.zeros(np.shape(data))
	for i in range (m):
		ind=True
		for j in range (n):
			mask[i][j]=ind
			if data[i][j] == thresh:
				ind=False
	return np.bool_(mask)
	"""

def  test_LAS_model():
	start,max_time,max_chars=ld.load_data("/mnt/Data/data_resized_8000_b0.npz")
	config=Config(max_time = max_time, timestep = max_chars + 2)
	all_train_loss=[]
	all_dev_loss=[]
	min_loss=0.0
	with tf.Graph().as_default():
		model=LASmodel(config)
		init=tf.global_variables_initializer()
		saver=tf.train.Saver()
		with tf.Session() as sess:
			print "[INFO] Building session..."
			sess.run(init)
			print "[INFO] Start to train..."
			for epoch in range (model.config.n_epochs):
				print "[INFO] Epoch", epoch, "started"
				start_time=time.time()
				train_loss=0.0
				dev_loss=0.0
				for i in range (num_train):
					print "[INFO] Loading data file", i
					ith, _ , foo=ld.load_data(all_train[i])
					inputs=ith[0].reshape((-1, config.nblocks, config.blockdim))
					labels=ith[1]
					seqs=np.ceil(ith[2] / float(config.blocksz)) * config.blocksz
					mask=to_mask(labels,EOS)
					print "[INFO] Training on data file", i
					train_loss+=model.run_epoch(sess, inputs, labels,seqs,mask)
				train_loss=train_loss/num_train	
				all_train_loss.append(train_loss)
				#For every epoch, evaluate on development set
				for i in range(num_dev):	
					print "[INFO] Loading dev file", i
					dev, _ , foo=ld.load_data(all_dev[i])
					dev_inputs=dev[0].reshape((-1, config.nblocks, config.blockdim))
					dev_labels=dev[1]
					dev_seqs=np.ceil(dev[2] / float(config.blocksz)) * config.blocksz
					mask_dev=to_mask(dev_labels,EOS)
					print "[INFO] Testing on dev file", i
					dev_loss+=model.dev_model(sess,dev_inputs,dev_labels,dev_seqs,mask_dev)
				dev_loss=dev_loss/num_dev		
				all_dev_loss.append(dev_loss)
				if epoch ==0:
					min_loss=dev_loss
					saver.save(sess,model.config.model_output)
				if epoch > 1:
					if dev_loss < min_loss:
						saver.save(sess,model.config.model_output)		
				duration=time.time()-start_time
				print 'Epoch {:}: train loss = {:.2f} dev loss={:.2f} ({:.3f} sec)'.format(epoch, train_loss,dev_loss ,duration)
		all_train_loss=np.array(all_train_loss)
		all_dev_loss=np.array(all_dev_loss)
		curve=np.zeros((2,config.n_epochs))
		curve[0]=all_train_loss
		curve[1]=all_dev_loss
		np.save("curve", curve, allow_pickle=True, fix_imports=True)

		##Evaluate on test set
		with tf.Session() as sess_test:
			print "[INFO] Loading test file"
			test,_,foo=ld.load_data(TEST)
			test_inputs=test[0].reshape((-1, config.nblocks, config.blockdim))
			test_labels=test[1]
			test_seqs=np.ceil(test[2] / float(config.blocksz)) * config.blocksz
			test_mask=to_mask(test_labels,EOS)
			init=tf.global_variables_initializer()
			sess_test.run(init)
			test_saver= tf.train.import_meta_graph(model.config.model_output+'.meta')
			test_saver.restore(sess_test, tf.train.latest_checkpoint('./'))			
			print "[INFO] Final testing..."
			test_truevals, test_pred, test_loss,test_correct=model.test_model(sess_test,test_inputs,test_labels,test_seqs,test_mask)
			print 'Test loss:{:.2f}, Test accuracy={:.3f}'.format(test_loss,test_correct)
			np.save("Testresult",np.array([test_loss,test_correct]),True,True)
			pred_file = "Test_pred_{:.4f}".format(time.time())
			np.savez(pred_file, pred=test_pred[:100], truevals = test_truevals[:100])
			print "[INFO] Prediction written to file", pred_file+".npz"

if __name__ == "__main__":
	test_LAS_model()
	print "[INFO] ALL DONE"
