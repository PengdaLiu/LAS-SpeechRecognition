import tensorflow as tf
from model import Model
import numpy as np
from utils.general_utils import get_minibatches
import load_data as ld
import time

EOS = 32

class Config:#Several +drop out
	n_classes = 33
	batch_size = 100
	#U = 16   
	n_mfcc = 60#Must be 4 times pblstm_hidden
	pblstm_hidden = 15
	n_features = 40
	aslstm_state =  100
	hidden_atten1 = 100
	hidden_atten2 = 100
	hidden_dist= 100
	context_final = 100
	model_output= "model_weights"
	#max_time = 128
	#timestep = 82
	n_epochs = 100
	lr = 1e-3 

	def __init__(self, timestep=82, max_time=128):
		self.timestep=timestep
		self.max_time=max_time
		self.U=max_time/8

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
		self.input_placeholder = tf.placeholder(tf.float32	, \
			shape=[None, self.config.max_time, self.config.n_features])
		self.labels_placeholder =tf.placeholder(tf.int32, shape=[None, self.config.timestep])
		self.seq_len_placeholder = tf.placeholder(tf.int32, shape=[None,])
		self.mask_placeholder=tf.placeholder(tf.bool,shape=[None,self.config.timestep])

	def create_feed_dict(self, inputs_batch, labels_batch=None,seq_batch=None,mask_batch=None):
		feed_dict={self.input_placeholder:inputs_batch, self.labels_placeholder:labels_batch,self.seq_len_placeholder:seq_batch,self.mask_placeholder : mask_batch}
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

		hidden_size=self.config.pblstm_hidden
		max_time=self.config.max_time
		fcell_l1 = tf.nn.rnn_cell.LSTMCell(hidden_size)
		bcell_l1 = tf.nn.rnn_cell.LSTMCell(hidden_size)

		fcell_l2 = tf.nn.rnn_cell.LSTMCell(hidden_size)
		bcell_l2 = tf.nn.rnn_cell.LSTMCell(hidden_size)

		fcell_l3 = tf.nn.rnn_cell.LSTMCell(hidden_size)
		bcell_l3 = tf.nn.rnn_cell.LSTMCell(hidden_size)

		outputs_l1, state_l1 = tf.nn.bidirectional_dynamic_rnn(fcell_l1, bcell_l1, self.input_placeholder, \
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
			if i > 0: tf.get_variable_scope().reuse_variables()
			r=np.random.binomial(1,0.9)
			r=tf.constant(r,dtype=tf.float32)
			r=0
			random_sample=tf.reshape(tf.nn.softmax(preds[i]),[-1,self.config.n_classes])*r+(1-r)*tf.one_hot(self.labels_placeholder[:,i], self.config.n_classes, axis=-1,dtype=tf.float32)
			concatenated=tf.concat(1,[random_sample,c]) 
			output, state = stacked_lstm(concatenated, state)
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
	
	def run_epoch(self, sess, inputs, labels,seqs,mask):
		"""Runs an epoch of training.

		Args:
			sess: tf.Session() object
			inputs: np.ndarray of shape (n_samples, n_features)
			labels: np.ndarray of shape (n_samples, n_classes)
		Returns:
			average_loss: scalar. Average minibatch loss of model on epoch.
		"""
		correct,n_minibatches, total_loss = 0.0,0.0,0.0
		for input_batch, labels_batch ,seq_batch ,mask_batch in get_minibatches([inputs, labels,seqs,mask], self.config.batch_size):
			n_minibatches += 1
			batchcorrect,batchloss= self.train_on_batch(sess, input_batch, labels_batch,seq_batch,mask_batch)
			total_loss+=batchloss
			correct+=batchcorrect
		return total_loss / n_minibatches, correct/n_minibatches




	def run_test(self,sess,inputs,labels,seqs,mask):
		return self.test_model(sess,inputs,labels,seqs,mask)


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

def  test_LAS_model():
	train, test, max_time, max_chars = ld.load_data("../Data/data_nml.npz", new_format=False)
	inputs=train[0]
	labels=train[1]
	mask_train=to_mask(labels,EOS)
	seqs=train[2]
	test_inputs=test[0]
	test_labels=test[1]
	test_seqs=test[2]##For loop tune hyper
	mask_test=to_mask(test_labels,EOS)
	config=Config(max_time = max_time, timestep = max_chars + 2)
	with tf.Graph().as_default():
		model = LASmodel(config)
		init = tf.global_variables_initializer()
		saver=tf.train.Saver()
		with tf.Session() as sess:
			sess.run(init)
			losses = model.fit(saver,sess, inputs, labels,seqs,mask_train)
			saver.save(sess, model.config.model_output)
	with tf.Graph().as_default():
		model=LASmodel(config)
		saver = tf.train.Saver()
		with tf.Session() as session:
			init = tf.global_variables_initializer()
			session.run(init)
			new_saver = tf.train.import_meta_graph(model.config.model_output+'.meta')
			new_saver.restore(session, tf.train.latest_checkpoint('./'))
			loss= model.run_test(session,test_inputs,test_labels,test_seqs,mask_test)
	 		print "Test loss:"
	 		print loss


if __name__ == "__main__":
	test_LAS_model()

