import numpy as np
import tensorflow as tf
from random import randint

state_dim = 13;
action_dim = 4;

sess = tf.Session()

# Critic Network
class Critic:
	def __init__(self, input_layer=17, hidden_layer=100, output_layer=1):
		self.learning_rate = 0.01
		self.gamma = 0.99
		self.input_layer = input_layer
		self.hidden_layer = hidden_layer
		self.output_layer = output_layer
		self.critic_state_input = tf.placeholder(tf.float32, shape=[None, 13])
		self.critic_action_input = tf.placeholder(tf.float32, shape=[None, 4])
		self.critic_input = tf.concat((self.critic_state_input, self.critic_action_input), axis=1)
		self.critic_W1 = tf.Variable(tf.truncated_normal([self.input_layer, self.hidden_layer]), name="critic_W1")
		self.critic_b1 = tf.Variable(tf.truncated_normal([self.hidden_layer]), name='critic_b1')
		self.critic_W2 = tf.Variable(tf.truncated_normal([self.hidden_layer, output_layer]), name='critic_W2')
		self.critic_b2 = tf.Variable(tf.truncated_normal([self.output_layer]), name='critic_b2')

		self.critic_h1 = tf.nn.sigmoid(tf.add(tf.matmul(self.critic_input, self.critic_W1), self.critic_b1))
		self.critic_Q_value = tf.add(tf.matmul(self.critic_h1, self.critic_W2), self.critic_b2)

		# target
		self.target_input = tf.placeholder(tf.float32, shape=[None, self.input_layer])
		self.target_W1 = tf.Variable(tf.truncated_normal([self.input_layer, self.hidden_layer]), name="target_critic_W1")
		self.target_b1 = tf.Variable(tf.truncated_normal([self.hidden_layer]), name='target_critic_b1')
		self.target_W2 = tf.Variable(tf.truncated_normal([self.hidden_layer, output_layer]), name='target_critic_W2')
		self.target_b2 = tf.Variable(tf.truncated_normal([self.output_layer]), name='target_critic_b2')

		self.target_h1 = tf.nn.sigmoid(tf.add(tf.matmul(self.target_input, self.target_W1), self.target_b1))
		self.target_Q_value = tf.add(tf.matmul(self.target_h1,self.target_W2), self.target_b2)

		self.target_reward_input = tf.placeholder(tf.float32, shape=[None, 1])
		self.target_Q_values = self.target_reward_input + self.gamma * self.target_Q_value
		self.loss = tf.reduce_mean((self.critic_Q_value - self.target_Q_values)**2) # change the loss function
		self.optim = tf.train.AdamOptimizer().minimize(self.loss)
		
	def copy_params(self):
		# sess.run(tf.initialize_all_variables())
		# Copy params in target network params
		sess.run(self.target_W1.assign(self.critic_W1))
		sess.run(self.target_b1.assign(self.critic_b1))
		sess.run(self.target_W2.assign(self.critic_W2))
		sess.run(self.target_b2.assign(self.critic_b2))

	# def get_target_value(self, actor, batch):
	# 	target_action_for_new_state = actor.get_target_action(batch[:,18:31]) #passed new state
	# 	new_state_action = np.concatenate((batch[:,18:31], target_action_for_new_state),axis=1)
	# 	target_value = sess.run(self.target_Q_value, feed_dict={self.target_input:new_state_action})
	# 	reward = batch[:,17]
	# 	reward = reward.reshape(reward.size,1)
	# 	return reward + self.gamma * target_value

	# def get_critic_output(self, state_action):
	# 	Q_value = sess.run(self.critic_Q_value,feed_dict={self.critic_input:state_action})
	# 	return Q_value

	def update_critic_network(self, batch, target_action):		
		target_input = np.concatenate((batch[:,18:31], target_action), axis=1) #target_input.shape = (batch_size, 17)
		reward = batch[:,17]
		reward = reward.reshape(reward.size,1) #shape = (batch_size, 1)

		print("loss = ", sess.run(self.loss, feed_dict={self.critic_state_input:batch[:,0:13], self.critic_action_input:batch[:,13:17], self.target_input:target_input, self.target_reward_input:reward}))


		sess.run(self.optim, feed_dict={self.critic_state_input:batch[:,0:13],self.critic_action_input:batch[:,13:17], self.target_input:target_input, self.target_reward_input:reward})


# Actor Network
class Actor:
	def __init__(self, input_layer=13, hidden_layer=100, output_layer=4):
		self.input_layer = input_layer
		self.hidden_layer = hidden_layer
		self.output_layer = output_layer
		self.actor_input = tf.placeholder(tf.float32, shape=[None, self.input_layer])
		self.actor_W1 = tf.Variable(tf.truncated_normal([self.input_layer, self.hidden_layer]), name="actor_W1")
		self.actor_b1 = tf.Variable(tf.truncated_normal([self.hidden_layer]), name='actor_b1')
		self.actor_W2 = tf.Variable(tf.truncated_normal([self.hidden_layer, output_layer]), name='actor_W2')
		self.actor_b2 = tf.Variable(tf.truncated_normal([self.output_layer]), name='actor_b2')

		self.actor_h1 = tf.nn.sigmoid(tf.add(tf.matmul(self.actor_input, self.actor_W1), self.actor_b1))
		self.actor_action = tf.add(tf.matmul(self.actor_h1, self.actor_W2), self.actor_b2)

		# target
		self.target_input = tf.placeholder(tf.float32, shape=[None, self.input_layer])
		self.target_W1 = tf.Variable(tf.truncated_normal([self.input_layer, self.hidden_layer]), name="target_actor_W1")
		self.target_b1 = tf.Variable(tf.truncated_normal([self.hidden_layer]), name='target_actor_b1')
		self.target_W2 = tf.Variable(tf.truncated_normal([self.hidden_layer, output_layer]), name='target_actor_W2')
		self.target_b2 = tf.Variable(tf.truncated_normal([self.output_layer]), name='target_actor_b2')

		self.target_h1 = tf.nn.sigmoid(tf.add(tf.matmul(self.target_input, self.target_W1), self.target_b1))
		self.target_action = tf.add(tf.matmul(self.target_h1,self.target_W2), self.target_b2)

	def copy_params(self):
		# sess.run(tf.initialize_all_variables())
		# Copy params in target network params
		sess.run(self.target_W1.assign(self.actor_W1))
		sess.run(self.target_b1.assign(self.actor_b1))
		sess.run(self.target_W2.assign(self.actor_W2))
		sess.run(self.target_b2.assign(self.actor_b2))

	def select_action(self, state):
		action = sess.run(self.actor_action,feed_dict={self.actor_input:state})
		return action

	def get_target_action(self, state):
		tar_action = sess.run(self.target_action, feed_dict={self.target_input: state})
		# print("target action = \n", tar_action)
		return tar_action

	def update_actor_network(self, critic, batch):
		actor_critic_input = tf.concat((critic.critic_state_input, self.actor_action),axis=1)

		critic.actor_critic_h1 = tf.nn.sigmoid(tf.add(tf.matmul(actor_critic_input, critic.critic_W1), critic.critic_b1))
		critic.actor_critic_Q_value = tf.add(tf.matmul(critic.actor_critic_h1, critic.critic_W2), critic.critic_b2)

		J = tf.reduce_mean(critic.actor_critic_Q_value)
		print("J = ", sess.run(J, feed_dict={self.actor_input:batch[:,0:13], critic.critic_state_input:batch[:,0:13]}))

		actor_optimizer = tf.train.AdamOptimizer().minimize(-1*J, var_list=[self.actor_W1, self.actor_b1, self.actor_W2, self.actor_b2])
		sess.run(actor_optimizer, feed_dict={self.actor_input:batch[:,0:13], critic.critic_state_input:batch[:,0:13]})








class Replay:
	def __init__(self):
		self.replay_buffer = None
		self.max_replay_size = 10000
		self.batch_size = 16

	def store_transition(self, old_state, action, reward, new_state):
		tmp = np.concatenate((old_state,action))
		tmp = np.concatenate((tmp, reward))
		tmp = np.concatenate((tmp,new_state))
		tmp = tmp.reshape(1,tmp.shape[0])
		print("storing tmp of shape: ", tmp.shape)
		self.replay_buffer = tmp if self.replay_buffer is None else np.append(self.replay_buffer, tmp, axis=0)
		# np.append(self.replay_buffer, tmp)
		# Finite size replay_buffer
		if self.replay_buffer.shape[0]>self.max_replay_size:
			self.replay_buffer = self.replay_buffer[1000:]

	def select_random_batch(self):
		tmp = [randint(0,self.replay_buffer.shape[0]-1) for p in range(self.batch_size)]
		random_batch = self.replay_buffer[tmp]
		return random_batch







critic = Critic(state_dim+action_dim,100,1)
actor = Actor(state_dim,100,4)
sess.run(tf.global_variables_initializer())
critic.copy_params()
actor.copy_params()

replay = Replay()

num_episodes = 10
num_time_steps = 5
for epoch in range(num_episodes):
	print("episode : ", epoch+1)
	# reset the environment and recieve initial state
	state = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1]) #1 X 13
	# state=state.reshape(1, 13)
	for t in range(num_time_steps):
		action = actor.select_action(state.reshape(1,13))
		action = action[0] #action was a matrix, now it is an np.array(4,)
		# execute this action and recieve new_state,reward
		new_state = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2])
		# new_state = new_state.reshape(1,13)
		reward = [3]
		reward = np.array(reward)
		replay.store_transition(state, action, reward, new_state)
		state = new_state 

		# select a random batch from replay_buffer
		if epoch > 5:
			batch = replay.select_random_batch()
			target_action = actor.get_target_action(batch[:,18:31]) # shape = (batch_size, 4)
			critic.update_critic_network(batch, target_action)



			# update actor_network
			actor.update_actor_network(critic, batch) 
		# batch = np.array(batch)

		# remove next two lines
		# Q_values = critic.get_critic_output(batch[:,0:17])
		# target_Q_values = critic.get_target_value(actor, batch)

		




# exec(open("DDPG.py").read())