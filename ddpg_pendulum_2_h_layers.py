import numpy as np
import tensorflow as tf
from random import randint
import gym
state_dim = 3;
action_dim = 1;

sess = tf.Session()

# Critic Network
class Critic:
	def __init__(self, input_layer=4, hidden_layer1=100, hidden_layer2=100, output_layer=1):
		# self.learning_rate = 0.01
		self.tau = 0.001
		self.gamma = 0.99
		self.input_layer = input_layer
		self.hidden_layer1 = hidden_layer1
		self.hidden_layer2 = hidden_layer2
		self.output_layer = output_layer
		self.critic_state_input = tf.placeholder(tf.float32, shape=[None, 3])
		self.critic_action_input = tf.placeholder(tf.float32, shape=[None, 1])
		self.critic_input = tf.concat((self.critic_state_input, self.critic_action_input), axis=1)
		self.critic_W1 = tf.Variable(tf.truncated_normal([self.input_layer, self.hidden_layer1]), name="critic_W1")
		self.critic_b1 = tf.Variable(tf.truncated_normal([self.hidden_layer1]), name='critic_b1')
		self.critic_W2 = tf.Variable(tf.truncated_normal([self.hidden_layer1, hidden_layer2]), name='critic_W2')
		self.critic_b2 = tf.Variable(tf.truncated_normal([self.hidden_layer2]), name='critic_b2')
		self.critic_W3 = tf.Variable(tf.truncated_normal([self.hidden_layer2, self.output_layer]), name="critic_W3")
		self.critic_b3 = tf.Variable(tf.truncated_normal([self.output_layer]), name='critic_b3')

		self.critic_h1 = tf.nn.relu(tf.add(tf.matmul(self.critic_input, self.critic_W1), self.critic_b1))
		self.critic_h2 = tf.nn.relu(tf.add(tf.matmul(self.critic_h1, self.critic_W2), self.critic_b2))
		self.critic_Q_value = tf.add(tf.matmul(self.critic_h2, self.critic_W3), self.critic_b3)

		# target
		self.target_input = tf.placeholder(tf.float32, shape=[None, self.input_layer])
		self.target_W1 = tf.Variable(tf.truncated_normal([self.input_layer, self.hidden_layer1]), name="target_critic_W1")
		self.target_b1 = tf.Variable(tf.truncated_normal([self.hidden_layer1]), name='target_critic_b1')
		self.target_W2 = tf.Variable(tf.truncated_normal([self.hidden_layer1, self.hidden_layer2]), name="target_critic_W2")
		self.target_b2 = tf.Variable(tf.truncated_normal([self.hidden_layer2]), name='target_critic_b2')
		self.target_W3 = tf.Variable(tf.truncated_normal([self.hidden_layer2, output_layer]), name='target_critic_W3')
		self.target_b3 = tf.Variable(tf.truncated_normal([self.output_layer]), name='target_critic_b3')

		self.target_h1 = tf.nn.relu(tf.add(tf.matmul(self.target_input, self.target_W1), self.target_b1))
		self.target_h2 = tf.nn.relu(tf.add(tf.matmul(self.target_h1, self.target_W2), self.target_b2))
		self.target_Q_value = tf.add(tf.matmul(self.target_h2,self.target_W3), self.target_b3)

		self.target_reward_input = tf.placeholder(tf.float32, shape=[None, 1])
		self.target_Q_values = self.target_reward_input + self.gamma * self.target_Q_value
		self.loss = tf.reduce_mean((self.critic_Q_value - self.target_Q_values)**2) # change the loss function
		self.optim = tf.train.AdamOptimizer().minimize(self.loss)

		self.a = self.target_W1.assign(tf.add(tf.multiply(self.critic_W1,self.tau), tf.multiply(self.target_W1, (1.0-self.tau) )))
		self.b = self.target_b1.assign(tf.add(tf.multiply(self.critic_b1,self.tau), tf.multiply(self.target_b1, (1.0-self.tau) )))
		self.c = self.target_W2.assign(tf.add(tf.multiply(self.critic_W2,self.tau), tf.multiply(self.target_W2, (1.0-self.tau) )))
		self.d = self.target_b2.assign(tf.add(tf.multiply(self.critic_b2,self.tau), tf.multiply(self.target_b2, (1.0-self.tau) )))
		self.e = self.target_W3.assign(tf.add(tf.multiply(self.critic_W3,self.tau), tf.multiply(self.target_W3, (1.0-self.tau) )))
		self.f = self.target_b3.assign(tf.add(tf.multiply(self.critic_b3,self.tau), tf.multiply(self.target_b3, (1.0-self.tau) )))
		
	def copy_params(self):
		# sess.run(tf.initialize_all_variables())
		# Copy params in target network params
		sess.run(self.target_W1.assign(self.critic_W1))
		sess.run(self.target_b1.assign(self.critic_b1))
		sess.run(self.target_W2.assign(self.critic_W2))
		sess.run(self.target_b2.assign(self.critic_b2))
		sess.run(self.target_W3.assign(self.critic_W3))
		sess.run(self.target_b3.assign(self.critic_b3))

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
		target_input = np.concatenate((batch[:,5:8], target_action), axis=1) #target_input.shape = (batch_size, 17)
		reward = batch[:,4]
		reward = reward.reshape(reward.size,1) #shape = (batch_size, 1)

		# print("critic loss = ", sess.run(self.loss, feed_dict={self.critic_state_input:batch[:,0:3], self.critic_action_input:batch[:,3:4], self.target_input:target_input, self.target_reward_input:reward}), end=' ')
		sess.run(self.optim, feed_dict={self.critic_state_input:batch[:,0:3],self.critic_action_input:batch[:,3:4], self.target_input:target_input, self.target_reward_input:reward})

	def update_target_network(self, tau):
		self.tau = tau
		# self.a = self.target_W1.assign(tf.add(tf.multiply(self.critic_W1,tau), tf.multiply(self.target_W1, (1.0-tau) )))
		# self.b = self.target_b1.assign(tf.add(tf.multiply(self.critic_b1,tau), tf.multiply(self.target_b1, (1.0-tau) )))
		# self.c = self.target_W2.assign(tf.add(tf.multiply(self.critic_W2,tau), tf.multiply(self.target_W2, (1.0-tau) )))
		# self.d = self.target_b2.assign(tf.add(tf.multiply(self.critic_b2,tau), tf.multiply(self.target_b2, (1.0-tau) )))
		sess.run([self.a,self.b,self.c,self.d,self.e,self.f])
		# sess.run(self.target_W1.assign(tau*self.critic_W1 + (1-tau)*self.target_W1))
		# sess.run(self.target_b1.assign(tau*self.critic_b1 + (1-tau)*self.target_b1))
		# sess.run(self.target_W2.assign(tau*self.critic_W2 + (1-tau)*self.target_W2))
		# sess.run(self.target_b2.assign(tau*self.critic_b2 + (1-tau)*self.target_b2))

# Actor Network
class Actor:
	def __init__(self, critic, input_layer=3, hidden_layer1=100, hidden_layer2=100, output_layer=1, action_range=2.0):
		self.tau = 0.001
		self.input_layer = input_layer
		self.hidden_layer1 = hidden_layer1
		self.hidden_layer2 = hidden_layer2
		self.output_layer = output_layer
		self.actor_input = tf.placeholder(tf.float32, shape=[None, self.input_layer])
		self.actor_W1 = tf.Variable(tf.truncated_normal([self.input_layer, self.hidden_layer1]), name="actor_W1")
		self.actor_b1 = tf.Variable(tf.truncated_normal([self.hidden_layer1]), name='actor_b1')
		self.actor_W2 = tf.Variable(tf.truncated_normal([self.hidden_layer1, hidden_layer2]), name='actor_W2')
		self.actor_b2 = tf.Variable(tf.truncated_normal([self.hidden_layer2]), name='actor_b2')
		self.actor_W3 = tf.Variable(tf.truncated_normal([self.hidden_layer2, self.output_layer]), name="actor_W3")
		self.actor_b3 = tf.Variable(tf.truncated_normal([self.output_layer]), name='actor_b3')

		self.actor_h1 = tf.nn.relu(tf.add(tf.matmul(self.actor_input, self.actor_W1), self.actor_b1))
		self.actor_h2 = tf.nn.relu(tf.add(tf.matmul(self.actor_h1, self.actor_W2), self.actor_b2))
		self.raw_actor_action = tf.nn.tanh(tf.add(tf.matmul(self.actor_h2, self.actor_W3), self.actor_b3))
		self.actor_action = tf.multiply(tf.cast(action_range, tf.float32), self.raw_actor_action)

		# target
		self.target_input = tf.placeholder(tf.float32, shape=[None, self.input_layer])
		self.target_W1 = tf.Variable(tf.truncated_normal([self.input_layer, self.hidden_layer1]), name="target_actor_W1")
		self.target_b1 = tf.Variable(tf.truncated_normal([self.hidden_layer1]), name='target_actor_b1')
		self.target_W2 = tf.Variable(tf.truncated_normal([self.hidden_layer1, hidden_layer2]), name='target_actor_W2')
		self.target_b2 = tf.Variable(tf.truncated_normal([self.hidden_layer2]), name='target_actor_b2')
		self.target_W3 = tf.Variable(tf.truncated_normal([self.hidden_layer2, output_layer]), name='target_actor_W3')
		self.target_b3 = tf.Variable(tf.truncated_normal([self.output_layer]), name='target_actor_b3')

		self.target_h1 = tf.nn.relu(tf.add(tf.matmul(self.target_input, self.target_W1), self.target_b1))
		self.target_h2 = tf.nn.relu(tf.add(tf.matmul(self.target_h1, self.target_W2), self.target_b2))
		self.raw_target_action = tf.nn.tanh(tf.add(tf.matmul(self.target_h2,self.target_W3), self.target_b3))
		self.target_action = tf.multiply(tf.cast(action_range, tf.float32), self.raw_target_action)


		self.actor_critic_input = tf.concat((critic.critic_state_input, self.actor_action),axis=1)
		self.actor_critic_h1 = tf.nn.relu(tf.add(tf.matmul(self.actor_critic_input, critic.critic_W1), critic.critic_b1))
		self.actor_critic_Q_value = tf.add(tf.matmul(self.actor_critic_h1, critic.critic_W2), critic.critic_b2)
		self.J = tf.reduce_mean(self.actor_critic_Q_value)
		self.actor_optimizer = tf.train.AdamOptimizer().minimize(-1*self.J, var_list=[self.actor_W1, self.actor_b1, self.actor_W2, self.actor_b2])
		
		self.a = self.target_W1.assign(tf.add(tf.multiply(self.actor_W1,self.tau), tf.multiply(self.target_W1, (1.0-self.tau) )))
		self.b = self.target_b1.assign(tf.add(tf.multiply(self.actor_b1,self.tau), tf.multiply(self.target_b1, (1.0-self.tau) )))
		self.c = self.target_W2.assign(tf.add(tf.multiply(self.actor_W2,self.tau), tf.multiply(self.target_W2, (1.0-self.tau) )))
		self.d = self.target_b2.assign(tf.add(tf.multiply(self.actor_b2,self.tau), tf.multiply(self.target_b2, (1.0-self.tau) )))
		self.e = self.target_W3.assign(tf.add(tf.multiply(self.actor_W3,self.tau), tf.multiply(self.target_W3, (1.0-self.tau) )))
		self.f = self.target_b3.assign(tf.add(tf.multiply(self.actor_b3,self.tau), tf.multiply(self.target_b3, (1.0-self.tau) )))
	
	def copy_params(self):
		sess.run(self.target_W1.assign(self.actor_W1))
		sess.run(self.target_b1.assign(self.actor_b1))
		sess.run(self.target_W2.assign(self.actor_W2))
		sess.run(self.target_b2.assign(self.actor_b2))
		sess.run(self.target_W3.assign(self.actor_W3))
		sess.run(self.target_b3.assign(self.actor_b3))

	def select_action(self, state):
		action = sess.run(self.actor_action,feed_dict={self.actor_input:state})
		return action

	def get_target_action(self, state):
		tar_action = sess.run(self.target_action, feed_dict={self.target_input: state})
		# print("target action = \n", tar_action)
		return tar_action

	def update_actor_network(self, critic, batch):
		# print("Expected reward value J = ", sess.run(self.J, feed_dict={self.actor_input:batch[:,0:3], critic.critic_state_input:batch[:,0:3]}), end=' ')
		sess.run(self.actor_optimizer, feed_dict={self.actor_input:batch[:,0:3], critic.critic_state_input:batch[:,0:3]})

	def update_target_network(self, tau):
		self.tau = tau
		# a = self.target_W1.assign(tf.add(tf.multiply(self.actor_W1,tau), tf.multiply(self.target_W1, (1.0-tau) )))
		# b = self.target_b1.assign(tf.add(tf.multiply(self.actor_b1,tau), tf.multiply(self.target_b1, (1.0-tau) )))
		# c = self.target_W2.assign(tf.add(tf.multiply(self.actor_W2,tau), tf.multiply(self.target_W2, (1.0-tau) )))
		# d = self.target_b2.assign(tf.add(tf.multiply(self.actor_b2,tau), tf.multiply(self.target_b2, (1.0-tau) )))
		sess.run([self.a,self.b,self.c,self.d,self.e,self.f])
		# sess.run(self.target_W1.assign(tau*self.actor_W1 + (1-tau)*self.target_W1))
		# sess.run(self.target_b1.assign(tau*self.actor_b1 + (1-tau)*self.target_b1))
		# sess.run(self.target_W2.assign(tau*self.actor_W2 + (1-tau)*self.target_W2))
		# sess.run(self.target_b2.assign(tau*self.actor_b2 + (1-tau)*self.target_b2))






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
		# print("storing tmp of shape: ", tmp.shape)
		self.replay_buffer = tmp if self.replay_buffer is None else np.append(self.replay_buffer, tmp, axis=0)
		
		if self.replay_buffer.shape[0]>self.max_replay_size:
			self.replay_buffer = self.replay_buffer[1000:]

	def select_random_batch(self):
		tmp = [randint(0,self.replay_buffer.shape[0]-1) for p in range(self.batch_size)]
		random_batch = self.replay_buffer[tmp]
		return random_batch




ENV_NAME = "Pendulum-v0"
env = gym.make(ENV_NAME)

action_range = env.action_space.high[0]


critic = Critic(state_dim+action_dim,400,10,1)
actor = Actor(critic,state_dim,400, 10, 1, action_range)
sess.run(tf.global_variables_initializer())
critic.copy_params()
actor.copy_params()

replay = Replay()



# exit(-1)
num_episodes = 1000
num_time_steps = 200
for epoch in range(num_episodes):
	print(epoch+1, end='  ')
	
	state = env.reset()
	reward_per_training_episode = 0

	for t in range(num_time_steps):
		action = actor.select_action(state.reshape(1,3))
		action = action[0] #action was a matrix, now it is an np.array(4,)
		# print("taking action : ", action, end=' ')
		################ action step ########################
		new_state, reward, done, info = env.step(action)
		# print("recieved reward= ",reward, end=' ')
		reward_per_training_episode += reward

		reward = np.array([reward])
		
		
		replay.store_transition(state, action, reward, new_state)
		state = new_state 

		
		if epoch > 1:
			# print("timestep: ", t)
			batch = replay.select_random_batch()
			target_action = actor.get_target_action(batch[:,5:8]) # shape = (batch_size, 4)
			critic.update_critic_network(batch, target_action)

			# update actor_network
			actor.update_actor_network(critic, batch) 

			critic.update_target_network(0.001)
			actor.update_target_network(0.001)

		if done:
			# print("\n *******************episode complete*************************")
			print("timesteps taken: ", t+1, "total reward: ",reward_per_training_episode )
			break
	# print("\n")
		
		




# exec(open("DDPG.py").read())