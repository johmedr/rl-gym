import gym
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, Dropout, MaxPool2D, Flatten
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K

import tensorflow as tf

import random
from collections import deque

def channel_first(shape):
	if(len(shape) == 3): # i.e. image
		return (shape[2], shape[0], shape[1])
	else:
		return shape

class A3CAgent:
	def __init__(self, env, sess, scope):
		self.env = env 
		self.sess = sess
		self.scope = scope

		self.learning_rate = 0.01
		# self.epsilon = 1.0
		self.epsilon = .5

		# self.epsilon_decay = .995
		self.epsilon_decay = .85
		self.gamma = .95
		self.l2 = 1e-3
		self.memory = deque(maxlen=2000)

		if len(self.env.action_space.shape) == 1:
			self.action_shape = self.env.action_space.shape
		elif isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
			self.action_shape = tuple([self.env.action_space.n])

		if len(self.env.observation_space.shape) == 3:
			self.observation_shape = channel_first(self.env.observation_space.shape)
		else: 
			self.observation_shape = self.env.observation_space.shape

		self.actor_state_input, self.actor_model = self._create_actor_model()
		_, self.target_actor_model = self._create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_shape[0]])

		self.actor_grads = tf.gradients(self.actor_model.output, self.actor_model.trainable_weights, - self.actor_critic_grad)
		grads = zip(self.actor_grads, self.actor_model.trainable_weights)

		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		self.critic_state_input, self.critic_action_input, self.critic_model = self._create_critic_model()
		_, _, self.target_critic_model = self._create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

		self.sess.run(tf.initialize_all_variables())


	def _create_actor_model(self):
		'''
		Input : State
		Output : Action
		'''
		state_in = Input(shape=self.observation_shape)


		if len(state_in.shape) == 2: 
			flat = Dense(units=64, activation='relu')(state_in)

		elif len(state_in.shape) == 4:
			conv1 = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', kernel_regularizer=l2(self.l2))(state_in)
			pool1 = MaxPool2D()(conv1)

			conv2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', kernel_regularizer=l2(self.l2))(pool1)
			dropout = Dropout(0.1)(conv2)
			pool = MaxPool2D()(dropout)
			dropout = Dropout(0.1)(conv2)
			pool = MaxPool2D()(dropout)

			flat = Flatten()(pool)

		h1 = Dense(units=32, activation='relu', kernel_regularizer=l2(self.l2))(flat)
		h2 = Dense(units=64, activation='relu', kernel_regularizer=l2(self.l2))(h1)

		action_out = Dense(units=self.action_shape[0], activation='softmax')(h2)

		model = Model(inputs=state_in, outputs=action_out)

		if self.action_shape[0] == 1: 
			model.compile(loss="mse", optimizer=Adam(self.learning_rate))
		else: 
			model.compile(loss="categorical_crossentropy", optimizer=Adam(self.learning_rate))

		print("... [ Actor model ] ...")

		print(model.summary())

		return state_in, model


	def _create_critic_model(self):
		''' 
		Input : State, Action 
		Output : V(s|a), reward in time
		''' 
		state_in = Input(shape=self.observation_shape)


		if len(state_in.shape) == 2: 
			flat = Dense(units=64, activation='relu')(state_in)

		elif len(state_in.shape) == 4:

			conv1 = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', kernel_regularizer=l2(self.l2))(state_in)
			pool1 = MaxPool2D()(conv1)

			conv2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', kernel_regularizer=l2(self.l2))(pool1)
			dropout = Dropout(0.1)(conv2)
			pool = MaxPool2D()(dropout)

			flat = Flatten()(pool)

		h_state = Dense(units=64, activation='relu', kernel_regularizer=l2(self.l2))(flat)

		action_in = Input(shape=self.action_shape)

		h_action = Dense(units=64, activation='relu', kernel_regularizer=l2(self.l2))(action_in)

		merged = Add()([h_state, h_action])

		h1 = Dense(units=32, activation='relu', kernel_regularizer=l2(self.l2))(merged)
		h2 = Dense(units=64, activation='relu', kernel_regularizer=l2(self.l2))(h1)

		value_out = Dense(units=1, activation='relu')(h2)

		model = Model(inputs=[state_in, action_in], outputs=value_out)
		model.compile(loss="mse", optimizer=Adam(self.learning_rate))

		print("... [ Critic model ] ...")

		print(model.summary())

		return state_in, action_in, model

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def train(self):
		batch_size = 128
		loss = -1
		if len(self.memory) >= batch_size:
			rewards = []
			samples = random.sample(self.memory, batch_size)
			loss = self._train_critic(samples)
			self._train_actor(samples)
		return loss

	def _train_critic(self, samples):
		state_batch = []
		action_batch = []
		reward_batch = []
		for sample in samples: 
			cur_state, action, reward, new_state, done = sample
			action_value = 0

			if not done:
				target_action = self.target_actor_model.predict(new_state)
				future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]

				action_value = reward + self.gamma * future_reward

			# self.critic_model.fit([cur_state, target_action], [action_value], batch_size=1, verbose=True) 
			state_batch.append(cur_state)
			action_batch.append(action)
			reward_batch.append(action_value)

		state_batch = np.vstack(state_batch)
		action_batch = np.vstack(action_batch)
		reward_batch = np.vstack(reward_batch)

		return self.critic_model.fit([state_batch, action_batch], reward_batch, batch_size=len(state_batch), epochs=4, verbose=True) 

	def _train_actor(self, samples):
		for sample in samples: 
			cur_state, action, reward, new_state, done = sample

			predicted_action = self.actor_model.predict(cur_state)
			grads = self.sess.run(self.critic_grads, feed_dict={
					self.critic_state_input: cur_state, 
					self.critic_action_input: predicted_action
				})[0]


			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_state, 
				self.actor_critic_grad: grads
			})

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	def _update_actor_target(self):
		self.target_actor_model.set_weights(self.actor_model.get_weights())

	def _update_critic_target(self):
		self.target_critic_model.set_weights(self.critic_model.get_weights())

	def act(self, cur_state): 
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		else: 
			if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
				return np.argmax(self.actor_model.predict(cur_state))
			else:
				return self.actor_model.predict(cur_state)

	def save(self, filename):
		self.target_actor_model.save_weights("ta" + filename, overwrite=True)
		self.target_critic_model.save_weights("tc" + filename, overwrite=True)
		self.actor_model.save_weights("a" + filename, overwrite=True)
		self.critic_model.save_weights("c" + filename, overwrite=True)

	def load(self, filename):
		try:
			self.target_actor_model.load_weights("ta" + filename)
			self.target_critic_model.load_weights("tc" + filename)
			self.actor_model.load_weights("a" + filename)
			self.critic_model.load_weights("c" + filename)
		except Exception as e:
			pass




def format_state(env, actor_critic, state): 
	if len(env.observation_space.shape) <= 2:
		state = state.reshape((1, actor_critic.observation_shape[0]))
	elif len(env.observation_space.shape) == 3:
		state = state.reshape((1, 
					actor_critic.observation_shape[0], actor_critic.observation_shape[1], actor_critic.observation_shape[2]))

	return state

if __name__=="__main__":


	sess = tf.Session()
	K.set_session(sess)

	# ENV_NAME = 'Breakout-v0'
	# ENV_NAME = 'KungFuMaster-v0'
	# ENV_NAME = 'BipedalWalker-v2'
	# ENV_NAME = 'CartPole-v1'
	# ENV_NAME = 'Acrobot-v1'
	# ENV_NAME = 'JourneyEscape-v0'
	ENV_NAME = 'SpaceInvaders-v0'

	# env = gym.make('Assault-v0')
	# env = gym.make('Boxing-v0')
	env = gym.make(ENV_NAME)

	FILENAME = 'weights' + ENV_NAME


	actor_critic = A3CAgent(env, sess, None)
	actor_critic.load(FILENAME)
	
	num_trials = 0
	trial_len  = 500

	render = True


	while True:
		num_trials += 1

		cur_state = env.reset()
		action = env.action_space.sample()

		one_hot_action = action
		if isinstance(env.action_space, gym.spaces.discrete.Discrete):
			one_hot_action = np.array((1, actor_critic.action_shape[0]))

		done = False

		score = 0

		frame = 0
		loss = 0

		while not done:
			frame += 1

			if render:
				env.render()
			cur_state = format_state(env, actor_critic, cur_state)

			action = actor_critic.act(cur_state)

			if len(env.action_space.shape) == 1:
				action = action.reshape((1, env.action_space.shape[0]))
			elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
				one_hot_action = np.zeros((1, actor_critic.action_shape[0]))
				one_hot_action[0][action] = 1
			
			new_state, reward, done, _ = env.step(action)
			
			new_state = format_state(env, actor_critic, cur_state)
			if len(env.action_space.shape) == 1:
				actor_critic.remember(cur_state, action, reward, new_state, done)
			elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
				actor_critic.remember(cur_state, one_hot_action, reward, new_state, done)

			# if frame %10 == 0:
			# 	loss_history = actor_critic.train()
			# 	if isinstance(loss_history, int):  
			# 		loss += loss_history
			# 	else: 
			# 		loss += loss_history.history['loss'][-1]
			
			cur_state = new_state

			score += reward
		
		loss_history = actor_critic.train()
		actor_critic.update_target()

		print('[ %d ] %f - Loss : %f'%(num_trials,score,loss/frame))

		render = True
		if num_trials % 10 == 0: 
			render = True
		if num_trials % 50 == 0: 
			actor_critic.save(FILENAME)


