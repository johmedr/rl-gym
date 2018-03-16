import gym
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, Dropout, MaxPool2D, Flatten
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque

class A3CAgent:
	def __init__(self, env, sess, scope):
		self.env = env 
		self.sess = sess
		self.scope = scope

		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = .995
		self.gamma = .95
		self.tau   = .125
		self.memory = deque(maxlen=2000)

		self.actor_state_input, self.actor_model = self._create_actor_model()
		_, self.target_actor_model = self._create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])

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
		state_in = Input(shape=self.env.observation_space.shape)


		if len(state_in.shape) == 2: 
			flat = Dense(units=64, activation='relu')(state_in)

		elif len(state_in.shape) == 4:

			conv1 = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(state_in)
			conv2 = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(conv1)
			dropout = Dropout(0.1)(conv2)
			pool = MaxPool2D()(dropout)

			flat = Flatten()(pool)

		h1 = Dense(units=32, activation='relu')(flat)
		h2 = Dense(units=64, activation='relu')(h1)

		action_out = Dense(units=self.env.action_space.shape[0])(h2)

		model = Model(inputs=state_in, outputs=action_out)
		model.compile(loss="mse", optimizer=Adam(self.learning_rate))

		print("... [ Actor model ] ...")

		print(model.summary())

		return state_in, model


	def _create_critic_model(self):
		''' 
		Input : State, Action 
		Output : V(s|a), reward in time
		''' 
		state_in = Input(shape=self.env.observation_space.shape)


		if len(state_in.shape) == 2: 
			flat = Dense(units=64, activation='relu')(state_in)

		elif len(state_in.shape) == 4:

			conv1 = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(state_in)
			conv2 = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(conv1)
			dropout = Dropout(0.1)(conv2)
			pool = MaxPool2D()(dropout)

			flat = Flatten()(pool)

		h_state = Dense(units=32, activation='relu')(flat)

		action_in = Input(shape=self.env.action_space.shape)

		h_action = Dense(units=32, activation='relu')(action_in)

		merged = Add()([h_state, h_action])

		h1 = Dense(units=32, activation='relu')(merged)
		h2 = Dense(units=64, activation='relu')(h1)

		value_out = Dense(units=1)(h2)

		model = Model(inputs=[state_in, action_in], outputs=value_out)
		model.compile(loss="mse", optimizer=Adam(self.learning_rate))

		print("... [ Critic model ] ...")

		print(model.summary())

		return state_in, action_in, model

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def train(self):
		batch_size = 32
		if len(self.memory) >= batch_size:
			rewards = []
			samples = random.sample(self.memory, batch_size)
			self._train_critic(samples)
			self._train_actor(samples)


	def _train_critic(self, samples):
		state_batch = []
		action_batch = []
		reward_batch = []
		for sample in samples: 
			cur_state, action, reward, new_state, done = sample

			if not done:
				target_action = self.target_actor_model.predict(new_state)
				future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]

				action_value = reward + self.gamma * future_reward
		
			state_batch.append(cur_state)
			action_batch.append(action)
			reward_batch.append(reward)

		state_batch = np.vstack(state_batch)
		action_batch = np.vstack(action_batch)
		reward_batch = np.vstack(reward_batch)

		self.critic_model.fit([state_batch, action_batch], reward_batch, batch_size=len(state_batch), verbose=False) 

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


	def update_actor_target(self):
		self.target_actor_model.set_weights(self.actor_model.get_weights())

	def update_critic_target(self):
		self.target_critic_model.set_weights(self.critic_model.get_weights())

	def act(self, cur_state): 
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		else: 
			return self.actor_model.predict(cur_state)


if __name__=="__main__":
	sess = tf.Session()
	K.set_session(sess)
	env = gym.make("Pendulum-v0")
	actor_critic = A3CAgent(env, sess, None)
	
	num_trials = 10000
	trial_len  = 500

	while True:
		cur_state = env.reset()
		action = env.action_space.sample()
		done = False

		score = 0

		while not done:
			env.render()
			cur_state = cur_state.reshape((1, 
				env.observation_space.shape[0]))
			action = actor_critic.act(cur_state)
			action = action.reshape((1, env.action_space.shape[0]))
			
			new_state, reward, done, _ = env.step(action)
			new_state = new_state.reshape((1, 
				env.observation_space.shape[0]))
			
			actor_critic.remember(cur_state, action, reward, 
				new_state, done)
			actor_critic.train()
			
			cur_state = new_state

			score += reward

		print(score)
