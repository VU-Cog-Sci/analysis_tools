import numpy as np

class TD(object): 
	""" Learn state values using Temporal Difference learning  """

	def __init__(self, nstates=2.0, alpha=0.2, gamma=0.9, ld = 0, init_val=0.0): 

		self.V = np.ones(nstates) * init_val 	#initialize the value of the states arbitrarily  
		self.e = np.zeros(nstates)
		self.nstates = int(nstates) 					#amount of states to track 
		self.alpha = alpha  					#learning rate 
		self.gamma = gamma						#discounting factor 
		self.ld = ld # lambda

	
	def value(self, state): 
		""" update the value function of the state """

		return self.V[state]					#select the value function that should be updated


	def delta(self, pstate, reward, state):
		"""
		This is the core error calculation. Note that if the value
		function is perfectly accurate then this returns zero since by
		definition value(pstate) = gamma * value(state) + reward.
		"""
		return reward + (self.gamma * self.value(state)) - self.value(pstate)

	def train(self, pstate, reward, state):
		"""
		A single step of reinforcement learning.
		"""

		delta = self.delta(pstate, reward, state)

		self.e[pstate] += 1.0

		#for s in range(self.nstates):
		self.V += self.alpha * delta * self.e
		self.e *= (self.gamma * self.ld)

		return delta

class TD_model(object):
	def __init__(self, states, rewards ):
		self.states = np.array(states, dtype = int)
		self.rewards = np.array(rewards, dtype = int)

	def simulate_run(self, params, V0 = 0.0):
		init_val = V0
		td = TD(nstates=2, alpha=params['alpha'], gamma=params['gamma'], ld = params['ld'], init_val=init_val)
		values = np.ones((self.states.shape[0], 2)) * init_val
		d_values = np.zeros((self.states.shape[0], 2))
		for i in range(1,len(self.rewards)):
			td.train(self.states[i-1], self.rewards[i], self.states[i])
			values[i] = [td.value(0),td.value(1)]
			d_values[i] = [values[i-1,0]-values[i,0], values[i-1,1]-values[i,1]]
		self.values = values
		self.d_values = d_values
		return {'values':values, 'd_values':d_values }

