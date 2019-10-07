"""
Title: replay.py
Author: Jamie Kang (jamiekang@stanford.edu)

This Python script stores experiences (i.e. transition data) of reinforcement learning agent
it can learn from the experience more efficiently.

This script extends some of the existing methods (Experience Replay (ER) & Prioritized ER (PER))
to accommodate environments wehre some experiences can be very rare (i.e. sparse transition matrix).
"""

import numpy as np


class ExperienceReplay(object):
	"""
	Standard Experience Replay (uniform sample)
	"""
	def __init__(self, buffer_size, **kwargs):
		self.buffer_size = buffer_size
		self.buffer = [] ## every row is a (S,A,R,S_) tuple
		self.next_index = 0

	def __str__(self):
		return "ExperienceReplay"
	
	def store(self, state, action, reward, next_state, **kwargs):
		"""All non-adaptive ERs will inherit this"""
		experience = (state, action, reward, next_state)
		if self.next_index < len(self.buffer):
			self.buffer[self.next_index] = experience
		else:
			self.buffer.append(experience)
		self.next_index = (self.next_index + 1) % self.buffer_size

	def sample(self):
		""" Sample an experience """
		index = np.random.randint(len(self.buffer))
		return index

	def updatePriority(self, idx, TDerror):
		pass


class PrioritizedER(object):
	"""
	Prioritized Experience Replay (weights based on TD)
	"""
	def __init__(self, buffer_size, alpha, E, **kwargs):
		self.buffer_size = buffer_size
		self.buffer = [] ## every row is a (S,A,R,S_) tuple
		self.next_index = 0
		self.alpha = alpha
		self.E = E # to make every distribution non-zero
		self.priorities = [] 
		self.priority_sum = 0

	def __str__(self):
		return "PrioritizedER - TD Error"

	def store(self, state, action, reward, next_state, TDerror, **kwargs):
		experience = (state, action, reward, next_state)
		p = abs(TDerror) + self.E
		prob = p ** self.alpha
		if self.next_index < len(self.buffer): # If overwriting buffer
			self.buffer[self.next_index] = experience
			self.priority_sum += prob - self.priorities[self.next_index]
			self.priorities[self.next_index] = prob
		else:
			self.buffer.append(experience)
			self.priorities.append(prob)
			self.priority_sum += prob
		self.next_index = (self.next_index + 1) % self.buffer_size

	def updatePriority(self, idx, TDerror):	
		p = abs(TDerror) + self.E
		prob = p ** self.alpha
		self.priority_sum += prob - self.priorities[idx]
		self.priorities[idx] = prob

	def sample(self):
		pmf = [prior / self.priority_sum for prior in self.priorities]
		cmf = np.cumsum(pmf)
		idx = sum(cmf < np.random.random())
		return idx


class AsymmetricPrioritizedER(PrioritizedER):
	"""
	Asymmetric Prioritized Experience Replay

	prioritizes positive 'surprises' and negative 'surprises' differently
	"""
	def __init__(self, buffer_size, alpha, E, penalty, **kwargs):
		super(AsymmetricPrioritizedER, self).__init__(buffer_size, alpha, E)
		self.penalty = penalty # want this 0 < penalty < 1
	def __str__(self):
		return "AsymmPrioritizedER - TD Error"
	def store(self, state, action, reward, next_state, TDerror, **kwargs):
		if TDerror < 0:
			TDerror = self.penalty * TDerror
		super(AsymmetricPrioritizedER, self).store(state, action, reward, next_state, TDerror)
	def updatePriority(self, idx, TDerror):
		if TDerror < 0:
			TDerror = self.penalty * TDerror
		super(AsymmetricPrioritizedER, self).updatePriority(idx,TDerror)

class RarePrioritizedER(PrioritizedER):
	"""
	Rare Prioritized Experience Replay

	assigns additional weight to account for 'how rare' the transition is
	"""
	def __init__(self, buffer_size, alpha, E, num_state, num_action, **kwargs):
		super(RarePrioritizedER, self).__init__(buffer_size, alpha, E)
		self.F_mat = np.zeros((num_state,num_action))

	def store(self, state, action, reward, next_state, TDerror, **kwargs):
		super(RarePrioritizedER, self).store(state, action, reward, next_state, TDerror)
		self.F_mat[state,action] += 1

	def sample(self):
		self.proportion = []
		for x in self.buffer:
			state, action, _, _ = x
			self.proportion.append(1/(self.F_mat[state,action]**1))
		modified_prior = np.multiply(self.priorities, self.proportion)
		pmf = modified_prior / sum(modified_prior)
		cmf = np.cumsum(pmf)
		idx = sum(cmf < np.random.random())
		return idx



class ThresholdPrioritizedER(PrioritizedER):
	"""
	Threshold Prioritized Experience Replay

	only prioritizes transitions that have happened less than threshold 
	"""

	#TODO: keep track of number of sampling for each experience, and if it's > k, delete it
	#TODO: Also need to update next index so that new store can happen in the right place
	def __init__(self, buffer_size, alpha, E, num_state, num_action, threshold, **kwargs):
		super(ThresholdPrioritizedER, self).__init__(buffer_size, alpha, E)
		self.threshold = threshold
		self.U_mat = np.zeros((num_state,num_action))

	def sample(self):
		self.is_counted = []
		for x in self.buffer:
			state, action, _, _ = x
			self.is_counted.append(int(self.U_mat[state,action] < self.threshold))
		modified_prior = np.multiply(self.priorities, self.is_counted)
		if sum(modified_prior) == 0:
			pmf = self.priorities / sum(self.priorities)
		else:
			pmf = modified_prior / sum(modified_prior)
		cmf = np.cumsum(pmf)
		idx = sum(cmf < np.random.random())
		state, action, _, _ = self.buffer[idx]
		self.U_mat[state,action] += 1
		return idx

class SoftmaxPrioritizedER(PrioritizedER):
	def __init__(self, buffer_size, **kwargs):
		self.buffer_size = buffer_size
		self.buffer = [] ## every row is a (S,A,R,S_) tuple
		self.next_index = 0
		self.priorities = [] 
		self.priority_sum = 0	
	

	def store(self, state, action, reward, next_state, TDerror, **kwargs):
		experience = (state, action, reward, next_state)
		prob = np.exp(abs(TDerror))
		if self.next_index < len(self.buffer): # If overwriting buffer
			self.buffer[self.next_index] = experience
			self.priority_sum += prob - self.priorities[self.next_index]
			self.priorities[self.next_index] = prob
		else:
			self.buffer.append(experience)
			self.priorities.append(prob)
			self.priority_sum += prob
		self.next_index = (self.next_index + 1) % self.buffer_size

	def updatePriority(self, idx, TDerror):	
		prob = np.exp(abs(TDerror))
		self.priority_sum += prob - self.priorities[idx]
		self.priorities[idx] = prob
