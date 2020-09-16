# RL-rare-experience-replay

In this project, I build new "data recycle" tools that can be used in conjunction with reinforcement learning algorithms.

# Motivation:
* An Online RL agent observes a stream of transitions and learns from each experience e = (s, a, r, s’) incrementally
* Once used for update, each experience is discarded → waste
* Experience Replay has recently received much attention in the literature as a way to 
	1) Efficiently recycle experience data
	2) Reduce correlation in training when using value function approximation (eg. DQN)

# Goal:
* Investigate how efficiently Experience Replay and its variants use data
* Develop simple new algorithms that can better take advantage of unusual and/or rare data

# Method:
* To better focus on the “data recycle” aspect rather than correlation reduction, we use a tabular RL algorithm
* In particular, we use Q-Learning with 𝜀-greedy exploration
* Step size in the Q-update starts big and is decreased over episode for faster convergence
* I use a simplified version of the Deep Sea problem for simulation.

# Files:
In this repo, I've uploaded:
* replay.py: The "data recycle" algorithms written in Python,
* poster.pdf: Poster that I presented in a reinforcement learning class at Stanford,
* report.pdf: Short report of the findings.

Codes that I used to plot and simulate the problem environment will be available upon request.
