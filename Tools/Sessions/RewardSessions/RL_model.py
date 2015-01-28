import numpy as np


class RL_model(object):
	def __init__(self, stim_rewards_sample_points, stim_norewards_sample_points, fix_rewards_sample_points):
		self.stim_rewards_sample_points = stim_rewards_sample_points
		self.stim_norewards_sample_points = stim_norewards_sample_points
		self.fix_rewards_sample_points = fix_rewards_sample_points
		self.beta_indices = np.arange(len(stim_rewards_sample_points))[np.array([stim_rewards_sample_points, stim_norewards_sample_points, fix_rewards_sample_points]).sum(axis = 0, dtype = bool)]

	def simulate_run(self, params):
		time_scale_p = params['time_scale_p']; alpha_SR = params['alpha_SR']; alpha_SNR = params['alpha_SNR']; alpha_FR = params['alpha_FR'];
		input = np.sum(np.array([params['alpha_SR'] * self.stim_rewards_sample_points, params['alpha_FR'] * self.fix_rewards_sample_points, params['alpha_SNR'] * self.stim_norewards_sample_points]), axis = 0)
		Vs = np.zeros(self.stim_rewards_sample_points.shape)
		dVs = np.zeros(self.stim_rewards_sample_points.shape)
		V = 0
		for i in range(self.stim_rewards_sample_points.shape[0]):
			dV = input[i] - V
			V = V + params['time_scale_p'] * dV
			Vs[i] = V
			dVs[i] = dV
		self.Vs = Vs
		self.dVs = dVs
		return Vs, dVs			

	def simulate_results_for_fit(self, params, which_var, integration_window_length):
		self.simulate_run(params)
		this_simulation = [self.Vs, self.dVs][which_var]
		return np.array([this_simulation[bi:bi+integration_window_length] for bi in self.beta_indices]).sum(axis = 1)

