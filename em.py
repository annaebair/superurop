from scipy.stats import norm, truncnorm
import scipy.stats
from scipy.special import erf
from math import exp, sqrt, pi
import matplotlib.pyplot as plt
import numpy as np
import sklearn.mixture
import mixem
from mixem.distribution.distribution import Distribution

np.set_printoptions(threshold=np.nan)


class Moments():

	def __init__(self, mu, sigma, a, b):
		self.mu = mu
		self.sigma = sigma
		self.a = a
		self.b = b
		self.first_moment = self._first_moment()
		self.second_moment = self._second_moment()

	def _first_moment(self):
		a_pdf = norm.pdf(self.a, loc=self.mu, scale=self.sigma)
		b_pdf = norm.pdf(self.b, loc=self.mu, scale=self.sigma)
		a_cdf = norm.cdf(self.a, loc=self.mu, scale=self.sigma)
		b_cdf = norm.cdf(self.b, loc=self.mu, scale=self.sigma)
		M1 = self.mu - self.sigma*(b_pdf-a_pdf)/(b_cdf-a_cdf)
		return M1

	def _second_moment(self):
		alpha, beta = (self.a - self.mu)/self.sigma, (self.b - self.mu)/self.sigma
		a_pdf = norm.pdf(self.a, loc=self.mu, scale=self.sigma)
		b_pdf = norm.pdf(self.b, loc=self.mu, scale=self.sigma)
		if self.a == -np.inf:
			a_pdf_der = 0
		else:
			a_pdf_der = -a_pdf*alpha
		if self.b == np.inf:
			b_pdf_der = 0
		else:
			b_pdf_der = -b_pdf*beta
		a_cdf = norm.cdf(self.a, loc=self.mu, scale=self.sigma)
		b_cdf = norm.cdf(self.b, loc=self.mu, scale=self.sigma)
		derivative_term = (b_pdf_der - a_pdf_der)/(b_cdf-a_cdf)
		M2 = self.sigma**2 + self.mu**2 + self.sigma**2 * derivative_term - 2*self.mu*self.sigma*((b_pdf - a_pdf)/(b_cdf - a_cdf))
		return M2


class TruncatedNormalDistribution(Distribution):
	"""Truncated normal distribution with parameters (mu, sigma)."""

	def __init__(self, mu, sigma, lower, upper):
		self.mu = mu
		self.sigma = sigma
		self.a = lower
		self.b = upper
		self.ss_dist = truncnorm(a=(self.a-self.mu)/self.sigma, b=(self.b-self.mu)/self.sigma, \
			loc=self.mu, scale=self.sigma)

	def log_density(self, data):
		assert(len(data.shape) == 1), "Expect 1D data!"
		lower_cdf = norm.cdf(self.a, loc=self.mu, scale=self.sigma)
		upper_cdf = norm.cdf(self.b, loc=self.mu, scale=self.sigma)
		log_density = - (data - self.mu) ** 2 / (2 * self.sigma ** 2) - np.log(self.sigma) \
			- 0.5 * np.log(2 * np.pi) - np.log(upper_cdf - lower_cdf)
		log_density = np.where(np.logical_or(data < self.a, data > self.b) , -9999, log_density)
		return log_density

	def estimate_parameters(self, data, weights):
		assert(len(data.shape) == 1), "Expect 1D data!"
		wsum = np.sum(weights)
		moments = Moments(0, self.sigma, self.a-self.mu, self.b-self.mu)
		m_k = moments.first_moment
		H_k = self.sigma - moments.second_moment
		self.mu = np.sum(weights * data) / wsum - m_k
		self.sigma = np.sqrt(np.sum(weights * (data - self.mu) ** 2) / wsum + H_k)

	def __repr__(self):
		return "TruncNorm[μ={mu:.4g}, σ={sigma:.4g}]".format(mu=self.mu, sigma=self.sigma)


class CensoredNormalDistribution(Distribution):
	"""Censored normal distribution with parameters (mu, sigma)."""

	def __init__(self, mu, sigma, lower, upper):
		self.mu = mu
		self.sigma = sigma
		self.a = lower
		self.b = upper
		self.norm_dist = norm(loc=self.mu, scale=self.sigma)

	def log_density(self, data):
		assert(len(data.shape) == 1), "Expect 1D data!"
		log_density = - (data - self.mu) ** 2 / (2 * self.sigma ** 2) - np.log(self.sigma) - 0.5 * np.log(2 * np.pi)
		log_density = np.where(np.logical_or(data < self.a, data > self.b) , -9999, log_density)
		log_density = np.where(data == self.a, 50*self.norm_dist.cdf(self.a) , log_density)
		log_density = np.where(data == self.b, 50*(1-self.norm_dist.cdf(self.b)), log_density)
		return log_density

	def estimate_parameters(self, data, weights):
		assert(len(data.shape) == 1), "Expect 1D data!"
		wsum = np.sum(weights)
		left_trunc_moments = Moments(self.mu, self.sigma, -np.inf, self.a)
		right_trunc_moments = Moments(self.mu, self.sigma, self.b, np.inf)
		left_first_moment, left_second_moment = left_trunc_moments.first_moment, left_trunc_moments.second_moment
		right_first_moment, right_second_moment = right_trunc_moments.first_moment, right_trunc_moments.second_moment
		weights = np.array(weights)[:, np.newaxis]
		new_data1 = np.where(data == self.a, left_first_moment, data)[:, np.newaxis]
		new_data = np.where(new_data1 == self.b, right_first_moment, new_data1)
		self.mu = np.sum(np.multiply(new_data, weights)) / wsum
		S = (new_data - self.mu) ** 2
		left_R_term = left_second_moment - (left_first_moment ** 2)
		right_R_term = right_second_moment - (right_first_moment ** 2)
		left_indices = np.nonzero((new_data == left_first_moment))[0]
		for i in left_indices:
			S[i] += left_R_term
		right_indices = np.nonzero((new_data == right_first_moment))[0]
		for i in right_indices:
			S[i] += right_R_term
		self.sigma = np.sqrt(np.sum(S * weights) / wsum)

	def __repr__(self):
		return "CensoredNorm[μ={mu:.4g}, σ={sigma:.4g}]".format(mu=self.mu, sigma=self.sigma)
