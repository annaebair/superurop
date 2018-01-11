from scipy.stats import norm, truncnorm
import scipy.stats
from scipy.special import erf
from math import exp, sqrt, pi
import matplotlib.pyplot as plt
import numpy as np
import sklearn.mixture
import mixem
from mixem.distribution.distribution import Distribution


class Moments():

	def __init__(self, mu, sigma, a, b):
		self.mu = mu
		self.sigma = sigma
		self.a = a
		self.b = b
		self._zeroth_moment = self._zeroth_moment()
		self.first_moment = self._first_moment()
		self.second_moment = self._second_moment()

	# def _first_moment(self):
	# 	a_pdf = norm.pdf(self.a, loc=self.mu, scale=self.sigma)
	# 	b_pdf = norm.pdf(self.b, loc=self.mu, scale=self.sigma)
	# 	a_cdf = norm.cdf(self.a, loc=self.mu, scale=self.sigma)
	# 	b_cdf = norm.cdf(self.b, loc=self.mu, scale=self.sigma)

	# 	M1 = self.mu - self.sigma*(b_pdf-a_pdf)/(b_cdf-a_cdf)
	# 	return M1

	def _zeroth_moment(self):
		upper_erf = erf((self.b-self.mu)/(self.sigma*sqrt(2)))
		lower_erf = erf((self.a-self.mu)/(self.sigma*sqrt(2)))
		return 0.5*(upper_erf - lower_erf)

	# def _second_moment(self):
	# 	a_pdf = norm.pdf(self.a, loc=self.mu, scale=self.sigma)
	# 	b_pdf = norm.pdf(self.b, loc=self.mu, scale=self.sigma)
	# 	a_pdf_der = -a_pdf*self.a
	# 	b_pdf_der = -b_pdf*self.b
	# 	a_cdf = norm.cdf(self.a, loc=self.mu, scale=self.sigma)
	# 	b_cdf = norm.cdf(self.b, loc=self.mu, scale=self.sigma)

	# 	derivative_term = (b_pdf_der - a_pdf_der)/(b_cdf-a_cdf)

	# 	M2 = self.sigma**2 + self.mu**2 + self.sigma**2 * derivative_term - 2*self.mu*self.sigma*self.first_moment
	# 	return M2

	def _first_moment(self):
		lower_exp = exp(-((self.a-self.mu)/(self.sigma*sqrt(2)))**2)
		upper_exp = exp(-((self.b-self.mu)/(self.sigma*sqrt(2)))**2)
		return (self.sigma/sqrt(2*pi))*(lower_exp-upper_exp) + self.mu*self._zeroth_moment

	def _second_moment(self):
		lower_exp = exp(-((self.a-self.mu)/(self.sigma*sqrt(2)))**2)
		upper_exp = exp(-((self.b-self.mu)/(self.sigma*sqrt(2)))**2)
		return (self.sigma/sqrt(2*pi))*((self.a+self.mu)*lower_exp-(self.b+self.mu)*upper_exp) \
			+ (self.mu**2 + self.sigma**2)*self._zeroth_moment


class TruncatedNormalDistribution(Distribution):
    """Truncated normal distribution with parameters (mu, sigma)."""

    def __init__(self, mu, sigma, lower, upper):
        self.mu = mu
        self.sigma = sigma
        self.a = lower
        self.b = upper
        self.ss_dist = truncnorm(a=(self.a-self.mu)/self.sigma, b=(self.b-self.mu)/self.sigma, \
        	loc=self.mu, scale=self.sigma)
        # self.first_moment = Moments(self.mu, self.sigma, self.a, self.b).first_moment
        # self.second_moment = Moments(self.mu, self.sigma, self.a, self.b).second_moment

    def log_density(self, data):
        assert(len(data.shape) == 1), "Expect 1D data!"

        lower_cdf = norm.cdf(self.a, loc=self.mu, scale=self.sigma)
        upper_cdf = 1 - norm.cdf(self.b, loc=self.mu, scale=self.sigma)

        log_density = - ((data - self.mu) ** 2 / (2 * self.sigma ** 2) - np.log(self.sigma) - 0.5 * np.log(2 * np.pi))/(1-(lower_cdf+upper_cdf))

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


# if __name__ == "__main__":
	# data1 = np.random.lognormal(mean=0, sigma=0.75, size=1000)
	# data2 = np.random.lognormal(mean=10, sigma=0.9, size=1000)
	# data3 = np.concatenate((data2, data1))
	# print(data3)

	# plt.hist(data3, normed=True, bins=1000)
	# # plt.xlim(0, 1000)
	# plt.show()

	# mean_1, stdev_1 = norm.fit(data1)
	# mean, std = norm.fit(data3)

	# dist1 = mixem.distribution.LogNormalDistribution(0, 2)
	# dist2 = mixem.distribution.LogNormalDistribution(11, 3)
	# print(dist2.sigma)
	# print(dist2.mu)

	# weights, distributions, log_l = mixem.em(data3, [dist1, dist2], max_iterations=200)

	# print("Weights: ", weights)
	# print("Distributions: ", distributions)
	# print("Log Likelihood: ", log_l)

	# prob = mixem.probability(data3, weights, distributions)
	# print("Probability: ", len(prob))

