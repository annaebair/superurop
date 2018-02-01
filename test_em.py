import pytest
import numpy as np
import scipy.stats as s
from math import isclose
from em import Moments, TruncatedNormalDistribution, CensoredNormalDistribution
from mixem.distribution import NormalDistribution
from p_value import Data_Rep

class TestEM:

	################ Moments #################	

	def test_first_moment_standard_normal(self):
		first = Moments(0, 1, -np.inf, np.inf).first_moment
		assert first == 0

	def test_second_moment_standard_normal(self):
		second = Moments(0, 1, -np.inf, np.inf).second_moment
		assert second == 1.0

	def test_first_moment_positive_mean(self):
		first = Moments(5, 3, -np.inf, np.inf).first_moment
		assert first == 5

	def test_first_moment_negative_mean(self):
		first = Moments(-5, 3, -np.inf, np.inf).first_moment
		assert first == -5

	def test_second_moment_positive_mean(self):
		mu = 5
		sigma = 2
		a = -np.inf
		b = np.inf
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		expected = mu ** 2 + sigma ** 2
		assert actual == expected

	def test_second_moment_negative_mean(self):
		mu = -4
		sigma = 2
		a = -np.inf
		b = np.inf
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		expected = mu ** 2 + sigma ** 2
		assert actual == expected

	def test_second_moment_positive(self):
		mu = -4
		sigma = 2
		a = -np.inf
		b = np.inf
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		assert actual > 0

	def test_second_moment_positive_truncated(self):
		mu = 4
		sigma = 2
		a = -2
		b = 5
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		assert actual > 0

	def test_second_moment_positive_right_of_mean(self):
		mu = 4
		sigma = 2
		a = 5
		b = 7
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		assert actual > 0

	def test_second_moment_positive_left_of_mean(self):
		mu = 4
		sigma = 2
		a = 0
		b = 1
		actual = Moments(mu=mu, sigma=sigma, a=a, b=b).second_moment
		assert actual > 0

	################ Truncated log_density #################

	def test_trunc_log_density(self):
		mu=0
		sigma=1
		a=-1
		b=1
		data = np.array([-0.5, 0, 0.5])
		expected_dist = s.truncnorm(loc=mu, scale=sigma, a=a, b=b)
		actual_dist = TruncatedNormalDistribution(mu=mu, sigma=sigma, lower=a, upper=b)
		expected = expected_dist.logpdf(data)
		actual = actual_dist.log_density(data)
		assert (actual == expected).all()

	def test_trunc_log_density_shifted(self):
		mu=3
		sigma=2
		a=0
		b=4
		data = np.array([0.5, 1, 2, 3, 3.9])
		expected_dist = s.truncnorm(loc=mu, scale=sigma, a=(a-mu)/sigma, b=(b-mu)/sigma)
		actual_dist = TruncatedNormalDistribution(mu=mu, sigma=sigma, lower=a, upper=b)
		expected = expected_dist.logpdf(data)
		actual = actual_dist.log_density(data)
		for i in range(len(expected)):
			assert isclose(actual[i], expected[i], abs_tol=1e-10)

	def test_trunc_log_density_outside_range(self):
		mu=3
		sigma=2
		a=0
		b=4
		data = np.array([-1, 1, 2, 3, 5])
		expected_dist = s.truncnorm(loc=mu, scale=sigma, a=(a-mu)/sigma, b=(b-mu)/sigma)
		actual_dist = TruncatedNormalDistribution(mu=mu, sigma=sigma, lower=a, upper=b)
		expected = expected_dist.logpdf(data)
		expected[0] = -9999
		expected[-1] = -9999
		actual = actual_dist.log_density(data)
		for i in range(len(expected)):
			assert isclose(actual[i], expected[i], abs_tol=1e-10)

	################ Censored log_density #################

	def test_log_density_cens_norm(self):
		mu = 4
		sigma = 2
		a = 3
		b = 6
		data = np.array([2, 7])
		dist = CensoredNormalDistribution(mu=mu, sigma=sigma, lower=a, upper=b)
		actual = dist.log_density(data)
		expected = np.array([-9999, -9999])
		assert (actual == expected).all()

	def test_log_density_cens_norm_at_bounds(self):
		mu = 4
		sigma = 2
		a = 3
		b = 6
		data = np.array([3, 6])
		dist = CensoredNormalDistribution(mu=mu, sigma=sigma, lower=a, upper=b)
		actual = dist.log_density(data)
		assert all(i > 1 for i in actual)

	def test_compare_censored_to_normal(self):
		mu = 4
		sigma = 2
		a = 3
		b = 6
		data = np.array([3.5, 4, 5, 5.5])
		dist = CensoredNormalDistribution(mu=mu, sigma=sigma, lower=a, upper=b)
		norm = s.norm(loc=mu, scale=sigma)
		actual = dist.log_density(data)
		expected = norm.logpdf(data)
		assert (actual == expected).all()


	################ Truncated estimate_parameters #################	

	def test_estimate_parameters_trunc_norm(self):
		mu = 4
		sigma = 2
		a = 3
		b = 6
		data = np.random.normal(loc=mu, scale=sigma, size=1000)
		data = np.array(list(filter(lambda x: x > a and x < b, data)))
		dist = TruncatedNormalDistribution(mu=mu, sigma=sigma, lower=a, upper=b)
		actual = dist.estimate_parameters(data, np.ones((data.shape)))
		new_mu = dist.mu
		new_sigma = dist.sigma
		assert isclose(mu, new_mu, abs_tol=0.5)
		assert new_sigma > 0

	def test_estimate_parameters_trunc_norm_neg_mean(self):
		mu = -2
		sigma = 1
		a = -3
		b = -1
		data = np.random.normal(loc=mu, scale=sigma, size=1000)
		data = np.array(list(filter(lambda x: x > a and x < b, data)))
		dist = TruncatedNormalDistribution(mu=mu, sigma=sigma, lower=a, upper=b)
		actual = dist.estimate_parameters(data, np.ones((data.shape)))
		new_mu = dist.mu
		new_sigma = dist.sigma
		assert isclose(mu, new_mu, abs_tol=0.5)
		assert new_sigma > 0


	################ Censored estimate_parameters #################	

	def test_estimate_parameters_cens_norm(self):
		mu = 4
		sigma = 2
		a = 3
		b = 6
		data = np.random.normal(loc=mu, scale=sigma, size=1000)
		data = np.where(data <= a, a, data)
		data = np.where(data >= b, b, data)
		dist = CensoredNormalDistribution(mu=mu, sigma=sigma, lower=a, upper=b)
		actual = dist.estimate_parameters(data, np.ones((data.shape)))
		new_mu = dist.mu
		new_sigma = dist.sigma
		assert isclose(mu, new_mu, abs_tol=0.5)
		assert new_sigma > 0

	def test_estimate_parameters_cens_norm_neg_mean(self):
		mu = -2
		sigma = 1
		a = -3
		b = -1
		data = np.random.normal(loc=mu, scale=sigma, size=1000)
		data = np.array(list(filter(lambda x: x > a and x < b, data)))
		dist = CensoredNormalDistribution(mu=mu, sigma=sigma, lower=a, upper=b)
		actual = dist.estimate_parameters(data, np.ones((data.shape)))
		new_mu = dist.mu
		new_sigma = dist.sigma
		assert isclose(mu, new_mu, abs_tol=0.5)
		assert new_sigma > 0

	################ p-value #################

	def test_censored_p_values(self):
		data1 = np.random.normal(loc=0, scale=1, size=1000)
		data1 = np.where(data1 <= 0, 0, data1)
		data1 = np.where(data1 >= 1, 1, data1)

		data2 = np.random.normal(loc=3, scale=1, size=1000)
		data2 = np.where(data2 <= 2, 2, data2)
		data2 = np.where(data2 >= 4, 4, data2)

		dist1 = CensoredNormalDistribution(mu=0.5, sigma=1, lower=0, upper=1)
		dist2 = CensoredNormalDistribution(mu=1.5, sigma=1, lower=2, upper=4)

		# after dataRep, make sure that params are close to expected
		
		data = np.concatenate((data1, data2))
		dist_list = [dist1, dist2]
		mixture = Data_Rep(data, dist_list)

		observed = []
		for i in [-1, 1.5, 5]:
			observed.append(mixture.get_p_value(i))
		expected = [1.0, 0.5, 0.0]

		assert observed == expected

	def test_truncated_p_values(self):
		data1 = np.random.normal(loc=0, scale=1, size=1000)
		data2 = np.random.normal(loc=3, scale=1, size=1000)

		data1 = np.array(list(filter(lambda x: x > 0 and x < 1, data1)))
		data2 = np.array(list(filter(lambda x: x > 2 and x < 4, data2)))

		dist1 = TruncatedNormalDistribution(mu=0.5, sigma=1, lower=0, upper=1)
		dist2 = TruncatedNormalDistribution(mu=1.5, sigma=1, lower=2, upper=4)
		
		data = np.concatenate((data1, data2))
		dist_list = [dist1, dist2]
		mixture = Data_Rep(data, dist_list)

		dist_weight = len(data2)/len(data)
		expected = [1.0, dist_weight, 0.0]
		observed = []
		for i in [-1, 1.5, 5]:
			observed.append(mixture.get_p_value(i))

		for i in range(len(expected)):
			assert isclose(expected[i], observed[i], abs_tol=1e-8)



	# def test_censored_and_normal_p_values(self):
	# 	data1 = np.random.normal(loc=0, scale=1, size=1000)
	# 	data1 = np.where(data1 <= 0, 0, data1)
	# 	data1 = np.where(data1 >= 1, 1, data1)

	# 	data2 = np.random.normal(loc=3, scale=1, size=1000)

	# 	dist1 = CensoredNormalDistribution(mu=0.5, sigma=1, lower=0, upper=1)
	# 	dist2 = NormalDistribution(mu=3, sigma=1)

	# 	# after dataRep, make sure that params are close to expected
		
	# 	data = np.concatenate((data1, data2))
	# 	dist_list = [dist1, dist2]
	# 	mixture = Data_Rep(data, dist_list)
	# 	dist_weight = mixture.weights[0]
	# 	print("sf: ", s.norm.sf(-1.5))

	# 	observed = []
	# 	for i in [-1, 1.5, 5]:
	# 		observed.append(mixture.get_p_value(i))
	# 	expected = [0+s.norm.sf(-4), 1-(dist_weight+s.norm.cdf(-1.5)), 0+ s.norm.sf(2)] #manually get p value here

	# 	#get p values of normal
	# 	print(observed)

	# 	for i in range(len(expected)):
	# 		assert isclose(expected[i], observed[i], abs_tol=1e-4)
