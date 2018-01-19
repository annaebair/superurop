import pytest
import numpy as np
import scipy.stats as s
from math import isclose
from em import Moments, TruncatedNormalDistribution, CensoredNormalDistribution
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

	def test_log_density_trunc_norm(self):
		pass

	def test_log_density_trunc_right(self):
		pass

	def test_log_density_trunc_left(self):
		pass

	################ Censored log_density #################

	def test_log_density_cens_norm(self):
		pass

	def test_log_density_cens_right(self):
		pass

	def test_log_density_cens_left(self):
		pass

	################ Truncated estimate_parameters #################	

	def test_estimate_parameters_trunc_norm(self):
		pass

	def test_estimate_parameters_trunc_right(self):
		pass

	def test_estimate_parameters_trunc_left(self):
		pass

	################ Censored estimate_parameters #################	

	def test_estimate_parameters_cens_norm(self):
		pass

	def test_estimate_parameters_cens_right(self):
		pass

	def test_estimate_parameters_cens_left(self):
		pass

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
		

		




