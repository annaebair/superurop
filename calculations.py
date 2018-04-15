import mixem
import numpy as np
from math import exp, isclose
import math
import time
from scipy.stats import norm, expon, truncnorm, lognorm
import scipy.stats as ss
import matplotlib.pyplot as plt
from em import Moments, TruncatedNormalDistribution, CensoredNormalDistribution
from p_value import *


def generate_data(x):

	# Censored 1
	# predata1 = np.random.normal(loc=-1, scale=1, size=100000)
	# censored_data_1 = np.where(predata1 <= -2, -2, predata1)
	# censored_data_1 = np.where(censored_data_1 >= 3, 3, censored_data_1)
	# censored_dist_1 = CensoredNormalDistribution(mu=1.0, sigma=1, lower=-2, upper=3)
	# censored_dist_1_rand = CensoredNormalDistribution(mu=-1, sigma=1, lower=-2, upper=3)

	# Censored 2
	# predata2 = np.random.normal(loc=3, scale=1, size=1000)
	# censored_data_2 = np.where(predata2 <= -3, -3, predata2)
	# censored_data_2 = np.where(censored_data_2 >= 4, 4, censored_data_2)
	# censored_dist_2 = CensoredNormalDistribution(mu=1.5, sigma=1, lower=-3, upper=4)

	# Censored 2 over same range as Censored 1
	# predata2 = np.random.normal(loc=2, scale=1, size=100000)
	# censored_data_2 = np.where(predata2 <= -2, -2, predata2)
	# censored_data_2 = np.where(censored_data_2 >= 3, 3, censored_data_2)
	# censored_dist_2_rand = CensoredNormalDistribution(mu=2, sigma=1, lower=-2, upper=3)

	# Truncated 1
	predata3 = np.random.normal(loc=1, scale=1, size=100000)
	truncated_data_1 = np.array(list(filter(lambda x: x > 0 and x < 6, predata3)))
	truncated_dist_1_rand = TruncatedNormalDistribution(mu=0.5, sigma=1, lower=0, upper=6)

	# Truncated 2
	# predata4 = np.random.normal(loc=5, scale=1, size=146500)
	# truncated_data_2 = np.array(list(filter(lambda x: x > 4 and x < 6, predata4)))
	# truncated_dist_2_rand = TruncatedNormalDistribution(mu=3, sigma=1.004, lower=4, upper=6)

	# Truncated 2 over same range as Truncated 1
	predata4 = np.random.normal(loc=5, scale=1, size=146500)
	truncated_data_2 = np.array(list(filter(lambda x: x > 0 and x < 6, predata4)))
	truncated_dist_2_rand = TruncatedNormalDistribution(mu=1.8, sigma=1, lower=0, upper=6)

	# Normal 1
	# normal_data_1 = np.random.normal(loc=-1, scale=1, size=100000)
	# normal_dist_1 = mixem.distribution.NormalDistribution(mu=0.5, sigma=1)
	# normal_dist_1_rand = mixem.distribution.NormalDistribution(mu=-3, sigma=3)

	# Normal 2
	# normal_data_2 = np.random.normal(loc=3, scale=2, size=100000)
	# normal_dist_2 = mixem.distribution.NormalDistribution(mu=3.434, sigma=1.964)
	# normal_dist_2_rand = mixem.distribution.NormalDistribution(mu=1, sigma=0.5)

	# trunced_normal = np.array(list(filter(lambda x: x > 0 and x < 2, normal_data_2)))
	# trunced_normal_dist = TruncatedNormalDistribution(mu=3, sigma=2, lower=0, upper=2)

	mixture, joint_pdf, data = organize_data(x, truncated_data_1, truncated_data_2, truncated_dist_1_rand, truncated_dist_2_rand)
	return mixture, joint_pdf, data

def organize_data(x, data1, data2, dist1, dist2):

	data = np.concatenate((data1, data2))
	np.random.shuffle(data)
	# train_set = data[1000:]
	# val_set = data[:1000]
	# sorted_val = np.array(sorted(val_set))

	dist_list = [dist1, dist2]
	mixture = Data_Rep(data, dist_list)
	post_scipy_dist_1, post_scipy_dist_2 = mixture.scipy_dists
	# took out train/val split here
	weights, distributions, log_l = mixem.em(data, dist_list, max_iterations=200, progress_callback=None)

	pdf1 = [post_scipy_dist_1.pdf(i) for i in x]
	pdf2 = [post_scipy_dist_2.pdf(i) for i in x]
	joint_pdf = [weights[0]*pdf1[i] + weights[1]*pdf2[i] for i in range(len(pdf1))]

	return mixture, joint_pdf, data


def get_normals_mixture(x, data):

	dist_list_norm = [mixem.distribution.NormalDistribution(mu=0.5, sigma=1), mixem.distribution.NormalDistribution(mu=2, sigma=1)]
	norm_mixture = Data_Rep(data, dist_list_norm) 
	#removed ref to train set
	norm_weights, norm_dists, norm_log_l = mixem.em(data, dist_list_norm, max_iterations=200, progress_callback=None)

	post_scipy_dist_1_norm, post_scipy_dist_2_norm = norm_mixture.scipy_dists
	norm_pdf1 = [post_scipy_dist_1_norm.pdf(i) for i in x]
	norm_pdf2 = [post_scipy_dist_2_norm.pdf(i) for i in x]
	norm_joint_pdf = [norm_weights[0]*norm_pdf1[i] + norm_weights[1]*norm_pdf2[i] for i in range(len(norm_pdf1))]

	return norm_mixture, norm_joint_pdf


def get_p_values(sample_nums, mixture):
	pvals = []
	for i in sample_nums:
		query = i
		p = mixture.get_p_value(query)
		pvals.append(p)
	return pvals


def significant_difference(mixture, norm_mixture):
	datamean = np.mean(data)
	datastd = np.std(data)
	datasize = len(data)
	all_calc_p = []
	ss_pvals = []
	diff_amt = []
	for i in range(len(x)):
		pdf_val, reg_p_val = mixture.alt_p_val_calc(x, x[i])
		# hack to fix problem of maximum value having a p value of zero
		if reg_p_val == 0 and pdf_val !=0:
			reg_p_val = all_calc_p[i-1]
		all_calc_p.append(reg_p_val)
		norm_p_val = norm_mixture.get_p_value(x[i])
		t, p = ss.ttest_ind_from_stats(datamean, datastd, datasize, x[i], 0, 1)
		ss_pvals.append(p)
		diff_amt.append(reg_p_val - norm_p_val)
	return all_calc_p, ss_pvals, diff_amt


def plots(x, joint_pdf, ss_pvals, all_calc_p, diff_amt):

	plt.figure()

	plt.subplot(221)
	plt.hist(data, bins=200, normed=True)
	plt.plot(x, joint_pdf)
	plt.ylim(0, 0.5)
	plt.xlim(-3,7)
	plt.xlabel("Data Values")
	plt.ylabel("Frequency")
	plt.title("Mixture of Truncated Distributions")

	plt.subplot(222)
	plt.plot(x, ss_pvals)
	plt.ylim(0,1)
	plt.xlim(-3, 7)
	plt.xlabel("Query point")
	plt.ylabel("P-value")
	plt.title("Standard P-value Calculation")

	plt.subplot(223)
	plt.plot(x, all_calc_p)
	plt.ylim(0, 1)
	plt.xlim(-3, 7)
	plt.xlabel("Query point")
	plt.ylabel("P-value")
	plt.title("New Method for Calculating P-values")

	plt.subplot(224)
	plt.plot(x, diff_amt)
	plt.ylim(-1, 1)
	plt.xlim(-3, 7)
	plt.xlabel("Query point")
	plt.ylabel("P-value")
	plt.title("Difference in P-value Calculations")

	plt.show()


if __name__ == "__main__":

	x = np.arange(-7, 7, 0.001)
	sample_nums = []
	
	mixture, joint_pdf, data = generate_data(x)
	norm_mixture, norm_joint_pdf = get_normals_mixture(x, data)
	all_calc_p, ss_pvals, diff_amt = significant_difference(mixture, norm_mixture)

	plots(x, joint_pdf, ss_pvals, all_calc_p, diff_amt)
