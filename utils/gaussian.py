# borrowed from Multivariate Gaussian Document Representation from Word Embeddings for Text Categorization by G. Nikolentzos

import sys
import numpy as np


def build_kernel_matrices(train_matrices, test_matrices, k, alpha, run_test=False):
	""" 
	Build kernel matrices

	"""
	mean_vectors_train = []
	norms_mean_vectors_train = []
	covariance_matrices_train = []
	norms_covariance_matrices_train = []
	for matrix in train_matrices:
		if matrix.shape[0] == 1:
			M = np.sum(matrix, axis=0)/float(matrix.shape[0])
			C = np.outer(M, M).flatten()
			mean_vectors_train.append(M)
			norms_mean_vectors_train.append(np.linalg.norm(M))
			covariance_matrices_train.append(C)
			norms_covariance_matrices_train.append(np.linalg.norm(C))
		elif matrix.shape[0] > 0:
			M = np.sum(matrix, axis=0)/float(matrix.shape[0])
			C = np.cov(matrix,rowvar=False).flatten()
			mean_vectors_train.append(M)
			norms_mean_vectors_train.append(np.linalg.norm(M))
			covariance_matrices_train.append(C)
			norms_covariance_matrices_train.append(np.linalg.norm(C))
		else:
			M = np.zeros(k)
			C = np.zeros((k, k)).flatten()
			mean_vectors_train.append(M)
			norms_mean_vectors_train.append(1)
			covariance_matrices_train.append(C)
			norms_covariance_matrices_train.append(1)

	kernel_matrix_train = np.zeros((len(train_matrices), len(train_matrices)), dtype=np.float16)
	for i in range(len(train_matrices)):
		for j in range(i,len(train_matrices)):
			kernel_matrix_train[i,j] = alpha*(np.dot(mean_vectors_train[i], mean_vectors_train[j])/(norms_mean_vectors_train[i]*norms_mean_vectors_train[j])) + (1-alpha)*(np.dot(covariance_matrices_train[i], covariance_matrices_train[j])/(norms_covariance_matrices_train[i]*norms_covariance_matrices_train[j]))
			kernel_matrix_train[j,i] = kernel_matrix_train[i,j]

	kernel_matrix_test = np.zeros((len(test_matrices), len(train_matrices)))
	if run_test:
		mean_vectors_test = []
		norms_mean_vectors_test = []
		covariance_matrices_test = []
		norms_covariance_matrices_test = []
		for matrix in test_matrices:
			if matrix.shape[0] == 1:
				M = np.sum(matrix, axis=0)/float(matrix.shape[0])
				C = np.outer(M, M).flatten()
				mean_vectors_test.append(M)
				norms_mean_vectors_test.append(np.linalg.norm(M))
				covariance_matrices_test.append(C)
				norms_covariance_matrices_test.append(np.linalg.norm(C))
			elif matrix.shape[0] > 0:
				M = np.sum(matrix, axis=0)/float(matrix.shape[0])
				C = np.cov(matrix,rowvar=False).flatten()
				mean_vectors_test.append(M)
				norms_mean_vectors_test.append(np.linalg.norm(M))
				covariance_matrices_test.append(C)
				norms_covariance_matrices_test.append(np.linalg.norm(C))
			else:
				M = np.zeros(k)
				C = np.zeros((k, k)).flatten()
				mean_vectors_test.append(M)
				norms_mean_vectors_test.append(1)
				covariance_matrices_test.append(C)
				norms_covariance_matrices_test.append(1)

		for i in range(len(test_matrices)):
			for j in range(len(train_matrices)):
				kernel_matrix_test[i,j] = alpha*(np.dot(mean_vectors_test[i], mean_vectors_train[j])/(norms_mean_vectors_test[i]*norms_mean_vectors_train[j])) + (1-alpha)*(np.dot(covariance_matrices_test[i], covariance_matrices_train[j])/(norms_covariance_matrices_test[i]*norms_covariance_matrices_train[j])) 
	return kernel_matrix_train, kernel_matrix_test

