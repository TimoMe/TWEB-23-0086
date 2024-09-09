from collections import defaultdict
import csv
import pickle as pkl

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from networkx import read_weighted_edgelist, to_scipy_sparse_array


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)


class ModifiedAdsorption(object):
    '''
    classdocs
    '''

    def read_graph_seeds(self, graph_file, seed_file, nb_seeds=10000, delimiter="\t"):
        """
        Quick and dirty way to get the adjacency matrix and seeds!
        """
        g_mat = load_pickle(graph_file)
        # G = from_(graph_file, delimiter=delimiter)
        W = g_mat
        nodes_G = [i for i in range(g_mat.shape[0])]
        label_index = defaultdict(list)
        # Deal with seeds
        file_lines = open(seed_file, "r").readlines()
        seed_nodes, seed_labels, seed_values = [], [], []
        for idx, line in enumerate(file_lines):
            split_line = line.split(delimiter)
            node, label, value = split_line[0], split_line[1], float(split_line[2])
            seed_nodes.append(node)
            seed_labels.append(label)
            seed_values.append(value)
            label_index[label].append(idx)

        # Store golden_labels/seeds node name and their real value
        golden_labels = dict(zip(seed_nodes, seed_labels))
        # print(golden_labels)

        unique_labels = sorted(set(seed_labels))
        labels_dict = {e: idx for idx, e in enumerate(unique_labels)}  # [L1, L2, L3,..., DUMMY]

        # load lr result & concat seed matrix
        infer_result_valid = np.load('../HUG/result/LR_cmu_valid.npy')
        infer_result_test = np.load('../HUG/result/LR_cmu_test.npy')
        infer_result = np.vstack((infer_result_valid, infer_result_test))
        # Build the seeds matrix: number of nodes x number of labels + 1
        seeds_matrix = np.zeros(
            [W.shape[0], len(set(seed_labels))+1])  # We add 1 because of the "dummy" label used in the algorithm

        # Draw a n sample for each type of label
        # label_samples = {}
        # for k, v in label_index.items():
        #     temp_nb_seeds = nb_seeds if len(v) > nb_seeds else len(v)
        #     label_samples[k] = np.random.choice(v, temp_nb_seeds)

        # Once we have the sample for each class, we build the seeds matrix
        for label, seed_idx in label_index.items():
            label_j = labels_dict[label]
            for i in seed_idx:
                seeds_matrix[i, label_j] = seed_values[i]

        seeds_matrix[9475: ,:] = infer_result
        print('The number of labeled nodes is ', len(seed_nodes)+len(infer_result_valid)+len(infer_result_test))

        return W, seeds_matrix, unique_labels, golden_labels, nodes_G

    def __init__(self, graph_file, seeds_file, nb_seeds=10, tol=7e-2, maxiter=1):

        '''
        Constructor
        '''
        self._W, self._Y, labels, self._golden_labels, self._nodes_list = self.read_graph_seeds(graph_file,
                                                                                                seeds_file,
                                                                                                nb_seeds=nb_seeds)
        self._labels = labels
        self._mu1 = 1
        self._mu2 = 0.1
        self._mu3 = 0
        self._beta = 2.
        self._tol = tol
        self._get_initial_probs()
        self.max_iter = maxiter

    def _get_initial_probs(self):
        print('\t..Getting initial probabilities.')
        nr_nodes = self._W.shape[0]

        # Calculate Pr transition probabilities matrix
        self._Pr = lil_matrix((nr_nodes, nr_nodes))
        W_coo = self._W.tocoo()
        print(len(W_coo.sum(0).tolist()))
        col_sums = {k: v for k, v in enumerate(W_coo.sum(0).tolist())}
        for i, j, v in zip(W_coo.row, W_coo.col, W_coo.data):
            # print "\t%d\t%d" % (i,j)
            self._Pr[i, j] = v / col_sums[j]

        # W_nnz = self._W.nonzero()
        # for v1, v2 in zip(W_nnz[0], W_nnz[1]):
        # # Get the sum fast
        # print "\t%d\t%d" % (v1,v2)
        # self._Pr[v1, v2] = self._W[v1, v2] / (self._W[:, v2].sum())

        print("\t\tPr matrix done.")
        # Calculate H entropy vector for each node
        self._H = lil_matrix((nr_nodes, 1))
        self._Pr = self._Pr.tocoo()

        for i, _, v in zip(self._Pr.row, self._Pr.col, self._Pr.data):
            self._H[i, 0] += -(v * np.log(v))

        # Pr_nnz = self._Pr.nonzero()
        # for v, u in zip(Pr_nnz[0], Pr_nnz[1]):
        # self._H[v, 0] += -(self._Pr[v, u] * np.log(self._Pr[v, u]))
        print("\t\tH matrix done.")

        # Calculate vector C (Cv)
        self._H = self._H.tocoo()
        # H_nnz = self._H.nonzero()
        self._C = np.ones((nr_nodes, 1))
        log_beta = np.log(self._beta)
        for i, _, v in zip(self._H.row, self._H.col, self._H.data):
            # print v
            self._C[i, 0] = (log_beta) / (np.log(self._beta + (1 / (np.exp(-v) + 0.00001))))

        # for i in H_nnz[0]:
        # self._C[i, 0] = (np.log(self._beta)) / (np.log(self._beta + np.exp(self._H[i, 0])))
        print("\t\tC matrix done.")

        # Calculate vector D (dv)
        # Get nodes that are labeled
        Y_nnz = self._Y.nonzero()
        print(self._Y.shape)
        print(len(set(Y_nnz[0])))
        self._D = lil_matrix((nr_nodes, 1))
        self._H = self._H.tolil()

        for i in set(Y_nnz[0]):
            # Check if node v is labeled            
            self._D[i, 0] = (1. - self._C[i, 0]) * np.sqrt(self._H[i, 0])

        print("\t\tD matrix done.")
        # Calculate Z vector
        self._Z = lil_matrix((nr_nodes, 1))
        c_v = self._C + self._D
        
        z_zero = 0
        for i in range(c_v.shape[0]):
            self._Z[i, 0] = np.max([c_v[i, 0], 1.])
            if self._Z[i, 0] == 0:
                z_zero += 1
        print("z_zero", z_zero)
        print("\t\tZ matrix done.")

        # Finally calculate p_cont, p_inj and p_abnd
        self._Pcont = np.zeros((nr_nodes, 1))
        self._Pinj = lil_matrix((nr_nodes, 1))
        # self._Pabnd = lil_matrix((nr_nodes, 1))
        
        for i in range(self._C.shape[0]):
            self._Pcont[i, 0] = self._C[i, 0] / self._Z[i, 0]
            if self._Pcont[i, 0] == 0:
                print("Negative p_cont", i)
        print("\t\tPcont matrix done.")
            
        for i in range(self._D.shape[0]):
            self._Pinj[i, 0] = self._D[i, 0] / self._Z[i, 0]
            
        # self._Pabnd[:, :] = 1.
        # pc_pa = self._Pcont + self._Pinj
        # pc_pa_nnz = pc_pa.nonzero()
        # for i in pc_pa_nnz[0]:
        #     self._Pabnd[i, 0] = 1. - pc_pa[i, 0]
        # for i in range(nr_nodes):
        # self._Pabnd[i, 0] = 1. - self._Pcont[i, 0] - self._Pinj[i, 0]

        # self._Pabnd = csr_matrix(self._Pabnd)
        self._Pcont = csr_matrix(self._Pcont)
        self._Pinj = csr_matrix(self._Pinj)
        print("\n\nDone getting probabilities...")

    def results(self):
        """
        Return the class determined by the maximum in each row of the Yh matrix. Doesnt
        take into account the dummy label
        """
        result_complete = []
        self._mad_class_index = np.squeeze(np.asarray(self._Yh[:, :self._Yh.shape[1] - 1].argmax(axis=1)))
        self._label_results = np.array([self._labels[r] for r in self._mad_class_index])
        # print(self._label_results)
        for i in range(len(self._label_results)):
            result_complete.append([self._nodes_list[i], self._label_results[i],
                                    self._golden_labels.get(str(self._nodes_list[i]), "NO GOLDEN LABEL")])
        # print(result_complete)
        self._result_complete = result_complete
        return self._label_results, self._result_complete

    def calculate_mad(self):
        print("\n...Calculating modified adsorption.")
        nr_nodes = self._W.shape[0]

        # 1. Initialize Yhat
        self._Yh = self._Y.copy()

        # 2. Calculate Mvv
        self._M = lil_matrix((nr_nodes, nr_nodes))
        # self._M = lil_matrix(np.diag((self._mu1*self._Pinj).toarray().flatten()) + (np.eye(nr_nodes)*self._mu3))
        """
        TODO: This does not work cause flattening and to array-ing is not memory cool so it fails. Need to find a way to
            build this initial matrices with sparse matrices
        """
        indeptr = self._W.indptr
        for v in range(nr_nodes):
            first_part = self._mu1 * self._Pinj[v, 0]
            second_part = 0.

            for u in list(self._W.indices[indeptr[v]:indeptr[v + 1]]):
                if u != v:
                    second_part += (self._Pcont[v, 0] * self._W[v, u] + self._Pcont[u, 0] * self._W[u, v])
            # print('second_part: ', second_part)
            self._M[v, v] = first_part + (self._mu2 * second_part) + self._mu3
            if second_part == 0 and v > 5685_1895:
                print(v, list(self._W.indices[indeptr[v]:indeptr[v + 1]]))
            # print(self._M[v, v])

        # W_coo = self._W.tocoo()
        #
        # for i, j, v in zip(W_coo.row, W_coo.col, W_coo.data):
        # second_part = self._mu2 * ((self._Pcont[i, 0] * v) + self._Pcont[j, 0] * v)
        # self._M[i, i] += second_part

        # self._M = csr_matrix(self._M)

        # 3. Begin main loop
        itero = 0
        r = lil_matrix((1, self._Y.shape[1]))
        r[-1, -1] = 1.
        Yh_old = lil_matrix((self._Y.shape[0], self._Y.shape[1]))

        # Main loop begins
        Pcont = np.array([[row[0] for idx in range(self._Pcont.shape[0])] for row in self._Pcont.toarray()])
        Pcont_t = Pcont.transpose()
        Pcont_ex = Pcont + Pcont_t
        W_csr = self._W.tocsr()
        weight_mat = W_csr.multiply(Pcont_ex).tocsr()
        # W_coo = self._W.tocoo()
        while not self._check_convergence(Yh_old, self._Yh, ) and self.max_iter > itero:
            itero += 1
            print(">>>>>Iteration:%d" % itero)
            self._D = lil_matrix((nr_nodes, self._Y.shape[1]))
            # 4. Calculate Dv
            print("\t\tCalculating D...")
            time_d = time.time()
            
            # for i, j, v in zip(W_coo.row, W_coo.col, W_coo.data):
            #     # self._D[i, :] += (Pcont[i][0] * v + Pcont[j][0] * v) * self._Yh[j, :]
            #     self._D[i, :] += (v * (Pcont[i][0] + Pcont[j][0])) * self._Yh[j, :]
                # print i
            print(type(weight_mat), weight_mat.shape, self._Yh.shape)
            self._D = weight_mat.dot(self._Yh)

            # print(self._D[0].todense())

            print("\t\tTime it took to calculate D:", time.time() - time_d)
            print
            # for v in range(nr_nodes):
            # for u in self._W[v, :].nonzero()[1]:
            # self._D[v, :] += (self._Pcont[v, 0] * self._W[v, u] + self._Pcont[u, 0] * self._W[u, v])\
            # * self._Yh[u, :]
            # print v,  self._D[v, :].todense()

            print("\t\tUpdating Y...")
            # 5. Update Yh
            time_y = time.time()
            Yh_old = self._Yh.copy()
            v_num = 0
            for v in range(len(self._golden_labels), 9475):
                # 6.
                second_part = ((self._mu1 * self._Pinj[v, 0] * self._Y[v, :]) +
                               (self._mu2 * self._D[v, :]) )
                if self._M[v, v] != 0.:
                    self._Yh[v, :] = second_part / (self._M[v, v])
                else:
                    v_num += 1
                # print v
            # self._Yh = csr_matrix(self._Yh)
            print("\t\tNumber of nodes with zero degree: ", v_num)
            print("\t\tTime it took to calculate Y:", time.time() - time_y)
            
            # repeat until convergence.

    def _check_convergence(self, A, B):
        if type(A) != np.ndarray:
            A = A.todense()
        mat_square = np.square(B-A).sum()
        # print(mat_square)
        diff = np.sqrt(np.abs(mat_square)/ 13265) 
        print(diff)
        if diff <= self._tol:
            return True
        else:
            print("\t\tNorm differences between Y_old and Y_hat: ", diff)

            return False


if __name__ == '__main__':
    import time

    # # test_file
    # graph_file = "./data/test_data/input_graph"
    # seed_file = "./data/test_data/seeds"

    graph_file = '../dataset/cmu/graph_mat_5_lr.pkl'
    seed_file = '../dataset/cmu/seeds'

    # W2, Y2, labels, golden_labels, nodes_names = read_graph_seeds(graph_file, seed_file)
    timo = time.time()

    mad = ModifiedAdsorption(graph_file, seed_file, maxiter=10)
    mad.calculate_mad()
    print("MAD took:", time.time() - timo)
    y_pred, y_pred_complete = mad.results()
    with open('../dataset/cmu/mad_result_1_0.1_cover.txt', 'w') as f:
        wt = csv.writer(f, delimiter='\t')
        wt.writerows(y_pred_complete)
    # # print y_pred
    # pass