import math
import numpy as np
from tqdm import tqdm
from scalesim.scale_config import scale_config as cfg
import os


class systolic_compute_ws:
    def __init__(self, num_drams=4):
        # Params set by user
        self.config = cfg()
        self.num_drams = num_drams

        self.ifmap_op_mats = [np.zeros((1, 1)) for _ in range(num_drams)]
        self.ofmap_op_mat = np.zeros((1, 1))
        self.filter_op_mats = [np.zeros((1, 1)) for _ in range(num_drams)]

        # Derived parameters
        self.Sr = 0
        self.Sc = 0
        self.T = 0

        self.arr_row = 0
        self.arr_col = 0

        self.row_fold = 1
        self.col_fold = 1

        # Generated matrices
        self.ifmap_op_mat_trans = np.zeros((1, 1))
        self.ifmap_prefetch_matrices = [np.zeros((1, 1)) for _ in range(num_drams)]
        self.filter_prefetch_matrices = [np.zeros((1, 1)) for _ in range(num_drams)]

        self.ifmap_demand_matrices = [np.zeros((1, 1)) for _ in range(num_drams)]
        self.ofmap_demand_matrix = np.zeros((1, 1))
        self.filter_demand_matrices = [np.zeros((1, 1)) for _ in range(num_drams)]

        # Generated metrics
        self.ifmap_reads = 0
        self.filter_reads = 0
        self.ofmap_writes = 0

        self.mapping_efficiency_per_fold = []
        self.compute_utility_per_fold = []

        # Flags
        self.params_set_flag = False
        self.prefetch_mat_ready_flag = False
        self.demand_mat_ready_flag = False

    #
    def set_params(self, config_obj=cfg(), ifmap_op_mats=None, ofmap_op_mat=np.zeros((1,1)), filter_op_mats=None):
        self.config = config_obj
        self.num_drams = len(ifmap_op_mats)

        self.ifmap_op_mats = ifmap_op_mats
        self.filter_op_mats = filter_op_mats
        self.ofmap_op_mat = ofmap_op_mat

        # Ensure dimensions are consistent across DRAMs
        assert all(ifmap_op_mat.shape == self.ifmap_op_mats[0].shape for ifmap_op_mat in self.ifmap_op_mats), "Dimension mismatch between IFMAP operands"
        assert all(filter_op_mat.shape == self.filter_op_mats[0].shape for filter_op_mat in self.filter_op_mats), "Dimension mismatch between Filter operands"

        self.Sr = self.ifmap_op_mats[0].shape[1]
        self.Sc = self.filter_op_mats[0].shape[1]
        self.T = self.ifmap_op_mats[0].shape[0]

        self.arr_row, self.arr_col = self.config.get_array_dims()

        self.row_fold = math.ceil(self.Sr / self.arr_row)
        self.col_fold = math.ceil(self.Sc / self.arr_col)

        self.params_set_flag = True

    #
    def create_prefetch_matrices(self):
        assert self.params_set_flag, 'Parameters are not set'

        for i in range(self.num_drams):
            self.create_ifmap_prefetch_mat(i)
            self.create_filter_prefetch_mat(i)

        self.prefetch_mat_ready_flag = True

    #
    def create_ifmap_prefetch_mat(self, dram_idx):
        assert self.params_set_flag, 'Parameters are not set'

        ifmap_op_mat = self.ifmap_op_mats[dram_idx]

        for fr in range(self.row_fold):
            start_col_idx = fr * self.arr_row
            end_col_idx = min(start_col_idx + self.arr_row, self.Sr)

            delta = self.arr_row - (end_col_idx - start_col_idx)

            this_fold_prefetch = ifmap_op_mat[:, start_col_idx: end_col_idx]

            # If there is under utilization, fill them with null requests
            if delta > 0:
                null_req_mat = np.ones((self.T, delta)) * -1
                this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

            if fr == 0:
                self.ifmap_prefetch_matrices[dram_idx] = this_fold_prefetch
            else:
                self.ifmap_prefetch_matrices[dram_idx] = np.concatenate((self.ifmap_prefetch_matrices[dram_idx], this_fold_prefetch), axis=0)

        # Fixing ISSUE #15, #16
        # Roll out the matrices along the diagonal to account for temporal locality when there is a skew in demand

        M, N = self.ifmap_prefetch_matrices[dram_idx].shape
        num_elems = M * N
        num_diags = M + N
        prefetches = np.zeros((1, num_elems))
        idx = 0

        pbar = tqdm(total=M * N, disable=True)
        # print('DEBUG: Total = ' + str(num_elems) + ' Diags = ' + str(num_diags))

        for diag_id in range(num_diags):
            max_row_id = min(diag_id, M - 1)
            min_row_id = max(0, diag_id - N + 1)
            valid_rows = max_row_id - min_row_id + 1

            for offset in range(valid_rows):
                row_id = max_row_id - offset
                col_id = diag_id - row_id

                elem = self.ifmap_prefetch_matrices[dram_idx][row_id][col_id]
                prefetches[0, idx] = elem
                idx += 1
                pbar.update(1)

        pbar.close()
        self.ifmap_prefetch_matrices[dram_idx] = prefetches

    #
    def create_filter_prefetch_mat(self, dram_idx):
        assert self.params_set_flag, 'Parameters are not set'

        filter_op_mat = self.filter_op_mats[dram_idx]

        for fc in range(self.col_fold):
            col_start_id = fc * self.arr_col
            col_end_id = min(col_start_id + self.arr_col, self.Sc)

            delta = self.arr_col - (col_end_id - col_start_id)

            this_fold_prefetch = filter_op_mat[:, col_start_id: col_end_id]

            if delta > 0:
                null_req_mat = np.ones((self.Sr, delta)) * -1
                this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

            if fc == 0:
                self.filter_prefetch_matrices[dram_idx] = this_fold_prefetch
            else:
                self.filter_prefetch_matrices[dram_idx] = np.concatenate((self.filter_prefetch_matrices[dram_idx], this_fold_prefetch), axis=0)


    #
    def create_demand_matrices(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.create_ifmap_demand_mat()
        self.create_filter_demand_mat()
        self.create_ofmap_demand_mat()

        assert self.ifmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'IFMAP and Filter demands out of sync'
        assert self.ofmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'OFMAP and Filter demands out of sync'
        assert self.ifmap_demand_matrix.shape[1] == self.arr_row, 'IFMAP demands exceed the rows'
        assert self.filter_demand_matrix.shape[1] == self.arr_col,'Filter demands exceed the cols'
        assert self.ofmap_demand_matrix.shape[1] == self.arr_col, 'OFMAP demands exceed the cols'

        self.demand_mat_ready_flag = True

        # Save demand matrices to separate files for each DRAM
        demand_matrices_dir = "demand_matrices"
        os.makedirs(demand_matrices_dir, exist_ok=True)

        for dram_idx in range(self.num_drams):
            ifmap_file_path = os.path.join(demand_matrices_dir, f"ifmap_demand_matrix_dram{dram_idx}.npy")
            filter_file_path = os.path.join(demand_matrices_dir, f"filter_demand_matrix_dram{dram_idx}.npy")

            np.save(ifmap_file_path, self.ifmap_demand_matrices[dram_idx])
            np.save(filter_file_path, self.filter_demand_matrices[dram_idx])

        ofmap_file_path = os.path.join(demand_matrices_dir, "ofmap_demand_matrix.npy")
        np.save(ofmap_file_path, self.ofmap_demand_matrix)

    #
    def create_ifmap_demand_mat(self, dram_idx):
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_prefix = self.arr_row
        inter_fold_gap_prefix_mat = np.ones((inter_fold_gap_prefix, self.arr_row)) * -1

        inter_fold_gap_suffix = self.arr_col - 1

        inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_row)) * -1

        ifmap_op_mat = self.ifmap_op_mats[dram_idx]

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                col_start_id = fr * self.arr_row
                col_end_idx = min(col_start_id + self.arr_row, self.Sr)
                delta = self.arr_row - (col_end_idx - col_start_id)

                # Indexing the cols with row start and row end idx are correct
                # See the comment on ifmap_prefetch generation
                this_fold_demand = ifmap_op_mat[:, col_start_id: col_end_idx]
                self.ifmap_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Take into account underutilization
                if delta > 0:
                    null_req_mat = np.ones((self.T, delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                # Account for the cycles for weights to load
                this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand), axis=0)

                # Account for the cycles for final output to drain out
                this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat), axis=0)

                # Add skew to the IFMAP demand matrix to reflect systolic pipeline fill
                this_fold_demand = skew_matrix(this_fold_demand)

                self.ifmap_demand_matrices[dram_idx].append(this_fold_demand)

        self.ifmap_demand_matrices[dram_idx] = np.concatenate(self.ifmap_demand_matrices[dram_idx])

    #
    def create_filter_demand_mat(self, dram_idx):
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_suffix = self.arr_row + self.arr_col + self.T - 2
        inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_col)) * -1

        filter_op_mat = self.filter_op_mats[dram_idx]

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                row_start_id = fr * self.arr_row
                row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                row_delta = self.arr_row - (row_end_idx - row_start_id)

                col_start_id = fc * self.arr_col
                col_end_idx = min(col_start_id + self.arr_col, self.Sc)
                col_delta = self.arr_col - (col_end_idx - col_start_id)

                this_fold_demand = filter_op_mat[row_start_id: row_end_idx, col_start_id: col_end_idx]
                self.filter_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Take into account underutilization
                if col_delta > 0:
                    null_req_mat = np.ones((this_fold_demand.shape[0], col_delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                if row_delta > 0:
                    null_req_mat = np.ones((row_delta, self.arr_col)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)

                # The filters are needed to be filled in reverse order to ensure that
                # the top element is pushed in last to maintain alignment with the input elements
                this_fold_demand = np.flip(this_fold_demand, 0)

                # Time for inputs to stream and the partial sums to drain out
                this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat), axis=0)

                # Calculate the mapping efficiency
                row_used = min(self.arr_row, row_end_idx - row_start_id)
                col_used = min(self.arr_col, col_end_idx - col_start_id)
                mac_used = row_used * col_used
                mapping_eff_this_fold = mac_used / (self.arr_row * self.arr_col)

                cycles_this_fold = this_fold_demand.shape[0] + this_fold_demand.shape[1] - 1
                compute_cycles_this_fold = mac_used * self.T
                compute_util_this_fold = compute_cycles_this_fold / (self.arr_row * self.arr_col * cycles_this_fold)

                self.mapping_efficiency_per_fold.append(mapping_eff_this_fold)
                self.compute_utility_per_fold.append(compute_util_this_fold)

                self.filter_demand_matrices[dram_idx].append(this_fold_demand)

        self.filter_demand_matrices[dram_idx] = np.concatenate(self.filter_demand_matrices[dram_idx])


    #
    def create_ofmap_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_prefix = 2 * self.arr_row - 1
        inter_fold_gap_prefix_mat = np.ones((inter_fold_gap_prefix, self.arr_col)) * -1

        ofmap_demand_matrix_list = []
        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                col_start_id = fc * self.arr_col
                col_end_idx = min(col_start_id + self.arr_col, self.Sc)
                col_delta = self.arr_col - (col_end_idx - col_start_id)

                this_fold_demand = self.ofmap_op_mat[:, col_start_id: col_end_idx]
                self.ofmap_writes += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Adding null requests when there is under utilization ie. no mapping along a few rows or cols
                if col_delta > 0:
                    null_req_mat = np.ones((this_fold_demand.shape[0], col_delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                # Now add the prefix matrix
                # These are the null demands to account for when the operands are streamed in
                # and the OFMAPS are not ready
                this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand), axis=0)

                # Add skew to the OFMAP demand matrix to reflect systolic pipeline fill
                this_fold_demand = skew_matrix(this_fold_demand)

                ofmap_demand_matrix_list.append(this_fold_demand)
                #if fr == 0 and fc == 0:
                #    self.ofmap_demand_matrix = this_fold_demand
                #else:
                #    self.ofmap_demand_matrix = np.concatenate((self.ofmap_demand_matrix, this_fold_demand), axis=0)
        self.ofmap_demand_matrix = np.concatenate(ofmap_demand_matrix_list)
    # END of OFMAP demand generation

    #
    def get_ifmap_prefetch_mat(self, dram_idx):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.ifmap_prefetch_matrices[dram_idx]

    #
    def get_filter_prefetch_mat(self, dram_idx):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.filter_prefetch_matrices[dram_idx]


    #
    def get_prefetch_matrices(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.ifmap_prefetch_matrix, self.filter_prefetch_matrix

    #
    def get_ifmap_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ifmap_demand_matrix

    #
    def get_filter_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.filter_demand_matrix

    #
    def get_ofmap_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ofmap_demand_matrix

    #
    def get_demand_matrices(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return [self.ifmap_demand_matrices[dram_idx] for dram_idx in range(self.num_drams)], \
               [self.filter_demand_matrices[dram_idx] for dram_idx in range(self.num_drams)], \
               self.ofmap_demand_matrix

    #
    def get_avg_mapping_efficiency(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.mapping_efficiency_per_fold)
        num = len(self.mapping_efficiency_per_fold)

        avg_mapping_eff = agg / num

        return avg_mapping_eff

    #
    def get_avg_compute_utilization(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.compute_utility_per_fold)
        num = len(self.compute_utility_per_fold)

        avg_compute_util = agg / num

        return avg_compute_util

    #
    def get_ifmap_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ifmap_reads

    #
    def get_filter_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.filter_reads

    #
    def get_ofmap_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ofmap_writes


#
def skew_matrix(self, matrix):
        """
        Generates a matrix skewed along the diagonal
        """
        M, N = matrix.shape
        num_elems = M * N
        num_diags = M + N
        skewed_matrix = np.zeros((1, num_elems))
        idx = 0

        for diag_id in range(num_diags):
            max_row_id = min(diag_id, M - 1)
            min_row_id = max(0, diag_id - N + 1)
            valid_rows = max_row_id - min_row_id + 1

            for offset in range(valid_rows):
                row_id = max_row_id - offset
                col_id = diag_id - row_id

                elem = matrix[row_id][col_id]
                skewed_matrix[0, idx] = elem
                idx += 1

        return skewed_matrix
