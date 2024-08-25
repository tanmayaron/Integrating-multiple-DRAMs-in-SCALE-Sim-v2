import time
import math
import numpy as np
from tqdm import tqdm
from memory.write_port import write_port

class write_buffer:
    def __init__(self):
        # Buffer properties: User specified
        self.total_size_bytes = 128
        self.word_size = 1
        self.active_buf_frac = 0.9

        # Buffer properties: Calculated
        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
        self.drain_buf_size = self.total_size_elems - self.active_buf_size

        # Backing interface properties
        self.backing_buffer = write_port()
        self.req_gen_bandwidth = 100

        # Status of the buffer
        self.free_space = self.total_size_elems
        self.drain_buf_start_line_id = 0
        self.drain_buf_end_line_id = 0

        # Helper data structures for faster execution
        self.line_idx = 0
        self.current_line1 = np.ones((1, 1)) * -1
        self.current_line2 = np.ones((1, 1)) * -1
        self.max_cache_lines = 2 ** 10
        self.trace_matrix_cache1 = np.zeros((1, 1))
        self.trace_matrix_cache2 = np.zeros((1, 1))

        # Access counts
        self.num_access = 0

        # Trace matrices
        self.trace_matrix1 = np.zeros((1, 1))
        self.trace_matrix2 = np.zeros((1, 1))
        self.cycles_vec = np.zeros((1, 1))

        # Flags
        self.state = 0
        self.drain_end_cycle = 0

        self.trace_valid = False
        self.trace_valid_2 = False
        self.trace_matrix_cache1_empty = True
        self.trace_matrix_cache2_empty = True
        self.trace_matrix_empty = True

    def set_params(self, backing_buf_obj,
                   total_size_bytes=128, word_size=1, active_buf_frac=0.9,
                   backing_buf_bw=100
                   ):
        self.total_size_bytes = total_size_bytes
        self.word_size = word_size

        assert 0.5 <= active_buf_frac < 1, "Valid active buf frac [0.5,1)"
        self.active_buf_frac = active_buf_frac

        self.backing_buffer = backing_buf_obj
        self.req_gen_bandwidth = backing_buf_bw

        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
        self.drain_buf_size = self.total_size_elems - self.active_buf_size
        self.free_space = self.total_size_elems

    def reset(self):
        self.total_size_bytes = 128
        self.word_size = 1
        self.active_buf_frac = 0.9

        self.backing_buffer = write_buffer()
        self.req_gen_bandwidth = 100

        self.free_space = self.total_size_elems
        self.active_buf_contents = []
        self.drain_buf_contents = []
        self.drain_end_cycle = 0

        self.trace_matrix = np.zeros((1, 1))

        self.num_access = 0
        self.state = 0

        self.trace_valid = False
        self.trace_matrix_cache_empty = True
        self.trace_matrix_empty = True

    def store_to_trace_mat_cache(self, elem):
        if elem == -1:
            return

        if self.current_line1.shape == (1, 1):  # This line is empty
            self.current_line1 = np.ones((1, self.req_gen_bandwidth)) * -1

        if self.current_line2.shape == (1, 1):  # This line is empty
            self.current_line2 = np.ones((1, self.req_gen_bandwidth)) * -1

        if elem % 2 == 0:  # Even addresses
            current_line = self.current_line1
        else:
            current_line = self.current_line2

        current_line[0, self.line_idx] = elem
        self.line_idx += 1
        self.free_space -= 1

        if not self.line_idx < self.req_gen_bandwidth:
            # Store to the cache matrix
            if self.trace_matrix_cache1_empty:
                self.trace_matrix_cache1 = self.current_line1
                self.trace_matrix_cache1_empty = False
            else:
                self.trace_matrix_cache1 = np.concatenate((self.trace_matrix_cache1, self.current_line1), axis=0)

            if self.trace_matrix_cache2_empty:
                self.trace_matrix_cache2 = self.current_line2
                self.trace_matrix_cache2_empty = False
            else:
                self.trace_matrix_cache2 = np.concatenate((self.trace_matrix_cache2, self.current_line2), axis=0)

            self.current_line1 = np.ones((1, 1)) * -1
            self.current_line2 = np.ones((1, 1)) * -1
            self.line_idx = 0

            if not self.trace_matrix_cache1.shape[0] < self.max_cache_lines:
                self.append_to_trace_mat()

            if not self.trace_matrix_cache2.shape[0] < self.max_cache_lines:
                self.append_to_trace_mat(force=True)

    def append_to_trace_mat(self, force=False):
        if force:
            if not self.line_idx == 0:
                if self.trace_matrix_cache1_empty:
                    self.trace_matrix_cache1 = self.current_line1
                    self.trace_matrix_cache2 = self.current_line2
                    self.trace_matrix_cache_empty = False
                else:
                    self.trace_matrix_cache1 = np.concatenate((self.trace_matrix_cache1, self.current_line1), axis=0)
                    self.trace_matrix_cache2 = np.concatenate((self.trace_matrix_cache2, self.current_line2), axis=0)

                self.current_line1 = np.ones((1, 1)) * -1
                self.current_line2 = np.ones((1, 1)) * -1
                self.line_idx = 0

        if self.trace_matrix_cache1_empty:
            return

        if self.trace_matrix_empty:
            self.trace_matrix1 = self.trace_matrix_cache1
            self.trace_matrix2 = self.trace_matrix_cache2
            self.drain_buf_start_line_id = 0
            self.trace_matrix_empty = False
        else:
            self.trace_matrix1 = np.concatenate((self.trace_matrix1, self.trace_matrix_cache1), axis=0)
            self.trace_matrix2 = np.concatenate((self.trace_matrix2, self.trace_matrix_cache2), axis=0)

        self.trace_matrix_cache1 = np.zeros((1, 1))
        self.trace_matrix_cache2 = np.zeros((1, 1))
        self.trace_matrix_cache_empty = True

    def service_writes(self, incoming_requests_arr_np, incoming_cycles_arr_np):
        assert incoming_cycles_arr_np.shape[0] == incoming_requests_arr_np.shape[0], 'Cycles and requests do not match'
        out_cycles_arr = []
        offset = 0

        for i in tqdm(range(incoming_requests_arr_np.shape[0]), disable=True):
            row = incoming_requests_arr_np[i]
            cycle = incoming_cycles_arr_np[i]
            current_cycle = cycle[0] + offset

            for elem in row:
                if elem == -1:
                    continue

                self.store_to_trace_mat_cache(elem)

                if current_cycle < self.drain_end_cycle:
                    if not self.free_space > 0:
                        offset += max(self.drain_end_cycle - current_cycle, 0)
                        current_cycle = self.drain_end_cycle

                elif self.free_space < (self.total_size_elems - self.drain_buf_size):
                    self.append_to_trace_mat(force=True)
                    self.drain_end_cycle = self.empty_drain_buf(empty_start_cycle=current_cycle)

            out_cycles_arr.append(current_cycle)

        num_lines = incoming_requests_arr_np.shape[0]
        out_cycles_arr_np = np.asarray(out_cycles_arr).reshape((num_lines, 1))

        return out_cycles_arr_np

    def empty_drain_buf(self, empty_start_cycle=0):
        # Calculate lines to fill buffer for each trace matrix
        lines_to_fill_dbuf_1 = int(math.ceil(self.drain_buf_size / self.req_gen_bandwidth / 2))
        lines_to_fill_dbuf_2 = int(math.ceil(self.drain_buf_size / self.req_gen_bandwidth / 2))

        # Adjust end line IDs based on calculated values
        self.drain_buf_end_line_id_1 = self.drain_buf_start_line_id + lines_to_fill_dbuf_1
        self.drain_buf_end_line_id_1 = min(self.drain_buf_end_line_id_1, self.trace_matrix1.shape[0])

        self.drain_buf_end_line_id_2 = self.drain_buf_start_line_id + lines_to_fill_dbuf_2
        self.drain_buf_end_line_id_2 = min(self.drain_buf_end_line_id_2, self.trace_matrix2.shape[0])

        # Retrieve requests and cycles for each trace matrix
        requests_arr_np1 = self.trace_matrix1[self.drain_buf_start_line_id: self.drain_buf_end_line_id_1, :]
        requests_arr_np2 = self.trace_matrix2[self.drain_buf_start_line_id: self.drain_buf_end_line_id_2, :]

        num_lines_1 = requests_arr_np1.shape[0]
        num_lines_2 = requests_arr_np2.shape[0]

        # Calculate data size to drain for each trace matrix
        data_sz_to_drain_1 = num_lines_1 * requests_arr_np1.shape[1]
        for elem in requests_arr_np1[-1, :]:
            if elem == -1:
                data_sz_to_drain_1 -= 1

        data_sz_to_drain_2 = num_lines_2 * requests_arr_np2.shape[1]
        for elem in requests_arr_np2[-1, :]:
            if elem == -1:
                data_sz_to_drain_2 -= 1

        # Update num_access for each trace matrix
        self.num_access += data_sz_to_drain_1
        self.num_access += data_sz_to_drain_2

        # Service writes for each trace matrix
        cycles_arr_1 = [x + empty_start_cycle for x in range(num_lines_1)]
        cycles_arr_np_1 = np.asarray(cycles_arr_1).reshape((num_lines_1, 1))
        serviced_cycles_arr_1 = self.backing_buffer.service_writes(requests_arr_np1, cycles_arr_np_1)

        cycles_arr_2 = [x + empty_start_cycle for x in range(num_lines_2)]
        cycles_arr_np_2 = np.asarray(cycles_arr_2).reshape((num_lines_2, 1))
        serviced_cycles_arr_2 = self.backing_buffer.service_writes(requests_arr_np2, cycles_arr_np_2)

        # Update cycles vectors for each trace matrix
        if not self.trace_valid:
            self.cycles_vec = serviced_cycles_arr_1
            self.trace_valid = True
        else:
            self.cycles_vec = np.concatenate((self.cycles_vec, serviced_cycles_arr_1), axis=0)

        if not self.trace_valid_2:
            self.cycles_vec_2 = serviced_cycles_arr_2
            self.trace_valid_2 = True
        else:
            self.cycles_vec_2 = np.concatenate((self.cycles_vec_2, serviced_cycles_arr_2), axis=0)

        # Calculate service end cycle for each trace matrix
        service_end_cycle_1 = serviced_cycles_arr_1[-1][0]
        service_end_cycle_2 = serviced_cycles_arr_2[-1][0]

        # Update free space and start line IDs for each trace matrix
        self.free_space += data_sz_to_drain_1
        self.free_space += data_sz_to_drain_2

        self.drain_buf_start_line_id = max(self.drain_buf_end_line_id_1, self.drain_buf_end_line_id_2)

        return service_end_cycle_1

    def empty_all_buffers(self, cycle):
        self.append_to_trace_mat(force=True)

        if self.trace_matrix_empty:
            return

        while self.drain_buf_start_line_id < self.trace_matrix1.shape[0]:
            self.drain_end_cycle = self.empty_drain_buf(empty_start_cycle=cycle)
            cycle = self.drain_end_cycle + 1

    def get_trace_matrix1(self):
        if not self.trace_valid:
            print('No trace has been generated yet')
            return

        trace_matrix1 = np.concatenate((self.cycles_vec, self.trace_matrix1), axis=1)
        return trace_matrix1

    def get_trace_matrix2(self):
        if not self.trace_valid:
            print('No trace has been generated yet')
            return

        trace_matrix2 = np.concatenate((self.cycles_vec, self.trace_matrix2), axis=1)
        return trace_matrix2

    def get_free_space(self):
        return self.free_space

    def get_num_accesses(self):
        assert self.trace_valid, 'Traces not ready yet'
        return self.num_access

    def get_external_access_start_stop_cycles(self):
        assert self.trace_valid, 'Traces not ready yet'
        start_cycle = self.cycles_vec[0][0]
        end_cycle = self.cycles_vec[-1][0]
        return start_cycle, end_cycle

    def print_trace1(self, filename):
        if not self.trace_valid:
            print('No trace has been generated yet')
            return
        trace_matrix1 = self.get_trace_matrix1()
        np.savetxt(filename, trace_matrix1, fmt='%s', delimiter=",")

    def print_trace2(self, filename):
        if not self.trace_valid:
            print('No trace has been generated yet')
            return
        trace_matrix2 = self.get_trace_matrix2()
        np.savetxt(filename, trace_matrix2, fmt='%s', delimiter=",")
