import time
import math
import numpy as np
from tqdm import tqdm
from scalesim.memory.write_port import write_port

class write_buffer:
    def __init__(self):
        # Buffer properties: User specified
        self.alt=0
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
        self.free_space1 = self.total_size_elems
        self.free_space2 = self.total_size_elems
        self.drain_buf_start_line_id1 = 0
        self.drain_buf_start_line_id2 = 0
        self.drain_buf_end_line_id1 = 0
        self.drain_buf_end_line_id2 = 0

        # Helper data structures for faster execution
        self.line_idx1 = 0
        self.line_idx2 = 0
        self.current_line1 = np.ones((1, 1)) * -1
        self.current_line2 = np.ones((1, 1)) * -1
        self.max_cache_lines = 2 ** 10              
        self.trace_matrix1_cache = np.zeros((1, 1))
        self.trace_matrix2_cache = np.zeros((1, 1))

        # Access counts
        self.num_access = 0

        # Trace matrix
        self.trace_matrix1 = np.zeros((1, 1))
        self.trace_matrix2 = np.zeros((1, 1))
        self.cycles_vec1 = np.zeros((1, 1))
        self.cycles_vec2 = np.zeros((1, 1))

        # Flags
        self.state = 0
        self.drain_end_cycle1 = 0
        self.drain_end_cycle2 = 0

        self.trace_valid1 = False
        self.trace_valid2 = False

        self.trace_matrix1_cache_empty = True
        self.trace_matrix2_cache_empty = True
        self.trace_matrix1_empty = True
        self.trace_matrix2_empty = True

    def set_params(self, backing_buf_obj, total_size_bytes=128, word_size=1, active_buf_frac=0.9, backing_buf_bw=100):
        self.total_size_bytes = total_size_bytes
        self.word_size = word_size

        assert 0.5 <= active_buf_frac < 1, "Valid active buf frac [0.5,1)"
        self.active_buf_frac = active_buf_frac

        self.backing_buffer = backing_buf_obj
        self.req_gen_bandwidth = backing_buf_bw

        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
        self.drain_buf_size = self.total_size_elems - self.active_buf_size
        self.free_space1 = self.total_size_elems
        self.free_space2 = self.total_size_elems

    def reset(self):
        self.alt=0
        self.total_size_bytes = 128
        self.word_size = 1
        self.active_buf_frac = 0.9

        self.backing_buffer = write_buffer()
        self.req_gen_bandwidth = 100

        self.free_space1 = self.total_size_elems
        self.free_space2 = self.total_size_elems
        self.active_buf_contents = []
        self.drain_buf_contents = []
        self.drain_end_cycle1 = 0
        self.drain_end_cycle2 = 0

        self.trace_matrix1 = np.zeros((1, 1))
        self.trace_matrix2 = np.zeros((1, 1))

        self.num_access = 0
        self.state = 0

        self.trace_valid1 = False
        self.trace_valid2 = False

        self.trace_matrix1_cache_empty = True
        self.trace_matrix2_cache_empty = True
        self.trace_matrix1_empty = True
        self.trace_matrix2_empty = True

    def store_to_trace_mat1_cache(self, elem):
        if elem == -1:
            return

        if self.current_line1.shape == (1,1):
            self.current_line1 = np.ones((1, self.req_gen_bandwidth)) * -1

        self.current_line1[0, self.line_idx1] = elem
        self.line_idx1 += 1
        self.free_space1 -= 1

        if not self.line_idx1 < self.req_gen_bandwidth:
            if self.trace_matrix1_cache_empty:
                self.trace_matrix1_cache = self.current_line1
                self.trace_matrix1_cache_empty = False
            else:
                self.trace_matrix1_cache = np.concatenate((self.trace_matrix1_cache, self.current_line1), axis=0)

            self.current_line1 = np.ones((1,1)) * -1
            self.line_idx1 = 0

            if not self.trace_matrix1_cache.shape[0] < self.max_cache_lines:
                self.append_to_trace_mat1()
        
    def store_to_trace_mat2_cache(self, elem):
        if elem == -1:
            return

        if self.current_line2.shape == (1,1):
            self.current_line2 = np.ones((1, self.req_gen_bandwidth)) * -1

        self.current_line2[0, self.line_idx2] = elem
        self.line_idx2 += 1
        self.free_space2 -= 1

        if not self.line_idx2 < self.req_gen_bandwidth:
            if self.trace_matrix2_cache_empty:
                self.trace_matrix2_cache = self.current_line2
                self.trace_matrix2_cache_empty = False
            else:
                self.trace_matrix2_cache = np.concatenate((self.trace_matrix2_cache, self.current_line2), axis=0)

            self.current_line2 = np.ones((1,1)) * -1
            self.line_idx2 = 0

            if not self.trace_matrix2_cache.shape[0] < self.max_cache_lines:
                self.append_to_trace_mat2()

    def append_to_trace_mat1(self, force=False):
        if force:
            if not self.line_idx1 == 0:
                if self.trace_matrix1_cache_empty:
                    self.trace_matrix1_cache = self.current_line1
                    self.trace_matrix1_cache_empty = False
                else:
                    self.trace_matrix1_cache = np.concatenate((self.trace_matrix1_cache, self.current_line1), axis=0)

                self.current_line1 = np.ones((1,1)) * -1
                self.line_idx1 = 0

        if self.trace_matrix1_cache_empty:
            return

        if self.trace_matrix1_empty:
            self.trace_matrix1 = self.trace_matrix1_cache
            self.drain_buf_start_line_id1 = 0
            self.trace_matrix1_empty = False
        else:
            self.trace_matrix1 = np.concatenate((self.trace_matrix1, self.trace_matrix1_cache), axis=0)

        self.trace_matrix1_cache = np.zeros((1,1))
        self.trace_matrix1_cache_empty = True

    def append_to_trace_mat2(self, force=False):
        if force:
            if not self.line_idx2 == 0:
                if self.trace_matrix2_cache_empty:
                    self.trace_matrix2_cache = self.current_line2
                    self.trace_matrix2_cache_empty = False
                else:
                    self.trace_matrix2_cache = np.concatenate((self.trace_matrix2_cache, self.current_line2), axis=0)

                self.current_line2 = np.ones((1,1)) * -1
                self.line_idx2 = 0

        if self.trace_matrix2_cache_empty:
            return

        if self.trace_matrix2_empty:
            self.trace_matrix2 = self.trace_matrix2_cache
            self.drain_buf_start_line_id2 = 0
            self.trace_matrix2_empty = False
        else:
            self.trace_matrix2 = np.concatenate((self.trace_matrix2, self.trace_matrix2_cache), axis=0)

        self.trace_matrix2_cache = np.zeros((1,1))
        self.trace_matrix2_cache_empty = True

    def service_writes(self, incoming_requests_arr_np, incoming_cycles_arr_np):
        assert incoming_cycles_arr_np.shape[0] == incoming_requests_arr_np.shape[0], 'Cycles and requests do not match'
        out_cycles_arr = []
        offset = 0

        DEBUG_num_drains = 0
        DEBUG_append_to_trace_times = []

        

        for i in tqdm(range(incoming_requests_arr_np.shape[0]), disable=True):
            row = incoming_requests_arr_np[i]
            cycle = incoming_cycles_arr_np[i]
            current_cycle = cycle[0] + offset

            for elem in row:
                # Pay no attention to empty requests
                if elem == -1:
                    continue

                if(self.alt==0):
                    self.alt=1
                    self.store_to_trace_mat1_cache(elem)
                else:
                    self.alt=0
                    self.store_to_trace_mat2_cache(elem)

                if current_cycle < max(self.drain_end_cycle1, self.drain_end_cycle2):
                    if not (self.free_space1 > 0 and self.free_space2 > 0):  # Update condition here
                        offset += max(max(self.drain_end_cycle1, self.drain_end_cycle2) - current_cycle, 0)
                        current_cycle = max(self.drain_end_cycle1, self.drain_end_cycle2)

                elif (self.free_space1 < (self.total_size_elems - self.drain_buf_size)) and (self.free_space2 < (self.total_size_elems - self.drain_buf_size)):
                    self.append_to_trace_mat1(force=True)
                    self.append_to_trace_mat2(force=True)
                    self.drain_end_cycle1, self.drain_end_cycle2 = self.empty_drain_buf(empty_start_cycle=current_cycle)

            out_cycles_arr.append(current_cycle)

        num_lines = incoming_requests_arr_np.shape[0]
        out_cycles_arr_np = np.asarray(out_cycles_arr).reshape((num_lines, 1))

        return out_cycles_arr_np


    def empty_drain_buf(self, empty_start_cycle=0):
        lines_to_fill_dbuf = int(math.ceil(self.drain_buf_size / self.req_gen_bandwidth))
        self.drain_buf_end_line_id1 = self.drain_buf_start_line_id1 + lines_to_fill_dbuf
        self.drain_buf_end_line_id1 = min(self.drain_buf_end_line_id1, self.trace_matrix1.shape[0])
        self.drain_buf_end_line_id2 = self.drain_buf_start_line_id2 + lines_to_fill_dbuf
        self.drain_buf_end_line_id2 = min(self.drain_buf_end_line_id2, self.trace_matrix2.shape[0])

        requests_arr_np1 = self.trace_matrix1[self.drain_buf_start_line_id1: self.drain_buf_end_line_id1, :]
        requests_arr_np2 = self.trace_matrix2[self.drain_buf_start_line_id2: self.drain_buf_end_line_id2, :]
        num_lines1 = requests_arr_np1.shape[0]
        num_lines2 = requests_arr_np2.shape[0]

        data_sz_to_drain1 = num_lines1 * requests_arr_np1.shape[1]
        data_sz_to_drain2 = num_lines2 * requests_arr_np2.shape[1]
        for elem in requests_arr_np1[-1,:]:
            if elem == -1:
                data_sz_to_drain1 -= 1
        for elem in requests_arr_np2[-1,:]:
            if elem == -1:
                data_sz_to_drain2 -= 1
        #self.num_access += data_sz_to_drain1 + data_sz_to_drain2
        self.num_access += data_sz_to_drain1 

        cycles_arr1 = [x+empty_start_cycle for x in range(num_lines1)]
        cycles_arr_np1 = np.asarray(cycles_arr1).reshape((num_lines1, 1))
        serviced_cycles_arr1 = self.backing_buffer.service_writes(requests_arr_np1, cycles_arr_np1)

        cycles_arr2 = [x+empty_start_cycle for x in range(num_lines2)]
        cycles_arr_np2 = np.asarray(cycles_arr2).reshape((num_lines2, 1))
        serviced_cycles_arr2 = self.backing_buffer.service_writes(requests_arr_np2, cycles_arr_np2)

        if not self.trace_valid1:
            self.cycles_vec1 = serviced_cycles_arr1
            self.trace_valid1 = True
        else:
            self.cycles_vec1 = np.concatenate((self.cycles_vec1, serviced_cycles_arr1), axis=0)

        if not self.trace_valid2:
            self.cycles_vec2 = serviced_cycles_arr2
            self.trace_valid2 = True
        else:
            self.cycles_vec2 = np.concatenate((self.cycles_vec2, serviced_cycles_arr2), axis=0)

        service_end_cycle1 = serviced_cycles_arr1[-1][0]
        service_end_cycle2 = serviced_cycles_arr2[-1][0]
        self.free_space1 += data_sz_to_drain1
        self.free_space2 += data_sz_to_drain2

        self.drain_buf_start_line_id1 = self.drain_buf_end_line_id1
        self.drain_buf_start_line_id2 = self.drain_buf_end_line_id2
        return service_end_cycle1, service_end_cycle2


    def empty_all_buffers(self, cycle):
        self.append_to_trace_mat1(force=True)
        self.append_to_trace_mat2(force=True)

        if self.trace_matrix1_empty or self.trace_matrix2_empty:
           return

        while self.drain_buf_start_line_id1 < self.trace_matrix1.shape[0] and self.drain_buf_start_line_id2 < self.trace_matrix2.shape[0]:
            self.drain_end_cycle1, self.drain_end_cycle2 = self.empty_drain_buf(empty_start_cycle=cycle)
            cycle = max(self.drain_end_cycle1, self.drain_end_cycle2) + 1

    def get_trace_matrix1(self):
        if not (self.trace_valid1 and self.trace_valid2):
            print('No trace has been generated yet')
            return

        trace_matrix1 = np.concatenate((self.cycles_vec1, self.trace_matrix1), axis=1)

        return trace_matrix1
    
    def get_trace_matrix2(self):
        if not (self.trace_valid1 and self.trace_valid2):
            print('No trace has been generated yet')
            return

        trace_matrix2 = np.concatenate((self.cycles_vec2, self.trace_matrix2), axis=1)

        return trace_matrix2

    def get_free_space(self):

        return max(self.free_space1, self.free_space2)

    def get_num_accesses(self):
        #assert self.trace_valid1, 'Traces not ready yet'
        return self.num_access

    def get_external_access_start_stop_cycles(self):
        #assert self.trace_valid1, 'Traces not ready yet'
        start_cycle = self.cycles_vec1[0][0]
        end_cycle = self.cycles_vec1[-1][0]

        return start_cycle, end_cycle

    def print_trace1(self, filename):
        #if not self.trace_valid1:
        #    print('No trace has been generated yet')
        #    return
        trace_matrix1 = self.get_trace_matrix1()
        np.savetxt(filename, trace_matrix1, fmt='%s', delimiter=",")


    def print_trace2(self, filename):
        #if not self.trace_valid1:
        #    print('No trace has been generated yet')
        #    return
        trace_matrix2 = self.get_trace_matrix2()
        np.savetxt(filename, trace_matrix2, fmt='%s', delimiter=",")

