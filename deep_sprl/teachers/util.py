class Buffer:

    def __init__(self, n_elements, max_buffer_size, reset_on_query):
        self.reset_on_query = reset_on_query
        self.max_buffer_size = max_buffer_size
        self.buffers = [list() for i in range(0, n_elements)]

    def update_buffer(self, datas):
        if isinstance(datas[0], list):
            for buffer, data in zip(self.buffers, datas):
                buffer.extend(data)
        else:
            for buffer, data in zip(self.buffers, datas):
                buffer.append(data)

        while len(self.buffers[0]) > self.max_buffer_size:
            for buffer in self.buffers:
                del buffer[0]

    def read_buffer(self, reset=None):
        if reset is None:
            reset = self.reset_on_query

        res = tuple([buffer for buffer in self.buffers])

        if reset:
            for i in range(0, len(self.buffers)):
                self.buffers[i] = []

        return res

    def __len__(self):
        return len(self.buffers[0])
