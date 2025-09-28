import numpy as np

class LiteralCode:
    def __init__(self):
        self.lit_num = 1
        self.aux_block_num = 0
        self.alloc = {}

    def append(self, lit_name, shape):
        if lit_name not in self.alloc.keys():
            size = np.prod(shape)
            self.alloc[lit_name] = (self.lit_num, size, shape)
            self.lit_num += size

    def auxiliary(self, shape):
        self.aux_block_num += 1
        lit_name = "_aux_lit_%d" % self.aux_block_num
        size = np.prod(shape)
        self.alloc[lit_name] = (self.lit_num, size, shape)
        self.lit_num += size
        return lit_name
    
    def encode(self, lit, value):
        lit_name = lit[0]
        indices = lit[1:]
        shifted, _, shape = self.alloc[lit_name]
        pos = np.ravel_multi_index(indices, shape) + shifted
        if not value:
            pos = -pos
        return int(pos)

    def decode(self, pos, value):
        for lit_name, (shifted, size, shape) in self.alloc.items():
            if shifted <= pos < shifted + size:
                return (value > 0, lit_name, np.unravel_index(pos - shifted, shape))
        raise ValueError("Literal is out of range")

    def reset(self):
        self.lit_num = 1
        self.temp_num = 0
        self.alloc = {}