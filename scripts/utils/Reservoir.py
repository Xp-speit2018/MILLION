# Implements reservoir sampling algorithm to sample from a stream of torch vectors.
import torch
import random

class Reservoir:
    def __init__(self, max_size=100, device='cuda:0', dim=128, dtype=torch.float16, name=''):
        self.name = name
        self.max_size = max_size
        self.reservoir = torch.zeros((max_size, dim), device=device, dtype=dtype)
        self.count = 0
        
    def __del__(self):
        # Clean up the reservoir
        if hasattr(self, 'reservoir'):
            del self.reservoir
        
    def add(self, x):
        assert x.size(0) == 1
        assert x.size(1) == self.reservoir.size(1)
        
        if self.count < self.max_size:
            self.reservoir[self.count] = x
            
        else:
            idx = random.randint(0, self.count)
            if idx < self.max_size:
                self.reservoir[idx] = x
            
        self.count += 1

    # def batch_add(self, xs):
    #     n = xs.size(0)
    #     assert xs.size(1) == self.reservoir.size(1)
        
    #     for i in range(n):
    #         self.add(xs[i:i+1])
               
    def batch_add(self, xs):
        n = xs.size(0)
        assert xs.size(1) == self.reservoir.size(1)
        
        # Remaining space in the reservoir
        remaining = self.max_size - self.count
        
        if remaining > 0:
            # If there is space, add as many as possible
            add_n = min(n, remaining)
            end = self.count + add_n
            self.reservoir[self.count:end] = xs[:add_n]
            self.count += add_n
            xs = xs[add_n:]  # Remaining samples to process
            n = xs.size(0)
            
            if n == 0:
                return  # Done adding
        
        # Now, the reservoir is full, perform reservoir sampling for the remaining samples
        # Probability that each new sample replaces an existing one
        p = self.max_size / (self.count + n)
        
        # Generate a random mask with probability p for each sample in xs
        # To decide which samples will replace reservoir elements
        replace_mask = torch.rand(n, device=xs.device) < p
        num_replace = replace_mask.sum().int().item()
        
        if num_replace > 0:
            # Select num_replace samples from xs
            replace_indices = replace_mask.nonzero(as_tuple=True)[0]
            selected_xs = xs[replace_indices[:num_replace]]
            
            # Randomly select reservoir indices to replace
            reservoir_indices = torch.randint(0, self.max_size, (num_replace,), device=xs.device)
            self.reservoir[reservoir_indices] = selected_xs
        
        self.count += n