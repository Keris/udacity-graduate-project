"""Replay Buffer."""

import random

from collections import namedtuple

Experience = namedtuple("Experience",
    field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        self.size = size  # maximum size of buffer
        self.memory = []  # internal memory (list)
        self.idx = 0  # current index into circular buffer

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Create an Experience object, add it to memory
        # Note: If memory is full, start overwriting from the beginning
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.idx] = Experience(state, action, reward, next_state, done)
        self.idx = (self.idx + 1) % self.size

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        # Return a list or tuple of Experience objects sampled from memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
