import random
from datetime import datetime
from threading import Lock
from typing import Optional

import sympy


def modexp(b, e, m):
    bits = [(e >> bit) & 1 for bit in range(e.bit_length())]
    s = b
    v = 1
    for bit in bits:
        if bit == 1:
            v *= s
            v %= m
        s *= s
        s %= m
    return v


class CyclicPRNG:
    """
    For more information about the existence of this class and why it does what it does,
    see https://github.com/natlas/natlas/wiki/Host-Coverage-Scanning-Strategy
    Edge case behavior:
        size < 1 -> Raise ValueError
        size = 1 -> Always return 1
        size = 2 -> The generator is always 2 (equivalent to consistent=true)
    """

    size: int
    modulus: int
    modulus_factors: dict
    generator: int
    start: int
    end: int
    current: int
    cycle_start_time: datetime
    completed_cycle_count: int
    consistent: bool
    mutex: Lock

    def __init__(
        self,
        cycle_size: int,
        consistent: bool = False,
        event_handler: Optional[callable] = None,
    ):
        """
        Initialize PRNG that restarts after cycle_size calls
        """
        self.notify = []
        # Each generator guards only its own cycle state. A lock shared between
        # instances would make every generator in the process contend with
        # every other one for no reason.
        self.mutex = Lock()
        self.modulus = 0
        self.modulus_factors: dict
        self.generator = 0
        self.start = 0
        self.end = 0
        self.current = 0
        self.cycle_start_time = None
        self.completed_cycle_count = 0
        self.consistent = False
        self.modulus_factors = {}
        self.size = cycle_size
        if self.size < 1:
            raise ValueError(
                "Random Number Generator must be given a positive non-zero integer"
            )
        if event_handler:
            self.register_event_handler(event_handler)

        self._init_cyclic_group()
        self._init_generator()
        self._init_permutation()
        self.cycle_start_time = datetime.utcnow()
        self.consistent = consistent
        self._emit_event("init")

    def register_event_handler(self, func: callable):
        """
        A simple event handler to be notified of events in the cyclic PRNG
        func must be a callable that takes a single parameter
        """
        if not callable(func):
            raise TypeError("Event handler must be callable")
        self.notify.append(func)

    def unregister_event_handler(self, func: callable):
        """
        Removes an event handler from the cyclic PRNG
        """
        if func in self.notify:
            self.notify.remove(func)

    def clear_event_handlers(self):
        """
        Clear all event handlers from the cyclic PRNG
        """
        self.notify.clear()

    def _emit_event(self, event):
        for func in self.notify:
            func(event)

    def _init_cyclic_group(self):
        def next_prime(num):
            num = num + 1 if (num % 2) == 0 else num + 2
            while not sympy.isprime(num):
                num = num + 2
            return num

        self.modulus = next_prime(self.size)
        self.modulus_factors = sympy.factorint(self.modulus - 1)

    def _init_generator(self):
        """
        Find a generator for the whole cyclic group.
        """
        found = False
        base = 0
        """
            If modulus is 3 then the only generator we can use is 2.
            Otherwise we infinite loop because always self.generator == base
        """
        if self.modulus == 3:
            self.generator = 2
            return
        while not found:
            base = random.randint(2, self.modulus - 1)
            found = self.generator != base and all(
                modexp(base, int((self.modulus - 1) / factor), self.modulus) != 1
                for factor in self.modulus_factors
            )
        self.generator = base

    def _cycle_until_in_range(self, element):
        """
        Cycle the element until it is self.size or less
        """
        while element > self.size:
            element = (element * self.generator) % self.modulus
        return element

    def _init_permutation(self):
        """
        Create a new permutation of the scope
        """
        exp = random.randint(2, self.modulus - 1)
        self.end = self._cycle_until_in_range(modexp(self.generator, exp, self.modulus))
        self.start = self._cycle_until_in_range(
            (self.end * self.generator) % self.modulus
        )
        self.current = self.start

    def _restart_cycle(self):
        """
        Restart PRNG Cycle

        Does not emit the restart event. The caller is responsible for emitting
        it once the lock has been released, so that handlers never run while
        other callers are blocked behind them.
        """
        if self.consistent:
            self.current = self.start
        else:
            self._init_generator()
            self._init_permutation()
        self.cycle_start_time = datetime.utcnow()
        self.completed_cycle_count += 1

    def get_random(self):
        """
        Gets the next random number from the permutation
        """
        if self.size <= 1:
            return 1
        with self.mutex:
            value = self.current
            self.current = self._cycle_until_in_range(
                (self.current * self.generator) % self.modulus
            )
            restarted = value == self.end
            if restarted:
                self._restart_cycle()
        # Emitted after the lock is released. A handler may be slow or may
        # raise, and neither should stall or deadlock other callers. The event
        # still fires before the final value of the cycle is returned.
        if restarted:
            self._emit_event("restart")
        return value
