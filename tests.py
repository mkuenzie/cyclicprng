import threading
import timeit

from cyclicprng import CyclicPRNG


def test_random_cycle():
    """
    Ensure that default cycles are randomly ordered
    """
    cycle_size = 10
    c = CyclicPRNG(cycle_size)
    first_cycle = [c.get_random() for _ in range(cycle_size)]
    second_cycle = [c.get_random() for _ in range(cycle_size)]
    assert first_cycle != second_cycle


def test_consistent_cycle():
    """
    Ensure that consistent cycles are in the same order
    """
    cycle_size = 10
    c = CyclicPRNG(cycle_size, consistent=True)
    first_cycle = [c.get_random() for _ in range(cycle_size)]
    second_cycle = [c.get_random() for _ in range(cycle_size)]
    assert first_cycle == second_cycle


def test_cycle_init_time():
    """
    Sanity checking that the prng can be initialized in a reasonable time.
    ip4_addr_space = 4294967296
    ip6_addr_space = 340282366920938463463374607431768211456
    """
    ip4_timer = timeit.Timer(
        "CyclicPRNG(4294967296)", setup="from cyclicprng import CyclicPRNG"
    )
    ip6_timer = timeit.Timer(
        "CyclicPRNG(340282366920938463463374607431768211456)",
        setup="from cyclicprng import CyclicPRNG",
    )
    assert all(i < 0.5 for i in ip4_timer.repeat(10, 1))
    assert all(i < 0.5 for i in ip6_timer.repeat(10, 1))


def test_size_one_cycle():
    """
    Edge case: In cycles of size one, [1] is the only valid cycle
    """
    cycle_size = 1
    c = CyclicPRNG(cycle_size)
    first_cycle = [c.get_random() for _ in range(cycle_size)]
    second_cycle = [c.get_random() for _ in range(cycle_size)]
    assert len(first_cycle) == len(second_cycle) == 1
    assert first_cycle == second_cycle == [1]


def test_size_two_cycle():
    """
    Edge case: In cycles of size 2, we expect results to always alternate
    i.e. [1,2,1,2] or [2,1,2,1] instead of [1,2,2,1] or [2,1,1,2]
    """
    cycle_size = 2
    c = CyclicPRNG(cycle_size)
    first_cycle = [c.get_random() for _ in range(cycle_size)]
    second_cycle = [c.get_random() for _ in range(cycle_size)]
    assert len(first_cycle) == len(second_cycle) == 2
    assert first_cycle == second_cycle


def test_event_handler_exception_does_not_wedge_generator():
    """
    A handler that raises must not leave the generator unusable.

    The exception is still the caller's to deal with, but the generator has to
    keep working for everyone else.
    """
    cycle_size = 4

    def boom(event):
        if event == "restart":
            raise RuntimeError("handler failed")

    c = CyclicPRNG(cycle_size, event_handler=boom)

    values, errors = [], []

    def drive():
        for _ in range(cycle_size * 3):
            try:
                values.append(c.get_random())
            except RuntimeError as exc:
                errors.append(exc)

    driver = threading.Thread(target=drive, daemon=True)
    driver.start()
    driver.join(timeout=10)

    assert not driver.is_alive(), "generator deadlocked after a handler raised"
    assert errors, "the handler exception should still reach the caller"
    assert len(values) + len(errors) == cycle_size * 3


def test_slow_handler_does_not_block_other_generators():
    """
    One generator's handler must not stall an unrelated generator.
    """
    release = threading.Event()
    entered = threading.Event()

    def slow(event):
        if event == "restart":
            entered.set()
            release.wait(timeout=10)

    blocking = CyclicPRNG(3, event_handler=slow)
    unrelated = CyclicPRNG(1000)

    spinner = threading.Thread(
        target=lambda: [blocking.get_random() for _ in range(6)], daemon=True
    )
    spinner.start()
    assert entered.wait(timeout=10), "the slow handler never ran"

    probe = threading.Thread(target=unrelated.get_random, daemon=True)
    probe.start()
    probe.join(timeout=5)
    blocked = probe.is_alive()

    release.set()
    spinner.join(timeout=10)

    assert not blocked, "an unrelated generator blocked behind another's handler"


def test_generators_have_independent_locks():
    assert CyclicPRNG(10).mutex is not CyclicPRNG(10).mutex


def test_restart_event_precedes_final_value():
    """
    The documented ordering: restart fires before the cycle's last value is
    returned to the caller.
    """
    cycle_size = 5
    seen = []
    c = CyclicPRNG(cycle_size, event_handler=seen.append)
    for _ in range(cycle_size):
        seen.append(c.get_random())

    assert seen[0] == "init"
    assert "restart" in seen
    assert seen.index("restart") == len(seen) - 2, (
        "restart must fire immediately before the final value of the cycle"
    )
