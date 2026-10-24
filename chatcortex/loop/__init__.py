"""The tradeoff-driven closed loop: synthesize -> execute -> calibrate.

Re-exports the loop entry points so users can start from the package::

    from chatcortex.loop import run_closed_loop
"""

from .closed_loop import LoopIteration, run_closed_loop

__all__ = ["LoopIteration", "run_closed_loop"]
