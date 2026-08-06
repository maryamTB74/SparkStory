"""Offline measurement of a finished book.

Deliberately not a critic. Nothing here feeds a revision loop -- a measurement that
also steers generation cannot be used to judge whether generation improved, which
is the whole reason this package sits outside ``nodes/``.

Two halves, kept apart on purpose. The computed metrics in
``metrics/deterministic.py`` are arithmetic over a ``Story``: free, offline, and
impossible for a model to game. The judge in ``metrics/judge.py`` is a model call,
so it can fail and it can drift. ``BookScorecard`` therefore keeps the two in
separate fields and never blends them into one number.
"""
