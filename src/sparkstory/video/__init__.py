"""Everything that shells out to ffmpeg, and nothing that does not.

Quarantined into one package for the reason ``retrieval/web/`` is: it is the only
part of this feature that touches the outside world, so isolating it keeps every
other module testable with a fake. Argument construction, exit codes and stderr
parsing live here; page selection, ordering and the record live in
``workflows/animate.py`` and are exercised with no subprocess at all.
"""
