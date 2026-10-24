"""Optional real-execution backends.

These require extra dependencies (see the ``real`` extra in
``pyproject.toml``); importing this package never pulls them in. The
heavy imports happen inside :class:`OpenAIBackend.__init__`, so the
zero-dependency core stays untouched.
"""
