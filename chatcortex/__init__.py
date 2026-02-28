from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("chatcortex")
except PackageNotFoundError:
    # Package not installed (e.g., running locally)
    __version__ = "0.0.0-dev"