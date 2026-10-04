"""`mk`: agent-friendly MMD rigging and set design for Blender. Run `mk <command> -h` for details and examples."""
import argparse
import importlib
import sys

from .. import __version__
from ..bridge import BlenderError
from .common import RUNTIME, USAGE, UsageError, emit

COMMANDS = ["assets", "build", "check", "doctor", "inspect", "look", "q", "serve"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Output is JSON on stdout. Exit codes: 0 ok, 1 a check failed, 2 usage error, 3 Blender/runtime error.")
    parser.add_argument("--version", action="version", version=f"mk-mmd {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for name in COMMANDS:
        importlib.import_module(f"mkmmd.cli.{name}").add(sub)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args) or 0
    except UsageError as e:
        emit({"error": str(e)})
        code = USAGE
    except BlenderError as e:
        emit({"error": str(e), "log": e.log, "trace": e.trace})
        code = RUNTIME
    except KeyboardInterrupt:
        code = RUNTIME
    sys.exit(code)


if __name__ == "__main__":
    main()
