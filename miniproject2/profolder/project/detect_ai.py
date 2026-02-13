import argparse
from yolo_gemini.cli import main as cli_main


def build_parser():
    return None  # Delegated to package CLI


def main():
    # Simply delegate to the library CLI so both entry points behave the same
    raise SystemExit(cli_main())


if __name__ == "__main__":
    main()
