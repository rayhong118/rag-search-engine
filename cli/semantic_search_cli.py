#!/usr/bin/env python3

import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    args = parser.parse_args()

    match args.command:
        case _:
            parser.print_help()
        case "verify":
            await verify_model()

if __name__ == "__main__":
    main()