#!/usr/bin/env python3

import sys
from src.solve.solve import solve


def main():
    path = sys.argv[1]
    with open(path) as file:
        print(solve(file))


if __name__ == '__main__':
    main()
