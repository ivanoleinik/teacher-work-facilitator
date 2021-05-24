#!/usr/bin/env python3

import sys
from src.solve.solve import do_check

if __name__ == '__main__':
    print(*do_check(sys.argv[1]), sep='\n')
