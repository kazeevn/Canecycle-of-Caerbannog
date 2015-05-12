import os.path
import sys 

from canecycle.cache import CacheReader


def main():
    test_file = sys.argv[1]
    cache_reader = CacheReader(test_file)
    for item in cache_reader:
        pass
    cache_reader.close()


if __name__ == '__main__':
    main()
