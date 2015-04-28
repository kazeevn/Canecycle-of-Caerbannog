import os.path
from canecycle.cache import CacheReader
import sys 

def main():
#    test_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                             "train_except.txt")
#    test_file = "train_big_except.can"
    test_file = sys.argv[1]
    cache_reader = CacheReader(test_file)
    for item in cache_reader:
        pass
    cache_reader.close()

if __name__ == '__main__':
    main()
