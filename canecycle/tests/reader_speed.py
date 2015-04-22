import os.path
from canecycle.hash_function import HashFunction
from canecycle.reader import Reader, read_shad_lsml_header

def main():
    test_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "train_except.txt")
    format_ = read_shad_lsml_header(test_file)
    hash_function = HashFunction(20)
    reader = Reader(hash_function, test_file, format_, 1)
    for line in reader:
        pass

if __name__ == '__main__':
    main()
    
