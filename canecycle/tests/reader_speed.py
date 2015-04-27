import os.path
from canecycle.hash_function import HashFunction
from canecycle.reader import from_shad_lsml

def main():
#    test_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                             "train_except.txt")
    test_file = "train_big_except.txt"
    reader = from_shad_lsml(test_file, 20)
    reader.restart(0)
    for line in reader:
        pass

if __name__ == '__main__':
    main()
    
