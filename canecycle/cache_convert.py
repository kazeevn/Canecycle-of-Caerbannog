#!/usr/bin/env python2
import argparse

from canecycle.reader import from_shad_lsml
from canecycle.cache import CacheWriter

def main():
    parser = argparse.ArgumentParser(description="Converter into the fast .can format")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input file in SHAD-LSML format")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output file to write in .can")
    parser.add_argument("-b", "--hash-size", type=int, required=True,
                        help="Hash table size in bits")
    args = parser.parse_args()
    reader = from_shad_lsml(args.input, args.hash_size)
    reader.restart(0)
    cache_writer = CacheWriter(reader.get_feature_columns_count(), args.hash_size)
    cache_writer.open(args.output)
    for item in reader:
        cache_writer.write_item(item)
    cache_writer.close()
    reader.close()

if __name__ == '__main__':
    main()
