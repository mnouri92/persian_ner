from os import listdir
from os.path import isfile, join
import sys

dir_path_in = sys.argv[1] #"/content/persian_ner/nsurl/tokenout/results"
dir_path_out = sys.argv[2] #"/content/persian_ner/nsurl/tokenout/final_results"

onlyfiles = [f for f in listdir(dir_path_in) if isfile(join(dir_path_in, f))]
for tmpfile in onlyfiles:
    file = join(dir_path_in, tmpfile)
    file_out = join(dir_path_out, tmpfile)
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = lines[:-1]
        with open(file_out, 'w') as fw:
            fw.writelines(lines)