import os

from PIL import Image

file_list = []


# traverse root directory, and list directories as dirs and files as files
def absolute_file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            file_list.append( os.path.abspath(os.path.join(dirpath, f)) )


absolute_file_paths('data')
zero_sizes = []
for a_file in file_list:
    a_size = os.path.getsize(a_file)
    # if a_size != 50:
    if 'png' in a_file:
        try:
            img = Image.open(a_file).convert("LA")
        except:
            print('a file: ', a_file)

print('zero sizes: ', zero_sizes)
