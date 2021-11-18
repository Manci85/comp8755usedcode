import os
import subprocess

file1 = open('/media/zhenyue-qin/Backup Plus/Yang Liu/f16-video-set/download-all-urls', 'r')
lines = file1.readlines()

for a_line in lines:
    print('a line: ', a_line.split(' ')[-1])
    if os.path.exists(a_line):
        continue
    else:
        list_files = subprocess.run(a_line.split(' '))
    break
