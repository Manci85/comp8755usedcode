import os
import subprocess

out = subprocess.Popen(['fuser', '-v', '/dev/nvidia0'],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
stdout, stderr = out.communicate()
print(stdout)
nums = [int(s) for s in stdout.split() if s.isdigit()]
for a_num in nums:
    # print('nums: ', a_num)
    subprocess.run(["kill", "-9", str(a_num)])
