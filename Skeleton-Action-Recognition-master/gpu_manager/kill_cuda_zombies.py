import os
import subprocess
import argparse


parser = argparse.ArgumentParser(description='CUDA Zombie Killer')
parser.add_argument(
    '--gpu-id',
    type=int,
    help='gpu_id'
)

args = parser.parse_args()

out = subprocess.Popen(['fuser', '-v', '/dev/nvidia'+str(args.gpu_id)],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
stdout, stderr = out.communicate()
print(stdout)
nums = [int(s) for s in stdout.split() if s.isdigit()]
for a_num in nums:
    subprocess.run(["kill", "-9", str(a_num)])