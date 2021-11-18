import subprocess
import os

#home_dictionary = "/media/manci/MyPassport"
#src = home_dictionary + "/skeleton/Collected-Skeleton-Data-Minimum/RGB/2021-05-17T14:00-16:00/Azure-1/2021-05-17T14h00-16h00-Azure-1-RGB-2JNW/"
#raw_directory = "/skeleton/Collected-Skeleton-Data-Minimum/RGB/2021-05-17T14:00-16:00/Azure-1/2021-05-17T14h00-16h00-Azure-1-RGB-2JNW/"
#dst = "/media/manci/MyPassport/output"+ raw_directory
#os.makedirs(dst)

import time

time_start=time.time()
time_end=time.time()


src = "/media/manci/MyPassport/skeleton/Collected-Skeleton-Data-Minimum/RGB/"
dst = "/media/manci/MyPassport/output/skeleton/Collected-Skeleton-Data-Minimum/RGB/"
raw = "/2021-07-10T14h30-16h30/Azure-5/2021-07-10T14h30-16h30-Azure-5-RGB_TOM2_videos"
os.makedirs(dst+raw)
for root, dirs, filenames in os.walk(src+raw, topdown=False):

    print(filenames)
    
    for filename in filenames:
        try:
            _format = ''
            if ".flv" in filename.lower():
                _format=".flv"
            if ".mp4" in filename.lower():
                _format=".mp4"
            if ".avi" in filename.lower():
                _format=".avi"
            if ".mov" in filename.lower():
                _format=".mov"

            inputfile = os.path.join(root, filename)
            

            outputfile = os.path.join(dst+raw, filename.replace(_format, ".mp4"))
            subprocess.call(['ffmpeg', '-i', inputfile, outputfile])  
        except:
            print("An exception occurred")
print('time cost',time_end-time_start,'s')