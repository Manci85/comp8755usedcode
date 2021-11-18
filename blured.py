import subprocess
import os
import time
import csv
from itertools import chain
#home_dictionary = "/media/manci/MyPassport"
#src = home_dictionary + "/skeleton/Collected-Skeleton-Data-Minimum/RGB/2021-05-17T14:00-16:00/Azure-1/2021-05-17T14h00-16h00-Azure-1-RGB-2JNW/"
#raw_directory = "/skeleton/Collected-Skeleton-Data-Minimum/RGB/2021-05-17T14:00-16:00/Azure-1/2021-05-17T14h00-16h00-Azure-1-RGB-2JNW/"
#dst = "/media/manci/MyPassport/output"+ raw_directory
#os.makedirs(dst)

start = time.time()
src = "/media/manci/projectdata/output/skeleton/Collected-Skeleton-Data-Minimum/RGB"
dst = "/media/manci/projectdata/output_blur"

with open('/media/manci/projectdata/raw_path_2.csv','r')as f:
    cs = list(csv.reader(f))
    path = list(chain.from_iterable(cs))
    
for raw in path:
    os.makedirs(dst+raw)
    for root, dirs, filenames in os.walk(src+raw, topdown=False):

        print(filenames)
    
        for filename in filenames:
            try:
                _format = ''
                if ".mp4" in filename.lower():
                    _format=".mp4"


                inputfile = os.path.join(root, filename)
            

                outputfile = os.path.join(dst+raw, filename.replace(_format, "_blured.mp4"))
                model_path = "/media/manci/projectdata/BlurryFaces-master/face_model/face.pb"
                #threshold = 0.4
                #print(inputfile,outputfile)
                #use os.system will break at ";",will lose data
                #os.system('python /media/manci/projectdata/BlurryFaces-master/src/auto_blur_video.py --input_video %s --output_video %s --model_path %s --threshold %f' % (inputfile,outputfile,model_path,threshold))
                
                command = ['python','/media/manci/projectdata/BlurryFaces-master/src/auto_blur_video.py','--input_video',inputfile,'--output_video',outputfile,'--model_path','/media/manci/projectdata/BlurryFaces-master/face_model/face.pb','--threshold','0.4']
                subprocess.call(command)
                #subprocess.run(['python','/media/manci/projectdata/BlurryFaces-master/src/auto_blur_video.py', '--input_video', inputfile,'--output_video', outputfile,'--model_path',model_path,'--threshold',threshold]) 
            except:
                print("An exception occurred")
end  = time.time()

print('time:',end-start)