#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 23:29:22 2021

@author: manci
"""

import os


def walk_files(path,endpoint=None):
    for root,dirs,filenames in os.walk(path,topdown=False):
        for dir in dirs:
            print(dir)
        '''for filename in filenames:
           inputfile = os.path.join(root, filename)
           folder_name,x = os.path.split(inputfile)
            
           #avi_name = os.path.basename(inputfile)
            #img = cv2.imread(inputfile)
            
           print(folder_name,x)
'''
''' try:
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
            print('[INFO] 1',inputfile)

            outputfile = os.path.join(dst+raw, filename.replace(_format, ".mp4"))
            subprocess.call(['ffmpeg', '-i', inputfile, outputfile])  
        except:
            print("An exception occurred")'''
        
path = r"/media/manci/MyPassport/skeleton/Collected-Skeleton-Data-Minimum/RGB/"
output_path = r"/media/manci/MyPassport/output/skeleton/Collected-Skeleton-Data-Minimum/RGB/"
walk_files(output_path)

