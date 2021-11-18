#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 18:26:10 2021

@author: manci
"""

import os

allpath=[]


def getallfile(path):
    allfilelist=os.listdir(path)
    # 遍历该文件夹下的所有目录或者文件
    for file in allfilelist:
        filepath=os.path.join(path,file)
        # 如果是文件夹，递归调用函数
        if os.path.isdir(filepath):
            getallfile(filepath)
        # 如果不是文件夹，保存文件路径及文件名
        elif os.path.isfile(filepath):
            allpath.append(filepath)
    return allpath


if __name__ == "__main__":
    rootdir = "/media/manci/projectdata/skeleton/Collected-Skeleton-Data-Minimum/Skeleton"
    files = getallfile(rootdir)
    txt = open('/media/manci/projectdata/Skeletonpath.csv', 'w+')
    for file in files:
        txt.write(file +'\n')
    txt.close()
