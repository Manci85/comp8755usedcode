#!/usr/bin/env bash

rsync  -e 'ssh -p 22' \
    -av --update  /home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ \
    zhenyue@10.2.5.2:/home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ \
    --info=progress2 \
    --exclude-from='rsync_ignore_paths.txt'

rsync  -e 'ssh -p 22' \
    -av --update  /home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ \
    zhenyue@10.2.5.3:/home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ \
    --info=progress2 \
    --exclude-from='rsync_ignore_paths.txt'

