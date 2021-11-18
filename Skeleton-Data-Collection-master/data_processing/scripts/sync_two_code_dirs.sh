#!/usr/bin/env bash

rsync -avu "/media/zhenyue-qin/Elements/Data-Collection/Skeleton-Data-Collection/" \
    "/media/zhenyue-qin/Seagate Expansion Drive/Collected-Skeleton-Data-Backup/Skeleton-Data-Collection/" \
    --info=progress2 \
    --exclude 'test_feeding_data'