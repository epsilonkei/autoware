#!/bin/bash
# -*- mode: sh; coding: utf-8-unix; -*-

BAGS_LIST=`ls /media/narwhal/horibe500GB/lexus_data/0405/*.bag`
for bag in $BAGS_LIST
do
    python rosbag2csv.py --bag_file ${bag}
done
