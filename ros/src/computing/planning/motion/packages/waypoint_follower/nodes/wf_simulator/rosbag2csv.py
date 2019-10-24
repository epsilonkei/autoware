#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
from os import getcwd, makedirs
from os.path import dirname, basename, splitext, join, exists

def rel2abs(path):
    '''
    Return absolute path from relative path input
    '''
    return join(getcwd(), path)

def rosbag_to_csv(path, topic_name):
    name = splitext(basename(path))[0]
    folder = dirname(path) + '_csv'
    if not exists(folder):
        makedirs(folder)
    suffix = topic_name.replace('/', '-')
    output_path = join(folder, name + '_' + suffix + '.csv')
    if exists(output_path):
        return output_path
    else:
        command = "rostopic echo -b {0} -p /{1} | sed -e 's/,/ /g' > {2}".format(path, topic_name, output_path)
        print (command)
        subprocess.check_call(command, shell=True)
        return output_path

def basename_to_csv(path, topic_name):
    name = splitext(basename(path))[0]
    suffix = topic_name.replace('/', '-')
    output_path = join(dirname(path), name + '_' + suffix + '.csv')
    return output_path

if __name__ == '__main__':
    topics = [ 'vehicle_cmd/ctrl_cmd/steering_angle', 'vehicle_status/angle', \
               'vehicle_cmd/ctrl_cmd/linear_velocity', 'vehicle_status/speed', \
               'current_pose/pose']
    pd_data = [None] * len(topics)
    parser = argparse.ArgumentParser(description='Convert rosbag to csv file with select-input')
    parser.add_argument('--bag_file', '-b', required=True, type=str, help='rosbag file', metavar='file')
    args = parser.parse_args()

    for i, topic in enumerate(topics):
        csv_log = rosbag_to_csv(rel2abs(args.bag_file), topic)
