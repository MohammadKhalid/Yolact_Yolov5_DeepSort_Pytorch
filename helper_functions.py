import os
import numpy as np
import json
import cv2
import shutil

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def create_groundtruth(path_to_images, path_to_json, path_to_destination):
    # make detination dir if does not exists
    directory = os.path.dirname(path_to_destination)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # open json file
    f = open(path_to_json)
    data = json.load(f)

    # traverse through images
    frame = 0
    directory = os.path.dirname(path_to_images)
    for file in os.listdir(directory):
        frame += 1
        # traverse through json
        for i in data['images']:
            # if filename matches with json
            if file == i['file_name']:
                # goto annotations in json and match ids
                for j in data['annotations']:
                    # if ids match get frame object_id and bbox
                    if j['image_id'] == i['id']:
                        line = flatten_list([[frame], j['object_id'], j['bbox'], [1, -1, -1, -1]])
                        # line = [int(float(x)) for x in line]
                        object_id = line[1]
                        # print(object_id)
                        # convert list to string
                        line = ','.join(str(x) for x in line)
                        # write to file gt.txt
                        if os.path.isfile(path_to_destination + 'gt.txt'):
                            a = open(path_to_destination + 'gt.txt','a')
                            a.write(line + '\n')
                            a.close()
                        else:
                            a = open(path_to_destination +'gt.txt','w+')
                            a.write(line + '\n')
                            a.close()
                        # print(file_name)
                        # print(flatten_list([[frame], j['object_id'], j['bbox'], [-1, -1, -1, -1]]))

# Remove commas to tracking results .txt file
def remove_comma(path_to_file, path_to_destination):
    f = open(path_to_file,'r')
    for line in f:
        line = line[:-2]
        if os.path.isfile(path_to_destination):
            a = open(path_to_destination,'a')
            a.write(line + '\n')
            a.close()
        else:
            a = open(path_to_destination,'w+')
            a.write(line + '\n')
            a.close()
    f.close()
    

if __name__ == "__main__":
    base = 'datasets/tracker_evaluation'
    path_images = base + '/ir/IR-04/'
    path_json = base + '/annotations/instances_trainval.json'
    path_destination = base + '/groundtruth/IR-04/gt/'
    # path_file = base + '/tracker0.2/IR-01.txt'
    # dest_file = base + '/tracker0.2/Copy-IR-01.txt'

    create_groundtruth(path_images, path_json, path_destination)
    # remove_comma(path_file, dest_file)

