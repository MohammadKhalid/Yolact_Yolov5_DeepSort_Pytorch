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

def move_images(path_to_source, path_to_json, path_to_destination):
    
    directory = os.path.dirname(path_to_destination)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    f = open(path_to_json)

    data = json.load(f)

    for i in data['images']:
        image_file  = i['file_name']
        print(image_file)
        shutil.move(path_to_source + image_file, path_to_destination)



    f.close()

def create_labels(path_to_source, path_to_json, path_to_destination):
    directory = os.path.dirname(path_to_destination)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    f = open(path_to_json)

    data = json.load(f)

    for i in data['annotations']:
        line = flatten_list([i['category_id'], i['bbox']])
        line[1] = (line[1] + (line[3] / 2)) / 512
        line[2] = (line[2] + (line[4] / 2)) / 512
        line[3] = line[3]  / 512
        line[4] = line[4]  / 512
        line = ' '.join(str(x) for x in line)
        for j in data['images']:
            if j['id'] == i['image_id']:
                file_name  = j['file_name']
                file_name = file_name.split('/')
                file_name = file_name[2].split('.')
                file_name = file_name[0]
                if os.path.isfile(path_to_destination+file_name+'.txt'):
                    a = open(path_to_destination+file_name+'.txt','a')
                    a.write(line + '\n')
                    a.close
                else:
                    a = open(path_to_destination+file_name+'.txt','w+')
                    a.write(line + '\n')
                    a.close
                # print(line)

    
    f.close()


if __name__ == "__main__":
    base = '../datasets/ir_data_2021_11_04'
    train = base + '/train_data'
    test = base + '/test_data'
    train_anno = train + '/annotations/instances_train.json' 
    val_anno = train + '/annotations/instances_val.json'
    test_anno = test + '/annotations/instances_trainval.json'
    
    # move_images(test, test_anno, '../datasets/ir_data_2021_11_04/test/images/')
    # create_labels(test, test_anno, '../datasets/ir_data_2021_11_04/test/labels/')
