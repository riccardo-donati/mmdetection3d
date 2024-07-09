import numpy as np
import open3d as o3d
import os
import json
from sklearn.model_selection import train_test_split
import shutil

dataset_folder = "../data/Hydra"
label_folder = os.path.join(dataset_folder,"label")
point_folder = os.path.join(dataset_folder,"bin")

c = 0
for i,label_filename in enumerate(os.listdir(label_folder)):
    point_filename = label_filename[:-5] + '.bin'
    new_name = '{:06d}'.format(i)
    # Construct full paths for the old and new names
    old_point_path = os.path.join(point_folder, point_filename)
    new_point_path = os.path.join(point_folder, new_name + '.bin')
    
    old_label_path = os.path.join(label_folder, label_filename)
    new_label_path = os.path.join(label_folder, new_name + '.json')
    

    if not os.path.exists(new_point_path) and not os.path.exists(new_label_path):
        # Rename files
        os.rename(old_point_path, new_point_path)
        os.rename(old_label_path, new_label_path)
        c += 1
    else:
        print("Error with duplicate names!")
print("Renamed {} files".format(str(c)))

# Convert json files to txt KITTI format
def convert_json_to_txt(json_file, txt_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(txt_file, 'w') as f:
        for item in data:
            obj_id = item['obj_id']
            category_name = item['obj_type']
            position = item['psr']['position']
            rotation = item['psr']['rotation']
            scale = item['psr']['scale']

            x, y, z = position['x'], position['y'], position['z']
            dx, dy, dz = scale['x'], scale['y'], scale['z']
            yaw = rotation['z']

            line = f"{category_name} {x:.3f} {y:.3f} {z:.3f} {dx:.3f} {dy:.3f} {dz:.3f} {yaw:.3f}\n"
            f.write(line)

label_folder =  os.path.join(dataset_folder,"label")
label_dest_folder =  os.path.join(dataset_folder,"labels")
if not os.path.exists(label_dest_folder):
    os.makedirs(label_dest_folder)

for filename in os.listdir(label_folder):
    # Replace 'input.json' and 'output.txt' with your actual input and output file paths
    convert_json_to_txt(os.path.join(label_folder,filename), os.path.join(label_dest_folder,filename[:-5]+'.txt'))

# Create Training/Testing structure
training_folder =  os.path.join(dataset_folder,"training")
training_lidars_folder = os.path.join(training_folder,"velodyne")
training_labels_folder = os.path.join(training_folder,"label_2")

testing_folder =  os.path.join(dataset_folder,"testing")
testing_lidars_folder = os.path.join(testing_folder,"velodyne")

os.makedirs(training_lidars_folder)
os.makedirs(training_labels_folder)

os.makedirs(testing_lidars_folder)

# Check Datatypes
pcl = np.fromfile(os.path.join(dataset_folder,"bin","000003.bin"), dtype=np.float32).reshape(-1,4)
print(pcl[0])

# Split Dataset
train = 0.9
val  = 0.1
test = 0.0

label_folder =  os.path.join(dataset_folder,"labels")
train_folder =  os.path.join(dataset_folder,"bin")
train_samples_files = os.listdir(train_folder)

trainval_file = "trainval.txt"
train_file = "train.txt"
val_file = "val.txt"
test_file = "test.txt"
n_test = int(test * len(train_samples_files))
n_trainval = len(train_samples_files) - n_test
n_train =  int(train*n_trainval)
n_val  = n_trainval - n_train


# Specify the proportion for the validation set (e.g., 20% for validation)
validation_size = 0.1

# Split the list into training and validation sets
train_set, validation_set = train_test_split(train_samples_files, test_size=validation_size, random_state=42)

image_set_dir =  os.path.join(dataset_folder,"ImageSets")
if not os.path.exists(image_set_dir):
    os.makedirs(image_set_dir)
with open(os.path.join(image_set_dir,trainval_file), "w") as f:
    for filename in train_set + validation_set:
        f.write(filename[:-4]+"\n")
        shutil.copy(os.path.join(train_folder,filename), training_lidars_folder)
        shutil.copy(os.path.join(label_folder,filename[:-4]+".txt"), training_labels_folder)
with open(os.path.join(image_set_dir,train_file), "w") as f:
    for filename in train_set:
        f.write(filename[:-4]+"\n")
with open(os.path.join(image_set_dir,val_file), "w") as f:
    for filename in validation_set:
        f.write(filename[:-4]+"\n")

with open(os.path.join(image_set_dir,test_file), 'w') as file:
    pass
