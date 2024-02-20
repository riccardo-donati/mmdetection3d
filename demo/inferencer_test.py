from mmdet3d.apis import LidarDet3DInferencer
import time

from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                LoadPointsFromFile)
from mmdet3d.models.data_preprocessors.data_preprocessor import \
    Det3DDataPreprocessor
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
import os 

def prepare_input(inputs):
    model_inputs = {}
    model_inputs["inputs"] = {}
    model_inputs["inputs"]["points"] = []
    max = 1

    files = []
    if not os.path.isdir(inputs["points"]):
        files.append(inputs["points"])
    else:
        files = [inputs["points"] + s for s in os.listdir(inputs["points"])]
    for i,sample_path in enumerate(files):
        inp = {}
        inp["lidar_points"] = {}
        inp["lidar_points"]["lidar_path"] = sample_path


        loader = LoadPointsFromFile(coord_type='LIDAR',load_dim=4,use_dim=4)
        packer = Pack3DDetInputs(keys=['points'])

        voxel_size = [0.16, 0.16, 4]
        preprocessor = Det3DDataPreprocessor(
            voxel=True,
            voxel_layer=dict(
                max_num_points=32,  # max_points_per_voxel
                point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                voxel_size=voxel_size,
                max_voxels=(16000, 40000)))

        input = loader(inp)
        input = packer(input)
        # input["inputs"]["points"] = input["inputs"]["points"].cuda()
        model_inputs["inputs"]["points"].append(input["inputs"]["points"])
        if i == max-1:
            print("Max Size!")
            break
    prep_input = preprocessor(model_inputs)
    prep_input["inputs"]["voxels"]["voxels"] = prep_input["inputs"]["voxels"]["voxels"].cuda()
    return prep_input
def main():
    # inference
    # pcl = './data/falcon/falcon1.bin'
    inferencer = LidarDet3DInferencer('pointpillars_donaset-car')
    model = inferencer.model

    pcl = 'data/DonaSet/training/velodyne/'
    inputs = dict(points=pcl)
    prep_input = prepare_input(inputs)

    model(prep_input["inputs"],prep_input["data_samples"], mode="predict")


    pcl = 'data/DonaSet/testing/velodyne/'
    inputs = dict(points=pcl)
    s = time.time()
    prep_input = prepare_input(inputs)
    res = model(prep_input["inputs"],prep_input["data_samples"], mode="predict")
    print("Time: {} ms".format((time.time()-s)*1000))
    # inferencer(inputs, show=True,pred_score_thr=0.3)

if __name__ == "__main__":
    main()