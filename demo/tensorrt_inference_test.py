
import sys
sys.path.append("/home/riccardo/LidarObjDetection2/mmdetection3d")
sys.path.append("/home/riccardo/LidarObjDetection2/mmdeploy/")

from mmdeploy.apis.inference import inference_model
from mmdeploy.apis.inference import get_model
from mmdet3d.apis import LidarDet3DInferencer
import os
import torch
import os
import numpy as np
import time
from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                LoadPointsFromFile)
from mmdet3d.models.data_preprocessors.data_preprocessor import \
    Det3DDataPreprocessor
import onnxruntime as ort
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs



def main():
    inferencer = LidarDet3DInferencer('pointpillars_donaset_container-car')
    model = inferencer.model
    pcl_path = "../data/DonaSet/training/velodyne/"

    model_cfg = "../configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_donaset-3d-car.py"
    deploy_cfg = "/home/riccardo/LidarObjDetection2/mmdeploy/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-kitti-32x4_fp16.py"
    backend_files = "deployed_models/test2/end2end.engine"
    img = "../data/DonaSet/training/velodyne/000000.bin"
    device = "cuda"

    tensorrt_model = get_model(model_cfg,deploy_cfg,[backend_files],img,device)
    inputs = prepare_input({"points": pcl_path}, batch_size=1,data_samples=False)["inputs"]
    res = tensorrt_model(inputs, mode="predict")
    
    c = 0
    pcl_path = "../data/DonaSet/training/velodyne/00000"
    while True:
        input_utente = input("Premi Invio per continuare (oppure digita 'exit' per uscire): ")
        
        if input_utente.lower() == 'exit':
            break  # Esci dal ciclo se l'utente digita 'exit'

        pcl_path_curr = pcl_path + str(c) + ".bin"
        print("Testing -> "+pcl_path_curr)
        s0 = time.time()
        inputs = prepare_input({"points": pcl_path_curr}, batch_size=1,data_samples=False)["inputs"]
        s1=time.time()
        res = tensorrt_model(inputs, mode="tensor")
        s2 =time.time()
        preds  = model.bbox_head.predict_by_feat(res["cls_score"],res["bbox_pred"],res["dir_cls_pred"])
        print("Preprocess: {} ms".format(round((s1-s0)*1000,2)))
        print("Inf: {} ms".format(round((s2-s1)*1000,2)))
        print("Post: {} ms".format(round((time.time()-s2)*1000,2)))
        print("Total: {} ms".format(round((time.time()-s0)*1000,2)))
        print(preds)
        c+=1

def prepare_input(inputs,batch_size=1, device="cuda",data_samples=False):
    model_inputs = {}
    model_inputs["inputs"] = {}
    model_inputs["inputs"]["points"] = []
    if data_samples:
        model_inputs["data_samples"] = []
    max = batch_size

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
        if data_samples:
            model_inputs["data_samples"].append(input["data_samples"])
        if i == max-1:
            print("Batch Size!")
            break
    prep_input = preprocessor(model_inputs)
    if device == "cuda":
        prep_input["inputs"]["voxels"]["voxels"] = prep_input["inputs"]["voxels"]["voxels"].cuda()
    return prep_input


if __name__ == "__main__":
    main()
