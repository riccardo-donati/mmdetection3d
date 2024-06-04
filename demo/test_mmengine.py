import time
from mmdet3d.apis import LidarDet3DInferencer

def main():
    inferencer = LidarDet3DInferencer('pointpillars_donaset_container-car')
    
    # Specify the path to your point cloud file (replace '../data/DonaSet/testing/velodyne/000001.bin' with your actual path)
    pcl_path = '../data/DonaSet/training/velodyne/000009.bin'
    
    inputs = dict(points=pcl_path)
    
    inferencer(inputs, show=False, pred_score_thr=0.3)
    # Perform inference
    start_time = time.time()
    inferencer(inputs, show=False, pred_score_thr=0.3)
    end_time = time.time()
    
    # Display the time taken for inference
    print("Time Taken: {:.2f} seconds".format(end_time - start_time))

if __name__ == "__main__":
    main()
