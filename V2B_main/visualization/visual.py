import os
import pickle
import numpy as np
import open3d as o3d
from LineMesh import LineMesh

def show_one_frame( pc, 
                    gt_bbox, 
                    pred_bbox,
                    capture_path=None,
                    window_pose=[960,540,100,100],
                    ):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_pose[0], height=window_pose[1], left=window_pose[2], top=window_pose[3])
    
    scene_color = np.ones((pc.shape[0], 3)) * 0.4
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.colors = o3d.utility.Vector3dVector(scene_color)

    # bbox
    lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], 
                        [4, 5], [5, 6], [6, 7], [7, 4], 
                        [0, 4], [1, 5], [2, 6], [3, 7]])

    gt_colors = np.array([[0., 1., 0.] for _ in range(len(lines_box))])         # green
    gt_line_mesh = LineMesh(gt_bbox, lines_box, gt_colors, radius=0.02)

    pred_colors = np.array([[1., 0., 0.] for _ in range(len(lines_box))])       # red
    pred_line_mesh = LineMesh(pred_bbox, lines_box, pred_colors, radius=0.02)

    vis.add_geometry(point_cloud)
    gt_line_mesh.add_line(vis)
    pred_line_mesh.add_line(vis)

    vis.run()

    # save picture
    if not capture_path is None:
        vis.capture_screen_image(capture_path)
        
    vis.destroy_window()

if __name__ == "__main__":
    data_path = 'visualization/data/kitti_car_2.pth'   
    which_dataset, category_name, tracklet_id = data_path.split('/')[-1].split('.')[0].split('_')
    
    file = open(data_path, "rb")
    data = pickle.load(file)
    file.close()
    
    pc = data['pointcloud']
    gt_bbox = data['gt_box']
    pred_bbox = data['pred_box']

    save_path = 'visualization/result/' + data_path.split('/')[-1].split('.')[0]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    frame_interval = 5
    window_pose = [800, 800, 100, 100]
    for i in range(0, len(pc), frame_interval): 
        show_one_frame( pc[i].T, 
                        gt_bbox[i].T, 
                        pred_bbox[i].T,
                        window_pose=window_pose,
                        capture_path=save_path+'/{:0>4d}.jpg'.format(i),
                        )