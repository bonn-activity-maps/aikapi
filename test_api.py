from PythonAPI.aik import AIK
import numpy as np


dataset_dir = '/home/beatriz/Documentos/Work'   # For Bea
# dataset_dir = '/home/almartmen/Github/aikapi'   # For Alberto
dataset_name = '181129'

aik = AIK(dataset_dir, dataset_name, image_format='png')

# print(aik.get_calibration_params(3, 1))
# print(aik.get_persons_in_frame(801))
print(aik.get_poses_in_frame(801))
# print(aik.get_person_in_frame(800, 1))
pose3d = aik.get_pose_in_frame(799, 1)
# print(pose3d)

# print(aik.get_activities_for_person(2))

# print(aik.get_images_in_frame(1))
# aik.unroll_videos(force=True, video=1)
camera = aik.get_camera(3,1)

# points3d = np.array([[0.498339264765202, 3.2171029078369897, 1.5828869056621102]])
# points2d = camera.project_points(points3d)
# print(points2d)

# mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False]
# points2d_pose = camera.project_points(pose3d, True, mask)

points2d_pose = camera.project_points(pose3d)

print(points2d_pose)
