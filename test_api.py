from PythonAPI.bam import BAM
import numpy as np


# dataset_dir = '/home/beatriz/Documentos/Work/final_datasets'   # For Bea
dataset_dir = '/home/almartmen/Github/aikapi'   # For Alberto
dataset_name = '181129'

bam = BAM(dataset_dir, dataset_name, image_format='png')

# print(bam.get_persons_in_frame(800))
# print(bam.get_poses_in_frame(801))
# person3d = bam.get_person_in_frame(800, 1)
# print(person3d)
# pose3d = bam.get_pose_in_frame(799, 1)
# print(pose3d)

# print(bam.get_activities_for_person(2))

# print(bam.get_images_in_frame(1))
# bam.unroll_videos(force=True, video=1)
# camera = bam.get_camera(3,1)

# points3d = np.array([[0.498339264765202, 3.2171029078369897, 1.5828869056621102]])
# points2d = camera.project_points(points3d)
# print(points2d)

# points2d_pose = camera.project_points(pose3d)
# print(points2d_pose)

# bam.unroll_videos()
print(bam.get_total_cameras())
print(bam.get_total_frames())
print(bam.get_persons_in_frame(bam.get_total_frames()))