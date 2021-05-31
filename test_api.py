from PythonAPI.bam import BAM
import numpy as np


dataset_dir = '/home/beatriz/Documentos/Work/final_datasets'   # For Bea
# dataset_dir = '/home/almartmen/Github/aikapi'   # For Alberto
dataset_name = '181129'

bam = BAM(dataset_dir, dataset_name, image_format='png')

# bam.unroll_videos()

# print(bam.get_persons_in_frame(800))
# print(bam.get_poses_in_frame(801))
# person3d = bam.get_person_in_frame(800, 1)
# print(person3d)
# pose3d = bam.get_pose_in_frame(799, 1)
# print(pose3d)

# print(bam.get_activities_for_person(2))

# print(bam.get_images_in_frame(1141))
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
print(bam.get_person_ids())
print(bam.get_static_object_ids())
# print(bam.get_activity_names())

# print(bam.get_annotations_for_person(19))
# print(bam.get_persons_in_frame(1000))
# p = bam.get_annotations_for_person(1)
# print(p)

### OBJECTS
# print(bam.get_static_objects_in_frame(2))
# print(bam.get_static_object_in_frame(3, 21))
# print(bam.get_annotations_for_static_object(21))

bam.extract_2d_poses('181126_2d')
print(bam.get_2d_poses(2239, 0))


