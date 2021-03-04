# Interface for accessing the AIK dataset.

import os
import subprocess
import json
import numpy as np
import cv2
import shutil
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple, Union, List, TypeVar
from PythonAPI.utils.camera import Camera

kp = TypeVar('kp', float, float, float)


class BAM:

    def __init__(self, dataset_dir: str, dataset_name: str, image_format: str = 'png') -> None:
        """
            Constructor for BAM class for reading and visualizing annotations.

        :param dataset_dir: absolute path to the folder containing the datasets
        :param dataset_name: name of the dataset folder
        :param image_format: format of images to extract video frames, png or jpeg (png by default)
        """
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(dataset_dir, self.dataset_name)
        self.cameras_dir = os.path.join(self.dataset_dir, 'cameras')
        self.videos_dir = os.path.join(self.dataset_dir, 'videos')

        # Check if dataset exists in the directory
        assert os.path.isdir(self.dataset_dir), "The dataset directory %r does not exist" % dataset_dir
        assert os.path.isdir(self.cameras_dir), "The dataset is not complete, cameras information is missing"

        assert image_format in ['png', 'jpeg'], "The image format should be png or jpeg"
        self.image_format = image_format

        self.num_cameras, self.num_frames = self._read_dataset_info()
        self.frame_format = "frame%09d"
        self.max_persons = 50

        # Load info to memory
        self.calibration_params = self._read_calibration_params()
        self.persons, self.objects, self.activities, self.person_ids, self.object_ids, self.activity_names = \
            self._read_annotations()

        print('Finish loading dataset')

    def _unroll_video(self, video: int) -> None:
        """
            Unrolls video in its corresponding folder. Creates cameraXX folder

        :param video: number of video
        """
        video_file = self.dataset_name + '_' + str(video).zfill(2) + '.mp4'

        # Create camera directory to store all frames
        camera = 'camera' + str(video).zfill(2)
        camera_dir = os.path.join(self.videos_dir, camera)
        os.mkdir(camera_dir)

        if self.image_format == 'jpeg':
            unroll = subprocess.run(["ffmpeg", "-i", os.path.join(self.dataset_dir, video_file), "-qscale:v", "2", "-vf", "scale=1280:720",
                                     os.path.join(camera_dir, self.frame_format + "." + self.image_format)])
        else:
            unroll = subprocess.run(["ffmpeg", "-i", os.path.join(self.dataset_dir, video_file), "-vf", "scale=1280:720",
                                     os.path.join(camera_dir, self.frame_format + "." + self.image_format)])
        # print("The exit code was: %d" % unroll.returncode)

    def unroll_videos(self, force: bool = False, video: Optional[int] = None) -> None:
        """
            Unrolls all the videos and stores them in the videos folder. This folder contains 12 folders (cameraXX) with
        the unrolled frames from each camera.
        Create videos folder if it doesn't exist.

        :param force: if True, the unrolled frames are deleted and the videos are unrolled again
        :param video: if None all videos are unrolled.
                      If it has a concrete value, only that video is unrolled
        """
        # Remove folders
        if force:
            if video is None:   # remove videos folder
                try:
                    shutil.rmtree(self.videos_dir)
                except OSError as e:
                    print("Error: %s : %s" % (self.videos_dir, e.strerror))
                os.mkdir(self.videos_dir)
            else:               # Only remove corresponding camera folder
                camera = 'camera' + str(video).zfill(2)
                camera_dir = os.path.join(self.videos_dir, camera)
                try:
                    shutil.rmtree(camera_dir)
                except OSError as e:
                    print("Error: %s : %s" % (camera_dir, e.strerror))
        else:
            # Create videos directory and unroll videos if the directory videos does not exist
            if not os.path.exists(self.videos_dir):
                os.mkdir(self.videos_dir)
            # Inform the user if the image format is different
            elif os.path.splitext(os.listdir(os.path.join(self.videos_dir, os.listdir(self.videos_dir)[0]))[0])[1].split('.')[1] != self.image_format:
                print("The image format is different from the unrolled frames. "
                      "If you want to unroll them again and overwrite the current frames, "
                      "please use the option force=True")
                return
            elif video is None:     # If videos are already unrolled and not force
                print("The videos are already unrolled. If you want to unroll them again and overwrite the current frames, "
                      "please use the option force=True")
                return
            else:                   # If we want to unroll 1 video
                camera = 'camera' + str(video).zfill(2)
                if camera in os.listdir(self.videos_dir):
                    print("Video", video, "is already unrolled. If you want to unroll it again and overwrite"
                                          " the current frames, please use the option force=True")
                    return

        # Unroll videos
        if video is None:   # Unroll all videos if none
            print('Unrolling videos. This may take a while...')
            for c in range(self.num_cameras):
                self._unroll_video(c)
        else:               # Unroll concrete video
            assert 0 <= int(video) <= self.num_cameras, "The should be between 0 and %r" % self.num_cameras
            print('Unrolling video', video, '. This may take a while...')
            self._unroll_video(int(video))

    def _read_dataset_info(self) -> Tuple[int, int]:
        """
            Reads general dataset information.
        
        :returns: Number of cameras and number of frames
        """
        print('Reading dataset information...')
        dataset_file = os.path.join(self.dataset_dir, 'dataset.json')
        assert os.path.isfile(dataset_file), "The dataset is not complete, the dataset.json file is missing"

        with open(dataset_file) as f:
            data = json.load(f)

        # Get total number of frames and upsample
        # num_frames = data['valid_frames'][-1] * 2

        # Total frames, for now.. --> Horrible hardcoding but necessary, please change it in the future!
        total_frames = {
            '181129': 64862 * 2 - 1,
            '190502': 89585 * 2,
            '190719': 87778 * 2,
            '190726': 88819 * 2,
        }
        return data['n_cameras'], total_frames[self.dataset_name]

    def _read_calibration_params(self) -> np.ndarray:
        """
            Reads camera calibration parameters for each camera.

        :returns: Numpy array with calibration parameters
        """
        print('Loading calibration parameters...')
        cameras_data = []

        for c in range(self.num_cameras):
            camera = 'camera' + str(c).zfill(2) + '.json'
            print('    ', camera+'...')
            with open(os.path.join(self.cameras_dir, camera)) as f:
                data = json.load(f)

            # # Store data for each frame in numpy array
            # camera_params = np.empty(0)
            # for d in data:
            #     frames = d['end_frame'] - d['start_frame']
            #     del d['start_frame']
            #     del d['end_frame']
            #     cam = np.full(frames, d)
            #     camera_params = np.append(camera_params, cam, axis=0)
            #
            cameras_data.append(data)
        return np.array(cameras_data, dtype=object)

    def _read_annotations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            Reads persons, objects, actions and ids information from file.

        :returns: Persons, objects, activities and ids numpy arrays with information in json format
        """
        print('Loading annotations...')
        with open(os.path.join(self.dataset_dir, self.dataset_name + '_unroll.json')) as f:
            # Read file by line: persons, objects, actions
            for i, json_obj in enumerate(f):
                json_data = json.loads(json_obj)

                # Process and separate data into persons, objects and activities
                if i == 0:
                    persons, person_ids = self._process_persons(json_data['persons'])
                elif i == 1:
                    objects, object_ids = self._process_objects(json_data['objects'])
                elif i == 2:
                    activities, activity_names = self._process_activities(json_data['actions'])
                else:
                    print('Incorrect format in annotation file')
                    exit()
        return persons, objects, activities, person_ids, object_ids, activity_names

    def _process_persons(self, persons_json: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Process persons and ids jsons ordered by frame.

        :param persons_json: numpy array with jsons that contain persons information
        :returns: tuple with numpy arrays with person info and ids in json format for each frame
        """
        print('Processing persons...')
        persons = []
        ids = []
        last_frame = -1
        for d in persons_json:
            frame = d['frame']

            # Complete if there are empty frames in between
            if frame > last_frame + 1:
                for i in range(last_frame+1, frame):
                    persons.append([])
            #
            # Add persons in current frame if there are persons in it
            if 'persons' in d:
                persons_in_frame = d['persons']

                # Get the ids of the annotated persons
                for person in persons_in_frame:
                    ids.append(person['pid'])

                persons.append(persons_in_frame)

            last_frame = frame

        # Add empty frames at the end if the dataset has unnanotated frames to avoid errors
        if len(persons) <= self.num_frames//2:
            for i in range(len(persons), self.num_frames//2+1):
                persons.append([])

        # Make sure that there are no repeated IDs
        ids = np.unique(np.array(ids, dtype=np.int))

        return np.array(persons, dtype=object), ids

    def _process_objects(self, objects_json: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Process objects ordered by frame.

        :param objects_json: numpy array with jsons that contain objects information
        :returns: tuple with numpy arrays with object info and ids in json format for each frame
        """
        print('Processing objects...')
        objects = []
        ids = []
        last_frame = -1
        for d in objects_json:
            frame = d['frame']

            # Complete if there are empty frames in between
            if frame > last_frame + 1:
                for i in range(last_frame + 1, frame):
                    objects.append([])

            # Add objects in current frame
            objects_in_frame = d['objects']

            # Get the ids of the annotated objects
            for obj in objects_in_frame:
                ids.append(obj['oid'])

            objects.append(objects_in_frame)
            last_frame = frame

        # Add empty frames at the end if the dataset has unnanotated frames to avoid errors
        if len(objects) <= self.num_frames // 2:
            for i in range(len(objects), self.num_frames // 2 + 1):
                objects.append([])

        # Make sure that there are no repeated IDs
        ids = np.unique(np.array(ids, dtype=np.int))

        return np.array(objects, dtype=object), ids

    def _process_activities(self, activities_json: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Process activities json and order by person id.

        :param activities_json: numpy array with jsons that contain activities information
        :returns: numpy array with activities info ordered by person id and all activities available in dataset
        """
        print('Processing actions...')
        activities = [[] for i in range(self.max_persons)]  # Initialize with empty activities
        activity_names = []

        for d in activities_json:
            pid = d['pid']
            del d['pid']
            activities[pid].append(d)
            activity_names.append(d['label'])

        # Make sure that there are no repeated activities
        activity_names = np.unique(np.array(activity_names, dtype=np.str))
        return np.array(activities, dtype=object), activity_names

    def _get_real_frame(self, frame: int) -> Tuple[bool, int, int]:
        """
            Gets the real frames or frames that need to be interpolated

        :param frame: number of frame
        :returns: True if the dataset contains the info about the frame, False if we have to interpolate.
                The real frame if it is stored in the dataset
                The previous and posterior frames if the information about that frame is not stored
        """
        # Even -> the frame info is not contained in the dataset
        if (frame % 2) == 0:
            real_frame = frame//2
            return False, real_frame, real_frame+1
        else:     # Odd -> return real frame
            return True, frame//2 + 1, -1

    def _get_calibration_params(self, video: int, frame: int) -> Tuple[bool, Union[dict, str]]:
        """
            Gets calibration parameters that satisfy given filter conditions. Multiply x2-1 in order to get upsampled frames in range

        :param video: video number
        :param frame: frame number
        :returns: calibration parameters in json format with K, rvec, tvec, distCoef, w and h
        """
        for p in self.calibration_params[video]:
            start_frame = p['start_frame'] * 2 - 1
            end_frame = p['end_frame'] * 2 - 1
            if start_frame <= frame <= end_frame:
                new_params = p.copy()
                new_params['start_frame'] = start_frame
                new_params['end_frame'] = end_frame
                return True, new_params
        return False, 'Frame ' + str(frame) + ' does not exist in dataset ' + self.dataset_name
    
    def get_camera(self, video: int, frame: int) -> Union[Camera, None]:
        """
            Gets Camera for specified video and frame. Multiply x2-1 in order to get upsampled frames in range

        :param video: video number
        :param frame: frame number
        :returns: Camera object
        """
        params = self._get_calibration_params(video, frame)
        # Check if we found the camera parameters
        if not (params[0]):
            print(params[1])
            return
        else:
            params = params[1]
            
        # Construct the Camera object
        camera = Camera(params['K'], params['rvec'], params['tvec'], params['distCoef'], params['w'], params['h'])

        # Return the camera
        return camera

    def _interpolate(self, kps1: List[List[kp]], kps2: List[List[kp]]) -> np.ndarray:
        """
            Interpolates all keypoints the frame in between kps1 and kps2

        :param kps1: keypoints from first frame
        :param kps2: keypoints from second frame
        :returns: numpy array with interpolated keypoints
        """
        interpolated_kps = []
        for i in range(len(kps1)):
            # If one of the two points is empty -> Not interpolate
            if len(kps1[i]) != 0 and len(kps2[i]) != 0:
                interpolated_coords = np.linspace(np.array(kps1[i]), np.array(kps2[i]), num=3).tolist()
                interpolated_kps.append(interpolated_coords[1])
            else:
                interpolated_kps.append([None, None, None])
        return np.array(interpolated_kps)

    def get_poses_in_frame(self, frame: int) -> np.ndarray:
        """
            Gets all poses annotated in the given frame. Interpolates keypoints if the frame is not annotated

        :param frame: frame number
        :returns: Poses annotated in the given frame in json format
        """
        return self._get_objects_in_frame(frame, np.array(['poseAIK']), 'pid', self.persons)

    def get_persons_in_frame(self, frame: int) -> np.ndarray:
        """
            Gets all persons annotated in the given frame. Interpolates keypoints if the frame is not annotated

        :param frame: frame number
        :returns: Persons annotated in the given frame in json format
        """
        return self._get_objects_in_frame(frame, np.array(['personAIK']), 'pid', self.persons)

    def get_static_objects_in_frame(self, frame: int) -> np.ndarray:
        """
            Gets all objects annotated in the given frame. Interpolates keypoints if the frame is not annotated

        :param frame: frame number
        :returns: Objects annotated in the given frame in json format
        """
        obj_types = np.array(['boxAIK', 'cylinderAIK'])
        return self._get_objects_in_frame(frame, obj_types, 'oid', self.objects)

    def _get_objects_in_frame(self, frame: int, obj_types: np.ndarray, id_type: str, original_objects: np.ndarray) -> np.ndarray:
        """
            Gets all objects with types in obj_types annotated in the given frame. Interpolates keypoints if the frame is not annotated

        :param frame: frame number
        :param obj_types: Array with poseAIK, personAIK, boxAIK or cylinderAIK
        :param id_type: key for id in json. pid for persons and poses, oid for objects
        :param original_objects: array where objects are searched. persons array for persons or poses, objects array for objects
        :returns: Objects of obj_types annotated in the given frame in json format
        """
        # If we need to look for personAIK or poseAIK, only 1 element is allowed in the obj_type array
        # to avoid error because persons and poses has the same pid
        if 'personAIK' in obj_types or 'poseAIK' in obj_types:
            assert len(obj_types) == 1, "Only 1 object type admitted when getting objects in frame"

        annotated, real_frame, next_frame = self._get_real_frame(frame)

        if annotated:
            objects = []
            for a in original_objects[real_frame]:
                if a['type'] in obj_types:
                    objects.append(a)
            return np.array(objects)
        else:
            objects = []
            # Interpolate all objects
            for p1 in original_objects[real_frame]:
                for p2 in original_objects[next_frame]:
                    if p1[id_type] == p2[id_type] and len(p1['location']) == len(p2['location']) \
                            and p1['type'] in obj_types and p2['type'] in obj_types:
                        interpolated_person = self._interpolate(p1['location'], p2['location'])
                        object_json = {
                            id_type: p1[id_type],
                            'location': interpolated_person.tolist(),
                            'type': p1['type']
                        }
                        # Add labels to interpolated object if it exists in other frames
                        if 'labels' in p1:
                            object_json['labels'] = p1['labels']
                        objects.append(object_json)
                        break
            return np.array(objects)

    def get_person_in_frame(self, frame: int, person_id: int) -> np.ndarray:
        """
            Gets person annotation for person_id in the given frame.

        :param frame: frame number
        :param person_id: person identifier
        :returns: Person annotated in the given frame in json format if exists, empty array otherwise
        """
        return self._get_object_in_frame(frame, person_id, np.array(['personAIK']), 'pid', self.persons)

    def get_pose_in_frame(self, frame: int, person_id: int) -> np.ndarray:
        """
            Gets pose annotation for person_id in the given frame.

        :param frame: frame number
        :param person_id: person identifier
        :returns: Pose in json format if exists, empty array otherwise
        """
        return self._get_object_in_frame(frame, person_id, np.array(['poseAIK']), 'pid', self.persons)

    def get_static_object_in_frame(self, frame: int, object_id: int) -> np.ndarray:
        """
            Gets static object annotation for object_id in the given frame.

        :param frame: frame number
        :param object_id: object identifier
        :returns: Object in json format if exists, empty array otherwise
        """
        obj_types = np.array(['boxAIK', 'cylinderAIK'])
        return self._get_object_in_frame(frame, object_id, obj_types, 'oid', self.objects)

    def _get_object_in_frame(self, frame: int, object_id: int, obj_types: np.array, id_type: str,
                             original_objects: np.ndarray) -> np.ndarray:
        """
            Gets annotation for person_id in the given frame.

        :param frame: frame number
        :param object_id: person identifier
        :param obj_types: Array with poseAIK, personAIK, boxAIK or cylinderAIK
        :param id_type: key for id in json. pid for persons and poses, oid for objects
        :param original_objects: array where objects are searched. persons array for persons or poses, objects array for objects
        :returns: Objects annotated in the given frame in json format if exists, empty array otherwise
        """
        # If we need to look for personAIK or poseAIK, only 1 element is allowed in the obj_type array
        # to avoid error because persons and poses has the same pid
        if 'personAIK' in obj_types or 'poseAIK' in obj_types:
            assert len(obj_types) == 1, "Only 1 object type admitted when getting objects in frame"

        annotated, real_frame, next_frame = self._get_real_frame(frame)
        if annotated:
            frame_annotation = original_objects[real_frame]
            for a in frame_annotation:
                if a[id_type] == object_id and a['type'] in obj_types:
                    # Substitute empty lists with None
                    pts3d = [x if x else [None, None, None] for x in a['location']]
                    return np.array(pts3d)
        else:
            p1, p2 = [], []
            # Search keypoints in previous and next frame to interpolate
            frame_annotation = original_objects[real_frame]
            for a in frame_annotation:
                if a[id_type] == object_id and a['type'] in obj_types:
                    p1 = a['location']
                    break
            frame_annotation = original_objects[next_frame]
            for a in frame_annotation:
                if a[id_type] == object_id and a['type'] in obj_types:
                    p2 = a['location']
                    break
            # Interpolate only if the object exists in both frames
            if p1 and p2:
                return self._interpolate(p1, p2)

        return np.array([])

    def get_activities_for_person(self, pid: int) -> np.ndarray:
        """
            Gets all activities annotated for the person with pid. Multiply ranges x2-1 in order to get correct ranges

        :param pid: person identifier
        :returns: Numpy array with activities for the specified person in json format
        """
        activities = []
        for a in self.activities[pid]:
            new_activity = a.copy()
            new_activity['start_frame'] = new_activity['start_frame'] * 2 - 1
            new_activity['end_frame'] = new_activity['end_frame'] * 2 - 1
            activities.append(new_activity)
        return np.array(activities)

    def get_images_in_frame(self, frame: int) -> np.ndarray:
        """
            Gets camera images for the specified frame in all the cameras

        :param frame: frame number
        :returns: Numpy array with images in numpy array format if the frame exists, empty array otherwise
        """

        # Check if it's the first time the frames are requested and the videos are not unrolled
        if not os.path.exists(self.videos_dir):
            self.unroll_videos()

        print("Searching for images of frame ", frame, "...")
        # Create the string of the name of the frame that we are going to search for in all camera folders
        frame_name = "frame" + ''.zfill(9)
        frame_string = str(frame)
        number_of_chars = len(frame_string)
        frame_name = frame_name[:-number_of_chars] + frame_string + "." + self.image_format
        
        print("Frame name: " + frame_name)

        # Get the paths to all cameras inside the videos folder sorted by name
        cameras_paths = [os.path.join(self.videos_dir, name) for name in os.listdir(self.videos_dir) if os.path.isdir(os.path.join(self.videos_dir,name))]
        cameras_paths.sort()

        # Get the frame_name image from those paths
        images = []
        print(cameras_paths)

        for path in cameras_paths:
            image = cv2.imread(os.path.join(path, frame_name), cv2.IMREAD_COLOR)
            print(os.path.join(path, frame_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        print("Images of frame ", frame, " retrieved.")
        return np.array(images)
    
    def get_person_ids(self) -> np.ndarray:
        """
            Returns the existing person ids in the dataset

        :returns: Numpy array with the existing IDs of the persons in the dataset
        """
        return self.person_ids

    def get_static_object_ids(self) -> np.ndarray:
        """
            Returns the existing object ids in the dataset

        :returns: Numpy array with the existing IDs of the objects in the dataset
        """
        return self.object_ids

    def get_activity_names(self) -> np.ndarray:
        """
            Returns the existing activities in the dataset

        :returns: Numpy array with the existing activity names in the dataset
        """
        return self.activity_names

    def get_total_frames(self) -> int:
        """
            Returns the total number of frames in the dataset

        :returns: Total number of frames in the dataset
        """
        return self.num_frames

    def get_total_cameras(self) -> int:
        """
            Returns the total number of cameras in the dataset

        :returns: Total number of cameras in the dataset
        """
        return self.num_cameras

    def get_annotations_for_person(self, person_id: int) -> np.ndarray:
        """
            Gets all annotations in the dataset for person_id.

        :param person_id: person identifier
        :returns: Numpy array with all annotations for person in json format if exists, empty array otherwise
        """
        return self._get_annotations_for_person_or_pose(person_id, np.array(['personAIK']))

    def get_annotations_for_pose(self, person_id: int) -> np.ndarray:
        """
            Gets all pose annotations in the dataset for person_id.

        :param person_id: person identifier
        :returns: Numpy array with all pose annotations for person in json format if exists, empty array otherwise
        """
        return self._get_annotations_for_person_or_pose(person_id, np.array(['poseAIK']))

    def get_annotations_for_static_object(self, object_id: int) -> np.ndarray:
        """
            Gets all annotations in the dataset for object_id.

        :param object_id: object identifier
        :returns: Numpy array with all annotations for object in json format if exists, empty array otherwise
        """
        obj_types = np.array(['boxAIK', 'cylinderAIK'])
        return self._get_annotations_for_object(object_id, obj_types, 'oid', self.objects)


    def distance_to_static_object(self, object_type: str, object_points: np.ndarray, point: np.ndarray) -> np.ndarray:
        """
            Calculate the distance vector (direction is the vector and distance is the module) between a point defined by 'point'
            and the object defined by 'object_points'


        :param object_type: type of the static object 'boxAIK' or 'cylinderAIK'
        :param object_points: array with the annotated 3D points of the static object
        :param point: 3D point for which we want to obtain the distance to the object
        :returns: Numpy array with the distance vector between the points and the object. If the point is inside of the object, the vector
                  returned will be [0,0,0]
        """
        
        if object_type == 'boxAIK':
            # ["tfl", "tfr", "tbl", "tbr", "bfl", "bfr", "bbl", "bbr"]
            box3D = self.create_box(object_points[0], object_points[1], object_points[2])

            # Focusing on the botom front left corner of the cube we will obtain the local coordinate system
            x_vector = (box3D[5] - box3D[4]) # bfr - bfl
            y_vector = (box3D[6] - box3D[4]) # tfl - bfl
            z_vector = (box3D[0] - box3D[4]) # bbl - bfl

            x_local = x_vector / np.linalg.norm(x_vector)
            y_local = y_vector / np.linalg.norm(y_vector)
            z_local = z_vector / np.linalg.norm(z_vector)

            # Now we have to find the rotation to align our local coordinate system with the world coordinate system
            rotation, _ = R.align_vectors([[1,0,0],[0,1,0],[0,0,1]], [x_local, y_local, z_local])

            # Now we can apply the rotation to the box coordinates and to the point
            box3D_a = rotation.apply(box3D)
            point_a = rotation.apply(point)

            # Find the limits of the rotated box
            x_array = box3D_a[:,0]
            y_array = box3D_a[:,1]
            z_array = box3D_a[:,2]

            min_x = np.min(x_array)
            max_x = np.max(x_array)
            min_y = np.min(y_array)
            max_y = np.max(y_array)
            min_z = np.min(z_array)
            max_z = np.max(z_array)
            
            # First check if the point is inside, to directly return [0,0,0]
            if (point_a[0] > min_x and point_a[0] < max_x) and (point_a[1] > min_y and point_a[1] < max_y) and (point_a[2] > min_z and point_a[2] < max_z):
                return [0,0,0]

            # If its not inside, we calculate the closest point within the cube
            closest_point = [0,0,0]

            # X coordinate
            if point_a[0] < min_x:
                closest_point[0] = min_x
            elif point_a[0] > max_x:
                closest_point[0] = max_x
            else:
                closest_point[0] = point_a[0]

            # Y coordinate
            if point_a[1] < min_y:
                closest_point[1] = min_y
            elif point_a[1] > max_y:
                closest_point[1] = max_y
            else:
                closest_point[1] = point_a[1]
            
            # Z coordinate
            if point_a[2] < min_z:
                closest_point[2] = min_z
            elif point_a[2] > max_z:
                closest_point[2] = max_z
            else:
                closest_point[2] = point_a[2]
            
            # Then return the distance
            distance = (closest_point - point_a)
            return distance
            
        elif object_type == 'cylinderAIK':
            # For the cylinderAIK we have 2 points, top face center and top face radius point
            center_top = object_points[0]
            radius_top = object_points[1]
            
            # Radius of the top face, will be used later
            radius_distance = np.linalg.norm(center_top - radius_top)

            # Check if the point is above the cylinder
            if point[2] >= center_top[2]:
                # Check if the point is also inside of the silhouette of the top circle
                center_top_2D = np.asarray([center_top[0], center_top[1]])
                radius_top_2D = np.asarray([radius_top[0], radius_top[1]])
                point_2D = np.asarray([point[0], point[1]])

                radius_distance_2D = np.linalg.norm(center_top_2D - radius_top_2D)
                distance_2D = np.linalg.norm(center_top_2D - point_2D)

                if distance_2D <= radius_distance_2D:
                    # Inside the silhouette. We just need to check the distance to the top face surface
                    # Obtain the projection of the point into the surface plane by changing the Z value of the point
                    projected_point = np.asarray([point[0], point[1], center_top[2]])
                    # Then calculate the distance between the original point and the projected one
                    distance = (projected_point - point)
                    return distance
                else: 
                    # Outside the silhouette. We need to find the point in the top surface radius closest to the point
                    # Obtain the projection of the point into the surface plane by changing the Y value of the point
                    projected_point = np.asarray([point[0], point[1], center_top[2]])
                    # Obtain the directional normalized vector between the center of the surface and the projected point
                    direction_vector = (projected_point - center_top)
                    direction = direction_vector / np.linalg.norm(direction_vector)
                    # Multiply the direction by the radius of the surface to obtain the closest point on the edge
                    closest_point = center_top + (direction * radius_distance)
                    # Now we can just check the distance between the points
                    distance = (closest_point - point)
                    return distance
            else:
                # Find the cylinder center point at the same height as the outside point
                center_point = np.asarray([center_top[0], center_top[1], point[2]])
                # Obtain the directional normalized vector between the new center of the object and the point
                direction_vector = (point - center_point)
                direction = direction_vector / np.linalg.norm(direction_vector)
                # Multiply the direction by the radius to obtain the edge point of the object closest to the outside point 
                closest_point = center_point + (direction * radius_distance)
                # Now we can check the distance between the points
                distance = (closest_point - point)
                return distance

    def create_box(self, a, b, c):
        """
        Auxiliar function that given the 3 points stored for a boxAIK object will generate the 8 coordinates of the corners of the box

        :param a: (x, y, z) top-left point
        :param b: (x, y, z) top-right point
        :param c: (x, y, z) bottom-left or bottom-right point
        """
        proj_to_xy = lambda x: x[:2]
        get_angle = lambda x,y: (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

        ab = proj_to_xy(b) - proj_to_xy(a)
        ac = proj_to_xy(c) - proj_to_xy(a)
        bc = proj_to_xy(c) - proj_to_xy(b)

        ab_ac = np.abs(get_angle(ab, ac))
        ab_bc = np.abs(get_angle(ab, bc))

        x1, y1, z1 = a
        x2, y2, z2 = b
        x3, y3, z3 = c

        z = (z1 + z2)/2

        down = np.array([0., 0., z - z3])

        if ab_ac < ab_bc:  # 3. point is bottom-left
            back = np.array([ac[0], ac[1], 0])
        else:  # 3. point is bottom-right
            back = np.array([bc[0], bc[1], 0])

        tfl = np.array([x1, y1, z])
        tfr = np.array([x2, y2, z])

        tbl = tfl + back
        tbr = tfr + back

        bfl = tfl - down
        bfr = tfr - down

        bbl = bfl + back
        bbr = bfr + back

        return np.array([
            tfl, tfr,
            tbl, tbr,
            bfl, bfr,
            bbl, bbr
        ])

    def _get_annotations_for_object(self, object_id: int, obj_types: np.ndarray, id_type: str,
                                    original_objects: np.ndarray) -> np.ndarray:
        """
            Gets all annotations for object_id and obj_type in the whole dataset.

        :param object_id: object identifier
        :param obj_types: Array with poseAIK, personAIK, boxAIK or cylinderAIK
        :param id_type: key for id in json. pid for persons and poses, oid for objects
        :param original_objects: array where objects are searched. persons array for persons or poses, objects array for objects
        :returns: Numpy array with all objects annotations for object_id in json format if exists, empty array otherwise
        """
        # If we need to look for personAIK or poseAIK, only 1 element is allowed in the obj_type array
        # to avoid error because persons and poses has the same pid
        if 'personAIK' in obj_types or 'poseAIK' in obj_types:
            assert len(obj_types) == 1, "Only 1 object type admitted when getting objects in frame"

        annotations = []
        for frame in range(self.get_total_frames()):
            annotated, real_frame, next_frame = self._get_real_frame(frame)
            if annotated:
                for a in original_objects[real_frame]:
                    if a[id_type] == object_id and a['type'] in obj_types and len(a['location']) != 0:
                        a['frame'] = frame
                        annotations.append(a)
            else:
                # Interpolate between two frames
                frame_annotation = original_objects[real_frame]
                p1, p2 = {}, {}
                for a in frame_annotation:
                    if a[id_type] == object_id and a['type'] in obj_types:
                        p1 = a
                        break
                frame_annotation = original_objects[next_frame]
                for a in frame_annotation:
                    if a[id_type] == object_id and a['type'] in obj_types:
                        p2 = a
                        break
                # Interpolate only if the person exists in both frames
                if p1 and p2 and len(p1['location']) == len(p2['location']) and len(p1['location']) != 0:
                    interpolated_person = self._interpolate(p1['location'], p2['location'])
                    object_json = {
                        'type': p1['type'],
                        'location': interpolated_person.tolist(),
                        id_type: p1[id_type],
                        'frame': frame
                    }
                    # Add labels to interpolated object if it exists in other frames
                    if 'labels' in p1:
                        object_json['labels'] = p1['labels']
                    annotations.append(object_json)
        return np.array(annotations)

    def _get_annotations_for_person_or_pose(self, person_id: int, obj_type: str) -> np.ndarray:
        """
            Gets all annotations for person_id and obj_type in the whole dataset.

        :param person_id: person identifier
        :param obj_type: poseAIK or personAIK
        :returns: Numpy array with all person or pose annotations for person_id in json format if exists, empty array otherwise
        """
        annotations = []
        for frame in range(self.get_total_frames()):
            annotated, real_frame, next_frame = self._get_real_frame(frame)
            if annotated:
                for a in self.persons[real_frame]:
                    if a['pid'] == person_id and a['type'] == obj_type and len(a['location']) != 0:
                        a['frame'] = frame
                        annotations.append(a)
            else:
                # Interpolate between two frames
                frame_annotation = self.persons[real_frame]
                p1, p2 = {}, {}
                for a in frame_annotation:
                    if a['pid'] == person_id and a['type'] == obj_type:
                        p1 = a
                        break
                frame_annotation = self.persons[next_frame]
                for a in frame_annotation:
                    if a['pid'] == person_id and a['type'] == obj_type:
                        p2 = a
                        break
                # Interpolate only if the person exists in both frames
                if p1 and p2 and len(p1['location']) == len(p2['location']) and len(p1['location']) != 0:
                    interpolated_person = self._interpolate(p1['location'], p2['location'])
                    person_json = {
                        'type': p1['type'],
                        'location': interpolated_person.tolist(),
                        'pid': p1['pid'],
                        'frame': frame
                    }
                    annotations.append(person_json)
        return np.array(annotations)

