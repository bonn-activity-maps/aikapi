# Interface for accessing the AIK dataset.

import os
import subprocess
import json
import numpy as np
import cv2
import shutil
from typing import Optional, Tuple, Union, List, TypeVar
from PythonAPI.utils.camera import Camera

kp = TypeVar('kp', float, float, float)


class AIK:

    def __init__(self, dataset_dir: str, dataset_name: str, image_format: str = 'png') -> None:
        """
            Constructor for AIK class for reading and visualizing annotations.

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
        self.persons, self.objects, self.activities, self.ids = self._read_annotations()

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
            unroll = subprocess.run(["ffmpeg", "-i", os.path.join(self.dataset_dir, video_file), "-qscale:v", "2",
                                     os.path.join(camera_dir, self.frame_format + "." + self.image_format)])
        else:
            unroll = subprocess.run(["ffmpeg", "-i", os.path.join(self.dataset_dir, video_file),
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
        num_frames = data['valid_frames'][-1] * 2
        return data['n_cameras'], num_frames

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

    def _read_annotations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            Reads persons, objects, actions and ids information from file.

        :returns: Persons, objects, activities and ids numpy arrays with information in json format
        """
        print('Loading annotations...')
        with open(os.path.join(self.dataset_dir, self.dataset_name + '_unroll.json')) as f:
            json_data = json.load(f)

        # Process and separate data into persons, objects and activities
        persons, ids = self._process_persons(json_data['persons'])
        del json_data['persons']

        # print('Processing objects...')
        objects = np.array([])
        # for d in json_data['objects']:
        #     objects.append(d)
        del json_data['objects']

        activities = self._process_activities(json_data['actions'])
        del json_data['actions']

        return persons, objects, activities, ids

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

            # Add persons in current frame
            persons_in_frame = d['persons']
            
            # Get the ids of the annotated persons
            for person in persons_in_frame:
                ids.append(person['pid']) 
            
            persons.append(persons_in_frame)
            last_frame = frame
        
        # Make sure that there are no repeated IDs
        ids = np.unique(ids)

        return np.array(persons, dtype=object), ids

    def _process_activities(self, activities_json: np.ndarray) -> np.ndarray:
        """
            Process activities json and order by person id.

        :param activities_json: numpy array with jsons that contain activities information
        :returns: numpy array with activities info ordered by person id
        """
        print('Processing actions...')
        activities = [[] for i in range(self.max_persons)]  # Initialize with empty activities

        for d in activities_json:
            pid = d['pid']
            del d['pid']
            activities[pid].append(d)
        return np.array(activities, dtype=object)

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
        return self._get_persons_or_poses_in_frame(frame, 'poseAIK')

    def get_persons_in_frame(self, frame: int) -> np.ndarray:
        """
            Gets all persons annotated in the given frame. Interpolates keypoints if the frame is not annotated

        :param frame: frame number
        :returns: Persons annotated in the given frame in json format
        """
        return self._get_persons_or_poses_in_frame(frame, 'personAIK')

    def _get_persons_or_poses_in_frame(self, frame: int, obj_type: str) -> np.ndarray:
        """
            Gets all persons or poses annotated in the given frame. Interpolates keypoints if the frame is not annotated

        :param frame: frame number
        :param obj_type: poseAIK or personAIK
        :returns: Persons or poses annotated in the given frame in json format
        """
        annotated, real_frame, next_frame = self._get_real_frame(frame)
        if annotated:
            objects = []
            for a in self.persons[real_frame]:
                if a['type'] == obj_type:
                    objects.append(a)
            return np.array(objects)
        else:
            persons = []
            # Interpolate all persons
            for p1 in self.persons[real_frame]:
                for p2 in self.persons[next_frame]:
                    if p1['pid'] == p2['pid'] and len(p1['location']) == len(p2['location']) \
                            and p1['type'] == obj_type and p2['type'] == obj_type:
                        interpolated_person = self._interpolate(p1['location'], p2['location'])
                        person_json = {
                            'pid': p1['pid'],
                            'location': interpolated_person.tolist(),
                            'type': p1['type']
                        }
                        persons.append(person_json)
                        break
            return np.array(persons)

    def get_person_in_frame(self, frame: int, person_id: int) -> np.ndarray:
        """
            Gets person annotation for person_id in the given frame.

        :param frame: frame number
        :param person_id: person identifier
        :returns: Person annotated in the given frame in json format if exists, empty array otherwise
        """
        return self._get_person_or_pose_in_frame(frame, person_id, 'personAIK')

    def get_pose_in_frame(self, frame: int, person_id: int) -> np.ndarray:
        """
            Gets pose annotation for person_id in the given frame.

        :param frame: frame number
        :param person_id: person identifier
        :returns: Pose in json format if exists, empty array otherwise
        """
        return self._get_person_or_pose_in_frame(frame, person_id, 'poseAIK')

    def _get_person_or_pose_in_frame(self, frame: int, person_id: int, obj_type: str) -> np.ndarray:
        """
            Gets annotation for person_id in the given frame.

        :param frame: frame number
        :param person_id: person identifier
        :param obj_type: poseAIK or personAIK
        :returns: Person/Pose annotated in the given frame in json format if exists, empty array otherwise
        """
        annotated, real_frame, next_frame = self._get_real_frame(frame)
        if annotated:
            frame_annotation = self.persons[real_frame]
            for a in frame_annotation:
                if a['pid'] == person_id and a['type'] == obj_type:
                    # Substitute empty lists with None
                    pts3d = [x if x else [None, None, None] for x in a['location']]
                    return np.array(pts3d)
        else:
            p1, p2 = [], []
            # Search keypoints in previous and next frame to interpolate
            frame_annotation = self.persons[real_frame]
            for a in frame_annotation:
                if a['pid'] == person_id and a['type'] == obj_type:
                    p1 = a['location']
                    break
            frame_annotation = self.persons[next_frame]
            for a in frame_annotation:
                if a['pid'] == person_id and a['type'] == obj_type:
                    p2 = a['location']
                    break
            # Interpolate only if the person exists in both frames
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

        # Get the paths to all cameras inside the videos folder
        cameras_paths = [os.path.join(self.videos_dir, name) for name in os.listdir(self.videos_dir) if os.path.isdir(os.path.join(self.videos_dir,name))]
        
        # Get the frame_name image from those paths
        images = []

        for path in cameras_paths:
            image = cv2.imread(os.path.join(path, frame_name), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        print("Images of frame ", frame, " retrieved.")
        return np.array(images)
    
    def get_person_ids(self) -> np.ndarray:
        """
            Returns the existing person ids in the dataset

        :returns: Numpy array with the existing IDs of the persons in the dataset
        """
        return self.ids

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



