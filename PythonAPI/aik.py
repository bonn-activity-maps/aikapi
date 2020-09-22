# Interface for accessing the AIK dataset.

import os
import subprocess
import json
import numpy as np
import cv2
import shutil

class AIK:

    def __init__(self, dataset_dir, dataset_name, image_format='png'):
        '''
        Constructor AIK class for reading and visualizing annotations.
        :param dataset_dir (str): location of the dataset
        :param dataset_name (str): name of the dataset
        :param image_format (str): format of images to extract video frames, png or jpeg (png by default)
        '''
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(dataset_dir, self.dataset_name)
        self.cameras_dir = os.path.join(self.dataset_dir, 'cameras')
        self.videos_dir = os.path.join(self.dataset_dir, 'videos')

        # Check if dataset exists in the directory
        assert os.path.isdir(self.dataset_dir), "The dataset directory %r does not exist" % dataset_dir
        assert os.path.isdir(self.cameras_dir), "The dataset is not complete, cameras information is missing"

        assert image_format in ['png', 'jpeg'], "The image format should be png or jpeg"

        self.num_cameras = self._read_dataset_info()
        self.frame_format = "frame%09d"
        self.max_persons = 50

        # Create videos directory and unroll videos if the directory videos does not exist
        if not os.path.exists(self.videos_dir):
            os.mkdir(self.videos_dir)
            self.unroll_videos(image_format)
        # If the image format has changed, remove frames and unroll again
        elif os.path.splitext(os.listdir(os.path.join(self.videos_dir, 'camera00'))[0])[1].split('.')[1] != image_format:
            print('The image format has changed.')
            print('    Removing old files...')
            try:
                shutil.rmtree(self.videos_dir)
            except OSError as e:
                print("Error: %s : %s" % (self.videos_dir, e.strerror))
            os.mkdir(self.videos_dir)
            self.unroll_videos(image_format)


        # Load info to memory
        self.calibration_params = self._read_calibration_params()
        self.persons, self.objects, self.activities = self._read_annotations()
        print('Finish loading dataset')

    def unroll_videos(self, img_format):
        '''
        Unroll all the videos and store them in the videos folder. This folder contains 12 folders (cameraXX) with
        the unrolled frames from each camera.
        :param img_format (str): format of images to extract video frames, png or jpeg
        :return:
        '''
        print('Unrolling videos. This may take a while...')

        for c in range(self.num_cameras):
            video = self.dataset_name + '_' + str(c).zfill(2) + '.mp4'

            # Create camera directory to store all frames
            camera = 'camera' + str(c).zfill(2)
            camera_dir = os.path.join(self.videos_dir, camera)
            os.mkdir(camera_dir)

            unroll = subprocess.run(["ffmpeg", "-i", os.path.join(self.dataset_dir, video),
                                     os.path.join(camera_dir, self.frame_format+"."+img_format)])
            # print("The exit code was: %d" % unroll.returncode)

    def _read_dataset_info(self):
        '''
        Read general dataset information.
        :return: Number of cameras
        '''
        print('Reading dataset information...')
        dataset_file = os.path.join(self.dataset_dir, 'dataset.json')
        assert os.path.isfile(dataset_file), "The dataset is not complete, the dataset.json file is missing"

        with open(dataset_file) as f:
            data = json.load(f)
        return data['n_cameras']

    def _read_calibration_params(self):
        '''
        Read camera calibration parameters for each camera.
        :return: Numpy array with calibration parameters
        '''
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
        return np.array(cameras_data)

    def _read_annotations(self):
        '''
        Read persons, objects and actions information from file.
        :return: Persons, objects and activities numpy arrays with information in json format
        '''
        print('Loading annotations...')
        with open(os.path.join(self.dataset_dir, self.dataset_name + '_unroll.json')) as f:
            json_data = json.load(f)

        # Process and separate data into persons, objects and activities
        persons = self._process_persons(json_data['persons'])
        del json_data['persons']

        # print('Processing objects...')
        objects = []
        # for d in json_data['objects']:
        #     objects.append(d)
        del json_data['objects']

        activities = self._process_activities(json_data['actions'])
        del json_data['actions']

        return persons, objects, activities

    def _process_persons(self, persons_json):
        '''
        Process persons json and order by frame.
        :param persons_json : json with persons information
        :return: numpy array with person info in json format for each frame
        '''
        print('Processing persons...')
        persons = []
        last_frame = -1
        for d in persons_json:
            frame = d['frame']

            # Complete if there are empty frames in between
            if frame > last_frame + 1:
                for i in range(last_frame+1, frame):
                    persons.append([])

            # Add persons in current frame
            persons_in_frame = d['persons']
            persons.append(persons_in_frame)
            last_frame = frame
        return np.array(persons)

    def _process_activities(self, activities_json):
        '''
        Process activities json and order by person id.
        :param activities_json : json with activities information
        :return: array with activities info ordered by person id
        '''
        print('Processing actions...')
        activities = [[] for i in range(self.max_persons)]  # Initialize with empty activities

        for d in activities_json:
            pid = d['pid']
            del d['pid']
            activities[pid].append(d)
        return np.array(activities)

    def _get_real_frame(self, frame):
        '''
        Get the real frames or frames that need to be interpolated
        :param frame: number of frame
        :return: True if the dataset contains the info about the frame, False if we have to interpolate.
                The real frame if it is stored in the dataset
                The previous and posterior frames if the information about that frame is not stored
        '''
        # Even -> the frame info is not contained in the dataset
        if (frame % 2) == 0:
            real_frame = frame//2
            return False, real_frame, real_frame+1
        else:     # Odd -> return real frame
            return True, frame//2 + 1, -1

    def get_calibration_params(self, video, frame):
        '''
        Get calibration parameters that satisfy given filter conditions.
        :param video (int): video number
        :param frame (int): frame number
        :return: calibration parameters in json format with K, rvec, tvec, distCoef, w and h
        '''
        _, real_frame, _ = self._get_real_frame(frame)
        for p in self.calibration_params[video]:
            _, real_start_frame, _ = self._get_real_frame(p['start_frame'])
            _, real_end_frame, _ = self._get_real_frame(p['end_frame'])
            if real_start_frame <= real_frame <= real_end_frame:
                return p
        return 'Frame ' + str(frame) + ' does not exist in dataset ' + self.dataset_name

    def _interpolate(self, kps1, kps2):
        '''
            Interpolate all keypoints the frame in between kps1 and kps2
        :param kps1: keypoints from first frame
        :param kps2: keypoints from second frame
        :return: list with interpolated keypoints
        '''
        interpolated_kps = []
        for i in range(len(kps1)):
            # If one of the two points is empty -> Not interpolate
            if len(kps1[i]) != 0 and len(kps2[i]) != 0:
                interpolated_coords = np.linspace(np.array(kps1[i]), np.array(kps2[i]), num=3).tolist()
                interpolated_kps.append(interpolated_coords[1])
            else:
                interpolated_kps.append([])
        return interpolated_kps

    def get_persons_in_frame(self, frame):
        '''
        Get all persons annotated in given frame. Interpolate keypoints if the frame is not annotated
        :param frame (int): frame number
        :return: Persons in json format
        '''
        annotated, real_frame, next_frame = self._get_real_frame(frame)
        if annotated:
            return np.array(self.persons[real_frame])
        else:
            persons = []
            # Interpolate all persons
            for p1 in self.persons[real_frame]:
                for p2 in self.persons[next_frame]:
                    if p1['pid'] == p2['pid'] and len(p1['location']) == len(p2['location']):
                        interpolated_person = self._interpolate(p1['location'], p2['location'])
                        person_json = {
                            'pid': p1['pid'],
                            'location': interpolated_person
                        }
                        persons.append(person_json)
                        break
            return np.array(persons)

    # TODO: check types in export

    def get_person_in_frame(self, frame, person_id):
        '''
        Get annotation for person_id in given frame.
        :param frame (int): frame number
        :param person_id (int): person identifier
        :return: Person in json format if exists, info message otherwise
        '''
        annotated, real_frame, next_frame = self._get_real_frame(frame)
        if annotated:
            frame_annotation = self.persons[real_frame]
            for a in frame_annotation:
                if a['pid'] == person_id:
                    return a['location']
        else:
            p1, p2 = [], []
            # Search keypoints in previous and next frame to interpolate
            frame_annotation = self.persons[real_frame]
            for a in frame_annotation:
                if a['pid'] == person_id:
                    p1 = a['location']
                    break
            frame_annotation = self.persons[next_frame]
            for a in frame_annotation:
                if a['pid'] == person_id:
                    p2 = a['location']
                    break
            # Interpolate only if the person exists in both frames
            if p1 and p2:
                return self._interpolate(p1, p2)

        return 'Person ' + str(person_id) + ' is not annotated in frame ' + str(frame)

    def get_activities_for_person(self, pid):
        '''
        Get all activities annotated for person with pid.
        :param pid (int): person identifier
        :return: Numpy array with activities in json format
        '''
        return self.activities[pid]

    def get_images_in_frame(self, frame):
        '''
        Get camera images corresponding to the frame.
        :param frame (int): frame number
        :return: Array with images in numpy_array format if the frame exists, info message otherwise
        '''

        print("Searching for images of frame ", frame, "...")
        # Create the string of the name of the frame that we are going to search for in all camera folders
        frame_name = "frame" + ''.zfill(9)
        frame_string = str(frame)
        number_of_chars = len(frame_string)
        frame_name = frame_name[:-number_of_chars] + frame_string + ".png"
        
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
        return images



