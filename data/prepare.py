import os
import numpy as np
import cv2
from tqdm import tqdm


def calculate_pitch_yaw_roll(landmarks_2D,
                             cam_w=256,
                             cam_h=256,
                             radians=False):
    """ Return the pitch  yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """

    assert landmarks_2D is not None, 'landmarks_2D is None'

    # Estimated camera matrix values.
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    # dlib (68 landmark) trached points
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    # wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    # X-Y-Z with X pointing forward and Y on the left and Z up.
    # The X-Y-Z coordinates used are like the standard coordinates of ROS (robotic operative system)
    # OpenCV uses the reference usually used in computer vision:
    # X points to the right, Y down, Z to the front
    landmarks_3D = np.float32([
        [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT,
        [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT,
        [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
        [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
        [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
        [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
        [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
        [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
        [2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
        [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
        [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
        [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
        [0.000000, -7.415691, 4.070434],  # CHIN
    ])
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)

    # Applying the PnP solver to find the 3D pose of the head from the 2D position of the landmarks.
    # retval - bool
    # rvec - Output rotation vector that, together with tvec, brings points from the world coordinate system to the camera coordinate system.
    # tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix,
                                 camera_distortion)
    # Get as input the rotational vector, Return a rotational matrix

    # const double PI = 3.141592653;
    # double thetaz = atan2(r21, r11) / PI * 180;
    # double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / PI * 180;
    # double thetax = atan2(r32, r33) / PI * 180;

    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    return map(lambda k: k[0],
               euler_angles)  # euler_angles contain (pitch, yaw, roll)



def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = (1-alpha)*center[1] + beta*center[0]
    landmark_ = np.asarray([(M[0, 0]*x + M[0, 1]*y + M[0, 2],
                             M[1, 0]*x + M[1, 1]*y + M[1, 2]) for (x, y) in landmark])
    return M, landmark_


class ImageData:
    def __init__(self, line, image_dir, image_size=112, num_key_points=106):
        self.image_size = image_size
        self.num_key_points = num_key_points
        line = line.strip().split()

        assert (len(line) == (1 + 4 + 2 * self.num_key_points)), "the length of the line ({}) does not match".format(len(line))
        self.list = line
        self.image_name = self.list[0]
        self.image_path = os.path.join(image_dir, self.image_name)
        self.box = np.asarray(list(map(int, self.list[1:5])), dtype=np.int32)
        self.landmark = np.asarray(list(map(float, self.list[5:])), dtype=np.float32).reshape(-1, 2)

        self.image = None
        self.images = []
        self.landmarks = []

    def load_data(self, is_train, repeat):
        xy = np.min(self.landmark, axis=0).astype(np.int32)
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        boxsize = int(np.max(wh) * 1.2)
        xy = center - boxsize // 2
        zz = xy + boxsize
        x1, y1 = xy
        x2, y2 = zz

        assert os.path.exists(self.image_path), "{} does not exists".format(self.image_path)
        self.image = cv2.imread(self.image_path)
        height, width, _ = self.image.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        image_target = self.image[y1:y2, x1:x2]

        dx = max(0, -x1)
        dy = max(0, -y1)
        edx = max(0, x2 - width)
        edy = max(0, y2 - height)

        if dx > 0 or dy > 0 or edx >0 or edy > 0:
            image_target = cv2.copyMakeBorder(image_target, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if image_target.shape[0] == 0 or image_target.shape[1] == 0:
            exit()
        image_target = cv2.resize(image_target, (self.image_size, self.image_size))
        landmark = (self.landmark - xy) / boxsize #归一化
        assert (landmark>=0).all(), str(landmark) + str([dx, dy])
        assert (landmark<=1).all(), str(landmark) + str([dx, dy]) # ???
        self.images.append(image_target)
        self.landmarks.append(landmark)

        if is_train:
            while len(self.images) < repeat:
                angle = np.random.randint(-30, 30)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
                cy = cy + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
                M, landmark = rotate(angle, (cx, cy), self.landmark)

                image_rotate = cv2.warpAffine(self.image, M, (int(self.image.shape[1]*1.1), int(self.image.shape[0] * 1.1)))

                wh = np.ptp(landmark, axis=0).astype(np.int32)+1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh)*1.25))
                xy = np.asarray([cx - size//2, cy - size//2], dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark<0).any() or (landmark>1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = image_rotate.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                dx = max(0, -x1)
                dy = max(0, -y1)
                edx = max(0, x2-width)
                edy = max(0, y2-height)

                image_target = image_rotate[y1:y2, x1:x2]
                if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                    image_target = cv2.copyMakeBorder(image_target, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                image_target = cv2.resize(image_target, (self.image_size, self.image_size))

                self.images.append(image_target)
                self.landmarks.append(landmark)

    def save_data(self, img_dir, save_dir):
        labels = []
        TRACKED_POINTS = [43,46,97,101,35,39,89,93,77,83,52,61,53,0]
        for i, (image, landmark) in enumerate(zip(self.images, self.landmarks)):
            assert landmark.shape == (106, 2)
            images_prefix = self.image_name.split('.')[0]
            save_path_img = os.path.join(save_dir, 'images', images_prefix + '_' + str(i) + '.jpg')
            # assert not os.path.exists(save_path_img), save_path_img + ' has existed'
            cv2.imwrite(save_path_img, image)

            euler_angles_landmark = []
            for index in TRACKED_POINTS:
                euler_angles_landmark.append(landmark[index])
            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))
            landmark_str = ' '.join(list(map(str, landmark.reshape(-1).tolist())))

            label = '{} {} {} \n'.format(save_path_img, landmark_str, euler_angles_str)
            labels.append(label)
            return labels


def get_dataset_list(root_dir, img_dirs, annotation_dirs, is_train, test_ratio=0.1):
    labels = []
    annotation_txts = os.listdir(annotation_dirs)
    imgs_dir = os.listdir(img_dirs)
    save_dir_train = os.path.join(root_dir, 'train')
    save_dir_test = os.path.join(root_dir, 'test')
    if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)
    if not os.path.exists(save_dir_test):
        os.makedirs(save_dir_test)
    pbar1 = tqdm(total=len(imgs_dir))
    for _, (img_dir, annotation_txt) in enumerate(zip(imgs_dir,annotation_txts)):
        with open(os.path.join(root_dir, annotation_dirs,annotation_txt), 'r') as f:
            lines = f.readlines()
            index = int(len(lines) * (1-test_ratio))
            if is_train:
                lines = lines[0:index]
            else:
                lines = lines[index:]
            pbar2 = tqdm(total=len(lines))
            for i, line in enumerate(lines):
                    Img = ImageData(line, os.path.join(img_dirs,img_dir))
                    Img.load_data(is_train, 5)
                    label = Img.save_data(img_dir, save_dir_train if is_train else save_dir_test)
                    labels.append(label)
                    pbar2.update(1)
        pbar1.update(1)

    with open(os.path.join(save_dir_train if is_train else save_dir_test, 'landmarks.txt'), 'w') as f:
        for label in labels:
            f.writelines(label)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    img_dirs = 'WFLW_images'
    annotation_dirs = 'WFLW_annotations_106'
    img_dirs = os.path.join(root_dir, img_dirs)
    annotation_dirs = os.path.join(root_dir, annotation_dirs)
    get_dataset_list(root_dir, img_dirs, annotation_dirs, is_train=True)
    get_dataset_list(root_dir, img_dirs, annotation_dirs, is_train=False)
