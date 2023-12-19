import numpy as np 

# 给定fid文件指针，在文件中读取一个以name开头的MxN的矩阵。
def readVariable(fid, name, M, N):
    # rewind
    fid.seek(0, 0)

    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success == 0:
        return None

    # fill matrix
    line = line.replace("%s:" % name, "")
    line = line.split()
    assert len(line) == M * N
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat

def load_intrinsics(intrinsic_file, cam_id):
    if True:
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(" ")
            # intrisics of rectified cameras (including effect of R_rect_xxx)
            if line[0] == "P_rect_%02d:" % cam_id:
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                intrinsic_loaded = True
            elif line[0] == "R_rect_%02d:" % cam_id:
                R_rect = np.eye(4)
                R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
            # 图像的高度和宽度
            elif line[0] == "S_rect_%02d:" % cam_id:
                width = int(float(line[1]))
                height = int(float(line[2]))
        return K, R_rect

# 加载从unrectified camera坐标系（非图像坐标）（00，01，fisheye03，fisheye04）到GMU坐标系的变换矩阵。
def loadCalibrationCameraToPose(filename):
    # open file
    fid = open(filename, "r")

    # read variables
    Tr = {}
    cameras = ["image_00", "image_01", "image_02", "image_03"]
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))

    # close file
    fid.close()
    return Tr

# 加载从GMU coordinate到world coordinate的配置
def load_poses(pose_file):
    poses = np.loadtxt(pose_file)
    frames = poses[:, 0].astype(np.int32)
    poses = np.reshape(poses[:, 1:], (-1, 4, 4))
    print("Number of posed frames %d" % len(frames))
    return frames, poses

# 将相机在世界坐标系中平移和变换坐标的尺度大小。
def transform_pose(c2w, translate, scale):
    cam_center = c2w[:3, 3]
    cam_center = (cam_center + translate) * scale
    c2w[:3, 3] = cam_center
    return c2w
