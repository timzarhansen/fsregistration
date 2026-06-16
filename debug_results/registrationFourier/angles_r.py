import numpy as np


def angles_r(R, str_input):
    # solve a 3d rotation matrix R of the form Ra(th1)*Rb(th2)*Rc(th3)
    # for angles th1,th2,th3; where each of a,b c are one of x,y,z.
    # input str is a three-letter string of rotation axes, such as 'yzx'.
    # consecutive rotations along the same axis such as 'xxy' are not allowed.
    # 1st and 3rd rotations along different axes such as 'yzx' are tait-bryan,
    # along the same axis such as 'xzx' are euler.  12 possibilities in all.
    # Output is the vector [th1 th2 th3] and angles are in DEGREES,
    # with -180<(th1,th3)<180;  -90<th2<90 (tait-bryan), 0<th2<180 (euler).

    theta = np.zeros(3)

    # similarity transform matrices
    By = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])  # x<-->z  y(th)-->y(-th)
    Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # y rotation by 90 deg
    C = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # cycic, x-->y-->z-->x
    signy = 1

    # transform to RaRyRb
    if str_input[1] == 'x':
        R = C @ R @ np.linalg.inv(C)
    elif str_input[1] == 'z':
        R = np.linalg.inv(C) @ R @ C

    # tait-bryan, transform to RxRyRz
    if all(str_input == 'xzy') or all(str_input == 'yxz') or all(str_input == 'zyx'):
        R = By @ R @ np.linalg.inv(By)
        signy = -1

    # euler, transform to RxRyRx
    if all(str_input == 'xzx') or all(str_input == 'yxy') or all(str_input == 'zyz'):
        R = Ry90 @ R @ np.linalg.inv(Ry90)

    if str_input[0] != str_input[2]:  # tait-bryan
        theta[1] = signy * np.arcsin(R[0, 2]) * 180 / np.pi
        theta[0] = np.arctan2(-R[1, 2], R[2, 2]) * 180 / np.pi
        theta[2] = np.arctan2(-R[0, 1], R[0, 0]) * 180 / np.pi
    else:  # euler
        theta[1] = np.arccos(R[0, 0]) * 180 / np.pi
        theta[0] = np.arctan2(R[1, 0], -R[2, 0]) * 180 / np.pi
        theta[2] = np.arctan2(R[0, 1], R[0, 2]) * 180 / np.pi

    return theta
