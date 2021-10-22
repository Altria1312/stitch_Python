import cv2
import piexif
import numpy as np
from glob import glob
import os
import pathlib

import json
import time
import pandas as pd




def show(img, name="", t=0):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(t)
    cv2.destroyAllWindows()


def onmouse(event, x, y, flags, params):
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        params[y, x] = 1
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON:
        params[y, x] = 0

def dll():
    path = r"G:\PCL 1.12.0\lib\*0d.lib"
    save = []
    for name in glob(path):
        file = pathlib.Path(name)
        save.append(file.name+'\n')

    with open("./dll.txt", "w") as f:
        f.writelines(save)

# def request_test():
#     data = {
#         "MessageType": "GetTaskInfo",
#         "DeviceId": "Camera_0001"
#     }
#     imgdata = {
#         "MessageType": "GetImageInfo",
#         "TaskId": "20210830141816",
#         "Page": "1",
#         "PageSize": "1000"
#     }
#
#     json_file = json.dumps(data)
#     response = requests.get("http://124.226.212.192:6000", data=json_file, timeout=10)
#     res = response.content.decode("utf-8")
#     t = eval(response.json())
#     r = eval(res)
#     response.iter_content()
#     url = r["Data"][5]["Url"]
#     img_d = requests.get(url)
#     h = img_d.headers
#     img = img_d.content
#     with open("test.jpg", "wb") as f:
#         f.write(img)
#
#
#     t = bytearray(img)
#     s = cv2.imdecode(np.array(t, dtype="uint8"), cv2.IMREAD_UNCHANGED)
#     cv2.imshow("", s)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
#     pass
#
# class mywatch(FileSystemEventHandler):
#     def __init__(self):
#         super(mywatch, self).__init__()
#         # FileSystemEventHandler.__init__(self)
#
#     def on_created(self, event):
#         if event.is_directory:
#             print("dir")
#             return 0
#         else:
#             print("file")
#             return 1
#         print(event.src_path)

class A:
    def __init__(self):
        self.a = 1

    def test(self):
        print(self.a)

def cal_angle():
    path = "./121-1.txt"
    file = pd.read_table(path, comment="#", header=None, skiprows=1)

    data = file.values

    coord = data[data[:, 3] < 20, 8:10] * 1e7
    coord = coord.astype(np.int32)
    orient = []
    for i in range(1, coord.shape[0]-1):
        v1 = coord[i] - coord[i-1]
        v2 = coord[i+1] - coord[i]
        ang = v1[0] * v2[0] + v1[1] * v2[1]
        orient.append(ang)
    pass

def get_img_ori(img_list, n):
    coor = np.zeros((n, 2))
    root = "G:\\data\\20210817002"
    for i in range(n):
        path = os.path.join(root, img_list[i])
        exif_dict = piexif.load(path)
        lat = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
        lon = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]

        # t = lat[1][0]/lat[1][1] / 60
        # tt = lat[1][0]/lat[1][1]
        coor[i, 0] = (lat[0][0]/lat[0][1] + lat[1][0]/lat[1][1] / 60 + lat[2][0]/lat[2][1] / 3600) * 1e7
        coor[i, 1] = (lon[0][0]/lon[0][1] + lon[1][0]/lon[1][1] / 60 + lon[2][0]/lon[2][1] / 3600) * 1e7

    ang = np.zeros(n)
    v = coor[1:] - coor[:-1]
    for i in range(1, n-1):
        ang[i] = v[i, 0]*v[i-1, 0] + v[i, 1]*v[i-1, 1]

    is_change = ang < 0
    pass


if __name__ == "__main__":
    img_list = os.listdir("G:\\data\\20210817002")
    get_img_ori(img_list, 30)










    # path = "./20210830113621.jpg"
    # img = cv2.imread(path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # # show(gray)
    # # # _, otsu = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    # # otsu = np.where((gray>150) & (gray<155), 255, 0)
    # # otsu = otsu.astype(np.uint8)
    # # show(otsu)
    # # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # # plt.plot(hist)
    # # plt.show()
    #
    # rect = (105, 170, 780, 490)
    # mask = np.ones_like(gray, np.uint8) * 2
    #
    # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback("mask", onmouse, mask)
    #
    # while True:
    #     cv2.imshow("mask", img)
    #     if cv2.waitKey(30) & 0xff == 27:
    #         break
    #
    # cv2.destroyAllWindows()
    #
    # bgdModel = np.zeros((1, 65), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    #
    # cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
    #
    # # mask_show = mask * 80
    # mask_show = np.where((mask == 1) | (mask == 3), 1, 0)
    # mask_show = mask_show.astype(np.uint8)
    #
    # img = cv2.bitwise_and(img, img, mask=mask_show)
    # show(img)
    #
    # gray = cv2.GaussianBlur(gray, (3,3), 1)
    # dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    # dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    # grad = cv2.magnitude(dx, dy)
    # grad *= mask_show
    # grad = grad.astype(np.uint8)
    # ker = np.ones((3, 3))
    # grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, ker)
    # show(grad)
    #
    # _, test = cv2.threshold(grad, 150, 255, cv2.THRESH_BINARY_INV)
    # show(test)
    #
    #
    # pass
