import cv2
import numpy as np
from glob import glob
import os
import pathlib
import requests
import json
import time
from watchdog.observers import Observer
from watchdog.events import *

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

def request_test():
    data = {
        "MessageType": "GetTaskInfo",
        "DeviceId": "Camera_0001"
    }
    imgdata = {
        "MessageType": "GetImageInfo",
        "TaskId": "20210830141816",
        "Page": "1",
        "PageSize": "1000"
    }

    json_file = json.dumps(data)
    response = requests.get("http://124.226.212.192:6000", data=json_file, timeout=10)
    res = response.content.decode("utf-8")
    t = eval(response.json())
    r = eval(res)
    response.iter_content()
    url = r["Data"][5]["Url"]
    img_d = requests.get(url)
    h = img_d.headers
    img = img_d.content
    with open("test.jpg", "wb") as f:
        f.write(img)


    t = bytearray(img)
    s = cv2.imdecode(np.array(t, dtype="uint8"), cv2.IMREAD_UNCHANGED)
    cv2.imshow("", s)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    pass

class mywatch(FileSystemEventHandler):
    def __init__(self):
        super(mywatch, self).__init__()
        # FileSystemEventHandler.__init__(self)

    def on_created(self, event):
        if event.is_directory:
            print("dir")
            return 0
        else:
            print("file")
            return 1
        print(event.src_path)


if __name__ == "__main__":
    request_test()
    path = r"G:\python\qt5\data"

    h = mywatch()
    observer = Observer()
    observer.schedule(h, path, recursive=True)
    observer.start()
    print("sds")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    print("ttt")
    observer.join()

    print("sss")












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
