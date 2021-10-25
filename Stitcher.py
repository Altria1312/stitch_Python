import cv2
import numpy as np
import piexif
import time
import os

class myStitcher:
    def __init__(self, img_list):
        self.img_list = img_list
        self.scale = 2
        self.match_threhold = 0.6
        self.matcher = cv2.FlannBasedMatcher()

    def show(self, img, name="", t=0):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey(t)
        cv2.destroyAllWindows()

    def detect_compute(self, img):
        if cv2.__version__ == "4.5.1":
            extractor = cv2.SIFT_create(1000)
        else:
            extractor = cv2.xfeatures2d.SURF_create()
            # extractor = cv2.xfeatures2d_SIFT.create()
            # extractor = cv2.ORB_create(500)
        return extractor.detectAndCompute(img, None)

    def generate_feature(self, img):
        h = int(self.h / self.scale)
        w = int(self.w / self.scale)
        # 缩放
        src1 = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        return self.detect_compute(src1)

    def keypoint_match(self, des1, des2):
        matches = self.matcher.knnMatch(des1, des2, k=2)
        matched = []
        for m, n in matches:
            if m.distance < self.match_threhold * n.distance:
                matched.append(m)

        return matched

    def reset_kpt_coord(self, matched, kpt1, kpt2):
        ori = np.zeros((len(matched), 2))
        target = np.zeros_like(ori)
        for i in range(len(matched)):
            ori[i, 0] = kpt1[matched[i].queryIdx].pt[0] * self.scale
            ori[i, 1] = kpt1[matched[i].queryIdx].pt[1] * self.scale
            # ori.append((x, y))

            target[i, 0] = kpt2[matched[i].trainIdx].pt[0] * self.scale
            target[i, 1] = kpt2[matched[i].trainIdx].pt[1] * self.scale
            # target.append((x, y))

        return ori, target

    def exif_parse(self, img_path):
        exif_dict = piexif.load(img_path)
        lat = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
        lon = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]

        # t = lat[1][0]/lat[1][1] / 60
        # tt = lat[1][0]/lat[1][1]
        x = (lat[0][0]/lat[0][1] + lat[1][0]/lat[1][1] / 60 + lat[2][0]/lat[2][1] / 3600) * 1e7
        y = (lon[0][0]/lon[0][1] + lon[1][0]/lon[1][1] / 60 + lon[2][0]/lon[2][1] / 3600) * 1e7

        return (x, y)

    def get_warp_shape(self, H, img):
        # 变换后的区域
        pts = np.array([[0, img.shape[1], img.shape[1], 0],
                        [0, 0, img.shape[0], img.shape[0]],
                        [1, 1, 1, 1]])
        pts_H = H @ pts
        pts_H /= pts_H[-1, :]

        xy_min = np.round(np.min(pts_H, axis=1)[:-1]).astype(np.int32)
        xy_max = np.round(np.max(pts_H, axis=1)[:-1]).astype(np.int32)

        left = 0 if xy_min[0] >= 0 else -xy_min[0]
        right = self.w if xy_max[0] <= self.w else xy_max[0]
        buttom = xy_max[1]
        return left, right, buttom

    def weighted_blend(self):
        pass

    def opt_seam(self, cut, img2):
        src1 = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        src2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        E_color = (src2 - src1) ** 2

        sob_x = np.array([[-2, 0, 2],
                          [-1, 0, 1],
                          [-2, 0, 2]])
        sob_y = np.array([[-2, -1, -2],
                          [0, 0, 0],
                          [2, 1, 2]])

        src1_x = cv2.filter2D(src1, cv2.CV_32F, sob_x)
        src1_y = cv2.filter2D(src1, cv2.CV_32F, sob_y)
        src2_x = cv2.filter2D(src2, cv2.CV_32F, sob_x)
        src2_y = cv2.filter2D(src2, cv2.CV_32F, sob_y)

        E_geometry = (src1_x - src2_x) * (src1_y - src2_y)

        self.E = E_color + E_geometry

        # 确定初始位置
        t = int(self.w/2)
        path_row = [i for i in range(self.h) if src1[i, t] > 0]
        path_row = np.array(path_row).reshape(-1, 1)
        self.path_row = np.tile(path_row[1:-1], self.w)
        self.min_val = np.min(path_row) + 1
        self.max_val = np.max(path_row)

        path_row = np.tile(path_row, self.w)
        self.path_row = path_row[1:-1]
        # 初始能量
        self.path_energy = np.squeeze(self.E[self.path_row[:,0], 0])

        # # 单线程
        # for i in range(1, w):
        #     mids = self.path_row[:, i-1]
        #     lefts = np.maximum(mids-1, self.min_val)
        #     rights = np.minimum(mids+1, self.max_val)
        #     temp = np.vstack([lefts, mids, rights])
        #     rg = np.arange(temp.shape[1])
        #
        #     mid_E = self.E[mids, i]
        #     left_E = self.E[lefts, i]
        #     right_E = self.E[rights, i]
        #
        #     temp_E = np.vstack([left_E, mid_E, right_E])
        #     idx = np.argmin(temp_E, axis=0)
        #     temp_E = temp_E[idx, rg]
        #
        #     self.path_energy = self.path_energy + temp_E
        #     idx = temp[idx, rg]
        #     self.path_row[:, i] = idx

        # 单线程2
        mids_E = self.E[self.min_val:self.max_val]
        lefts_E = self.E[self.min_val-1:self.max_val-1]
        rights_E = self.E[self.min_val+1:self.max_val+1]
        temp_E = cv2.merge([lefts_E, mids_E, rights_E])
        idx = np.argmin(temp_E, axis=-1)
        # rg_x = np.tile(np.arange(w), [self.path_row.shape[0], 1])
        rg_y = np.tile(np.arange(path_row.shape[0]), [self.w,1]).T
        # temp_E = temp_E[rg_y, rg_x-1, idx]
        idx -= 1
        rg_y[1:-1, 1:] += idx[:, 1:]
        rg_y[0] += 1
        rg_y[-1] -= 1

        path_row[1:-1, 1:] += idx[:, 1:]
        prev = rg_y[1:-1, 0]
        for i in range(1, self.w):
            self.path_row[:, i] = path_row[rg_y[prev, i], i]
            self.path_energy += self.E[self.path_row[:, i], i]
            prev = rg_y[prev, i]

        # # 最小割算法
        # graph = maxflow.GraphInt()
        # nodeids = graph.add_grid_nodes((max_val-min_val+1, w))
        # structure = np.array([[0, 0, 1],
        #                       [0, 0, 1],
        #                       [0, 0, 1]])
        # weights = E[min_val:max_val+1]
        # cv2.normalize(-weights, weights, alpha=0, beta=10000, norm_type=cv2.NORM_MINMAX)
        # graph.add_grid_edges(nodeids, weights, structure)
        #
        # s_t = weights.max()
        # graph.add_grid_tedges(nodeids[:, 0], 1000, 0)
        # graph.add_grid_tedges(nodeids[:, -1], 0, 1000)
        #
        #
        # tt = graph.maxflow()
        # res = graph.get_grid_segments(nodeids)
        # res = res.astype(np.uint8) * 255
        # ttt = res.max()
        # self.show(res)

        opt_idx = np.argmin(self.path_energy)
        opt_path = self.path_row[opt_idx]

        rg = np.tile(np.arange(self.h), [self.w, 1]).T
        mask = np.greater(rg, opt_path).astype(np.uint8)

        # rg = np.arange(h)
        # for j in range(w):
        #     col = rg > opt_path[j]
        #     self.mask[:, j] = self.mask[:, j] * col

        # self.show(self.mask * 255)

        self.mask = cv2.merge([mask, mask, mask])
        img2 = (1-self.mask) * img2
        cut = self.mask * cut + img2

        return cut

    def start(self):
        # 第一张
        orient = 0
        img = cv2.imread(self.img_list[0])
        self.h = img.shape[0]
        self.w = img.shape[1]

        kpt_1, des_1 = self.generate_feature(img)
        # 获取位置信息
        pos_pre = self.exif_parse(self.img_list[0])
        pos_cur = self.exif_parse(self.img_list[1])

        n = len(self.img_list)
        for id in range(1, n):
            next_img = cv2.imread(self.img_list[id])
            # 获取位置信息
            if id + 1 < n:
                pos_next = self.exif_parse(self.img_list[id+1])
                cos = (pos_cur[0] - pos_pre[0]) * (pos_next[0] - pos_cur[0]) + \
                      (pos_cur[1] - pos_pre[1]) * (pos_next[1] - pos_cur[1])
                orient = 0 if cos > 0 else 1

                pos_pre = pos_cur
                pos_cur = pos_next

            # 计算特征点
            kpt_2, des_2 = self.generate_feature(next_img)
            # 特征点匹配、筛选
            matched = self.keypoint_match(des_1, des_2)
            if len(matched) < 4: break
            # 还原特征点坐标
            src, dts = self.reset_kpt_coord(matched, kpt_1, kpt_2)

            # dts = dts + np.array([1000, 0])
            # if id != 1:
            #     src = src + np.array([1000, 0])

            # 计算映射矩阵
            H, _ = cv2.findHomography(src, dts, cv2.RANSAC)
            size = self.get_warp_shape(H, img)
            H[0, -1] += size[0]
            w_expand = size[1] + size[0]
            h_expand = size[2]
            # 变换
            img = cv2.warpPerspective(img, H, (w_expand, h_expand))

            cut = img[0:self.h, size[0]:self.w+size[0]]
            img[0:self.h, size[0]:size[0]+self.w] = self.opt_seam(cut, next_img)

            # for next loop
            kpt_1 = kpt_2
            des_1 = des_2

            # self.show(img)
        return img

    def all_feature(self):
        # 第一张
        orient = 0
        img = cv2.imread(self.img_list[0])
        self.h = img.shape[0]
        self.w = img.shape[1]

        img1 = img
        n = len(self.img_list)
        for id in range(1, n):
            kpt_1, des_1 = self.generate_feature(img1)
            # # 获取位置信息
            # pos_pre = self.exif_parse(self.img_list[0])
            # pos_cur = self.exif_parse(self.img_list[1])
            #
            # # 获取位置信息
            # if id + 1 < n:
            #     pos_next = self.exif_parse(self.img_list[id+1])
            #     cos = (pos_cur[0] - pos_pre[0]) * (pos_next[0] - pos_cur[0]) + \
            #           (pos_cur[1] - pos_pre[1]) * (pos_next[1] - pos_cur[1])
            #     orient = 0 if cos > 0 else 1
            #
            #     pos_pre = pos_cur
            #     pos_cur = pos_next

            next_img = cv2.imread(self.img_list[id])
            # 计算特征点
            kpt_2, des_2 = self.generate_feature(next_img)
            # 特征点匹配、筛选
            matched = self.keypoint_match(des_1, des_2)
            if len(matched) < 4: break
            # 还原特征点坐标
            src, dts = self.reset_kpt_coord(matched, kpt_1, kpt_2)

            # dts = dts + np.array([1000, 0])
            # if id != 1:
            #     src = src + np.array([1000, 0])

            # 计算映射矩阵
            H, _ = cv2.findHomography(src, dts, cv2.RANSAC)
            size = self.get_warp_shape(H, img)
            H[0, -1] += size[0]
            w_expand = size[1] + size[0]
            h_expand = size[2]
            # 变换
            img = cv2.warpPerspective(img, H, (w_expand, h_expand))

            cut = img[0:self.h, size[0]:self.w+size[0]]
            img[0:self.h, size[0]:size[0]+self.w], img1 = self.opt_seam(cut, next_img)

            # for next loop
            # kpt_1 = kpt_2
            # des_1 = des_2

            self.show(img)
        return img



if __name__ == '__main__':
    root = "G:\\data\\20210817002"
    img_list = os.listdir(root)
    imgs = [os.path.join(root, name) for name in img_list]

    st = myStitcher(imgs[6:8])
    res = st.start()
    st.show(res)
    # cv2.imwrite("./img.jpg", res)

