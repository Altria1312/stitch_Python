import cv2
import numpy as np
import piexif
import time
import os
from math import sqrt
import matplotlib.pyplot as plt

class myStitcher:
    def __init__(self, img_list):
        self.img_list = img_list
        self.scale = 5
        self.match_threhold = 0.6
        self.matcher = cv2.FlannBasedMatcher()
        self.ransac_threshold1 = 3
        self.ransac_threshold2 = self.ransac_threshold1 / 2
        self.grid_cols = 1
        self.grid_rows = 4
        self.gamma = 0.0025


    def show(self, img, name="", t=0):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey(t)
        cv2.destroyAllWindows()

    def detect_compute(self, img):
        if cv2.__version__ == "4.5.1":
            extractor = cv2.SIFT_create()
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
            if ori[i, 0] == ori[i-1, 0]:
                continue
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

        return np.array([x, y])

    def get_warp_shape(self, H, img, orient=1):
        '''

        :param H:
        :param img:
        :param orient: 1表示竖直拼接， 0为横向拼接
        :return:
        '''
        # 变换后的区域
        pts = np.array([[0, img.shape[1], img.shape[1], 0],
                        [0, 0, img.shape[0], img.shape[0]],
                        [1, 1, 1, 1]])
        # h = cv2.invert(H)[1]
        pts_H = np.matmul(H, pts)
        pts_H /= pts_H[-1, :]

        xy_min = np.round(np.min(pts_H, axis=1)[:-1]).astype(np.int32)
        xy_max = np.round(np.max(pts_H, axis=1)[:-1]).astype(np.int32)

        left = 0 if xy_min[0] >= 0 else -xy_min[0]
        up = 0 if xy_min[1] >= 0 else -xy_min[1]
        if orient == 1:
            right = self.w if xy_max[0] <= self.w else xy_max[0]
            buttom = max(xy_max[1], self.h)
        elif orient == 0:
            right = max(xy_max[0], self.w)
            buttom = self.h if xy_max[1] <= self.h else xy_max[1]

        return np.array([[left, right, up, buttom]])

    def weighted_blend(self):
        pass

    def opt_seam(self, cut, img2, invert=1):
        src1 = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        src2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        E_color = cv2.absdiff(src2, src1)

        # sob_x = np.array([[-2, 0, 2],
        #                   [-1, 0, 1],
        #                   [-2, 0, 2]])
        # sob_y = np.array([[-2, -1, -2],
        #                   [0, 0, 0],
        #                   [2, 1, 2]])
        #
        # src1_x = cv2.filter2D(src1, cv2.CV_32F, sob_x)
        # src1_y = cv2.filter2D(src1, cv2.CV_32F, sob_y)
        # src2_x = cv2.filter2D(src2, cv2.CV_32F, sob_x)
        # src2_y = cv2.filter2D(src2, cv2.CV_32F, sob_y)
        #
        # E_geometry = np.sqrt((src1_x - src2_x)**2 + (src1_y - src2_y)**2)

        dx = cv2.Sobel(E_color, -1, 1, 0)
        dy = cv2.Sobel(E_color, -1, 1, 1)
        texture = cv2.magnitude(dx.astype(np.float32), dy.astype(np.float32))
        E_geometry = cv2.convertScaleAbs(texture)

        self.E = 0.1 * E_color + 0.9 * E_geometry

        # 确定初始位置
        t = int(self.w/2)
        path_row = [i for i in range(self.h) if src1[i, t] > 0]
        path_row = np.array(path_row).reshape(-1, 1)
        self.path_row = np.tile(path_row[1:-1], self.w)
        # self.min_val = np.min(path_row) + 1
        # self.max_val = np.max(path_row)

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
        mids_E = self.E[self.path_row[:, 0]]
        lefts_E = self.E[self.path_row[:, 0]-1]
        rights_E = self.E[self.path_row[:, 0]+1]
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

        mask = self.seam_blend(mask, opt_path, 50)
        if invert:
            mask = 1 - mask
        # self.mask = cv2.merge([mask, mask, mask])
        self.mask = np.concatenate([mask, mask, mask], axis=2)
        cut = self.mask * cut + (1-self.mask) * img2

        # temp = np.column_stack([tt, img2, (1-self.mask) * img2])
        # self.show(temp)

        return cut.astype(np.uint8)

    def seam_blend(self,mask, opt_path, bend):
        dist = np.arange(-bend, bend+1).reshape(-1, 1)
        dist = (bend - dist) / (2*bend)
        weight = 0.5 * (1 + np.cos(dist*np.pi))
        weight = np.tile(weight, (1, opt_path.shape[0]))

        temp = np.row_stack((mask, opt_path, weight))

        x = np.apply_along_axis(self.assign_weight, 0, temp, bend)

        return np.expand_dims(x, axis=2)

    def assign_weight(self, x, bend):
        up = int(max(0, x[self.h]-bend))
        up_dist = int(up - (x[self.h]-bend))
        down = int(min(self.h, x[self.h]+bend))
        down_dist = int(x[self.h]+bend - down)

        x[up:down+1] = x[self.h+1+up_dist:x.shape[0]-down_dist]

        return x[:self.h]

    def exposure_adj(self, img1, img2, src, dts):
        # 曝光差异
        ori = np.round(src).astype(np.int32)
        target = np.round(dts).astype(np.int32)

        i1 = cv2.GaussianBlur(img1, (5,5), 1)
        i2 = cv2.GaussianBlur(img2, (5,5), 1)
        bgr1 = i1[ori[:,1], ori[:,0]]
        bgr2 = i2[target[:,1], target[:,0]]

        res = np.zeros((3, 2))
        for i in range(3):
            A = np.vstack([bgr2[:, i], np.ones(bgr2.shape[0])]).T
            res[i,0], res[i,1] = np.linalg.lstsq(A, bgr1[:, i], rcond=None)[0]
        temp = img2.copy()

        img2 = img2 * res[:, 0] + res[:, 1]
        img2 = img2.astype(np.uint8)

        cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
        cv2.imshow("img1", img1)
        cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
        cv2.imshow("img2", img2)
        cv2.namedWindow("ori", cv2.WINDOW_NORMAL)
        cv2.imshow("ori", temp)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img1, img2

    def start(self):
        # 第一张
        orient = 1
        img = cv2.imread(self.img_list[0])
        self.h = img.shape[0]
        self.w = img.shape[1]
        kpt_1, des_1 = self.generate_feature(img)
        # 获取位置信息
        pos_pre = self.exif_parse(self.img_list[0])
        pos_cur = self.exif_parse(self.img_list[1])

        n = len(self.img_list)
        size = np.zeros((1, 4))
        is_coner = False
        for id in range(1, n):
            # 获取位置信息
            if id > 1:
                pos_next = self.exif_parse(self.img_list[id])
                a = pos_cur - pos_pre
                b = pos_cur - pos_next
                cos = np.sum(a*b) / (np.linalg.norm(a)*np.linalg.norm(b))
                if cos > -0.95 and cos > 0:
                    orient = 1 - orient
                    is_coner = not is_coner

                pos_pre = pos_cur
                pos_cur = pos_next

            if is_coner:
                is_coner = not is_coner
                continue

            next_img = cv2.imread(self.img_list[id])
            if orient == 0:
                next_img = cv2.rotate(next_img, cv2.ROTATE_180)
                invert = True
            else:
                invert = False


            # 计算特征点
            kpt_2, des_2 = self.generate_feature(next_img)
            # 特征点匹配、筛选
            matched = self.keypoint_match(des_1, des_2)
            if len(matched) < 4: break
            # 还原特征点坐标
            src, dts = self.reset_kpt_coord(matched, kpt_1, kpt_2)

            # 计算映射矩阵
            H, _ = cv2.findHomography(src, dts, cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold1)
            inliers = np.squeeze(_, axis=1).astype(np.bool)
            src = src[inliers] + size[0, ::2]
            dts = dts[inliers]

            # next_img, img = self.exposure_adj(next_img, img, dts, src)

            H, _ = cv2.findHomography(src, dts, cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold2)
            size = self.get_warp_shape(H, img, 1)
            # H[0, -1] += size[0]
            # H[1, -1] += size[2]
            w_expand = size[0, 1] + size[0, 0]
            h_expand = size[0, 2] + size[0, 3]
            dts += size[0, ::2]
            H, _ = cv2.findHomography(src, dts, cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold2)
            # 变换
            img = cv2.warpPerspective(img, H, (w_expand, h_expand))
            # img[size[0, 2]:self.h+size[0, 2], size[0, 0]:size[0, 0]+self.w] = next_img

            cut = img[size[0, 2]:self.h+size[0, 2], size[0, 0]:self.w+size[0, 0]]
            img[size[0, 2]:self.h+size[0, 2], size[0, 0]:size[0, 0]+self.w] = self.opt_seam(cut, next_img, invert)

            # img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]+next_img.shape[0]))
            # img[:next_img.shape[0], :next_img.shape[1]] = next_img
            # cut = img[:next_img.shape[0]]
            # mask = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
            # mask = (mask > 0).astype(np.uint8)
            # mask = cv2.GaussianBlur(mask, (5,5), 1)
            # mask = cv2.merge([mask, mask, mask])
            # img[:next_img.shape[0]] = mask * next_img + (1-mask) * cut

            self.show(img)
            # for next loop
            kpt_1 = kpt_2
            des_1 = des_2


        cv2.imwrite("./results/img12.jpg", img)
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

    # imgs = [r"G:\APAP-Image-Stitching-main\images\demo3\prague1.jpg",
    #         r"G:\APAP-Image-Stitching-main\images\demo3\prague2.jpg"]


    st = myStitcher(imgs[24:26])
    st.start()
    # cv2.imwrite("./img.jpg", res)

