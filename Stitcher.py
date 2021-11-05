import cv2
import numpy as np
import piexif
import time
import os
from sklearn.cluster import KMeans
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
        self.grid_cols = 10
        self.grid_rows = 10
        self.gamma = 0.0025


    def show(self, img, name="", t=0):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey(t)
        cv2.destroyAllWindows()

    def detect_compute(self, img):
        if cv2.__version__ == "4.5.1":
            extractor = cv2.SIFT_create(5000)
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

        return (x, y)

    def get_warp_shape(self, H, img, orient=1):
        # 变换后的区域
        pts = np.array([[0, img.shape[1], img.shape[1], 0],
                        [0, 0, img.shape[0], img.shape[0]],
                        [1, 1, 1, 1]])
        pts_H = H @ pts
        pts_H /= pts_H[-1, :]

        xy_min = np.round(np.min(pts_H, axis=1)[:-1]).astype(np.int32)
        xy_max = np.round(np.max(pts_H, axis=1)[:-1]).astype(np.int32)

        if orient == 1:
            left = 0 if xy_min[0] >= 0 else -xy_min[0]
            right = self.w if xy_max[0] <= self.w else xy_max[0]
            buttom = xy_max[1]
            return left, right, 0, buttom
        elif orient == 0:
            up = 0 if xy_min[1] >= 0 else -xy_min[1]
            buttom = self.h if xy_max[1] <= self.h else xy_max[1]
            right = xy_max[0]
            return 0, right, up, buttom


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
        # pos_pre = self.exif_parse(self.img_list[0])
        # pos_cur = self.exif_parse(self.img_list[1])

        n = len(self.img_list)
        for id in range(1, n):
            next_img = cv2.imread(self.img_list[id])
            # 获取位置信息
            # if id + 1 < n:
            #     pos_next = self.exif_parse(self.img_list[id+1])
            #     cos = (pos_cur[0] - pos_pre[0]) * (pos_next[0] - pos_cur[0]) + \
            #           (pos_cur[1] - pos_pre[1]) * (pos_next[1] - pos_cur[1])
            #     orient = 0 if cos > 0 else 1
            #
            #     pos_pre = pos_cur
            #     pos_cur = pos_next

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
            src_norm = self.normalize(src)
            dts_norm = self.normalize(dts)
            H, _ = cv2.findHomography(src_norm, dts_norm, cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold1)
            inliers = np.squeeze(_, axis=1).astype(np.bool)
            src = src_norm[inliers]
            dts = dts_norm[inliers]
            H, _ = cv2.findHomography(src, dts, cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold2)

        # img[:, :next_img.shape[1]] = mask * next_img + (1-mask) * cut
            # size = self.get_warp_shape(H, img, 0)
            # H[0, -1] += size[0]
            # H[1, -1] += size[2]
            # w_expand = size[1] + size[0]
            # h_expand = size[2] + size[3]
            # # 变换
            # img = cv2.warpPerspective(img, H, (w_expand, h_expand))
            #
            # cut = img[size[2]:self.h+size[2], size[0]:self.w+size[0]]
            # img[size[2]:self.h+size[2], size[0]:size[0]+self.w] = self.opt_seam(cut, next_img)
            #
            # # for next loop
            # kpt_1 = kpt_2
            # des_1 = des_2

            img = cv2.warpPerspective(img, H, (next_img.shape[1], 2*next_img.shape[0]))
            cut = img[:next_img.shape[0]]
            mask = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
            mask = (mask > 0).astype(np.uint8)
            mask = cv2.GaussianBlur(mask, (5,5), 1)
            mask = cv2.merge([mask, mask, mask])
            img[:next_img.shape[0]] = mask * next_img + (1-mask) * cut


        cv2.imwrite("./img.jpg", img)
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
    # ==================================
    def DHW(self):
        img = cv2.imread(self.img_list[0])
        self.h = img.shape[0]
        self.w = img.shape[1]
        kpt_1, des_1 = self.generate_feature(img)

        next_img = cv2.imread(self.img_list[1])
        # 计算特征点
        kpt_2, des_2 = self.generate_feature(next_img)
        # 特征点匹配、筛选
        matched = self.keypoint_match(des_1, des_2)
        # 还原特征点坐标
        src, dts = self.reset_kpt_coord(matched, kpt_1, kpt_2)

        x = np.mean(src, axis=0)
        init = np.array([[x[0], 0],
                        [x[0], img.shape[0]]])

        pred = KMeans(n_clusters=2, random_state=9, init=init).fit_predict(src)
        # plt.scatter(src[:, 0], src[:, 1], c=pred)
        # plt.show()
        pts_mask = pred > 0
        src_g = src[pts_mask]
        dts_g = dts[pts_mask]
        pts_mask = np.bitwise_not(pts_mask)
        src_d = src[pts_mask]
        dts_d = dts[pts_mask]

        H_g, _ = cv2.findHomography(src_g, dts_g, cv2.RANSAC)
        H_d, _ = cv2.findHomography(src_d, dts_d, cv2.RANSAC)
        t1 = time.time()
        d_g = self.get_homo_weight(img.shape, src_g)
        d_d = self.get_homo_weight(img.shape, src_d)
        t2 = time.time()
        print(t2-t1)
        w_g = d_g / (d_g + d_d)
        w_g = np.expand_dims(w_g, axis=2)
        w_d = 1 - w_g

        img_g = cv2.warpPerspective(img*w_g, H_g, (2*next_img.shape[1], next_img.shape[0]))
        img_d = cv2.warpPerspective(img*w_d, H_d, (2*next_img.shape[1], next_img.shape[0]))

        img = (img_g + img_d).astype(np.uint8)
        self.show(img)
        cut = img[:, :next_img.shape[1]]
        mask = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8)
        mask = cv2.GaussianBlur(mask, (5,5), 1)
        mask = cv2.merge([mask, mask, mask])
        img[:, :next_img.shape[1]] = mask * next_img + (1-mask) * cut

        cv2.imwrite("./img_dual.jpg", img)

        pass

    def get_homo_weight(self, shape, pts):
        h, w = shape[:-1]
        range_h = np.arange(h).reshape((-1,1))
        range_w = np.arange(w).reshape((1, -1))
        layer_x = np.tile(range_w, [h, 1]).reshape((h, w, 1))
        layer_y = np.tile(range_h, [1, w]).reshape((h, w, 1))
        coord = np.concatenate((layer_x, layer_y), axis=-1)

        dist = np.apply_along_axis(self.cal_dist, axis=1, arr=pts, coord=coord)
        dist_min = np.max(dist, axis=0)

        return dist_min

    def cal_dist(self, pt, coord):
        temp = coord - pt
        dist = 1 / np.linalg.norm(temp, ord=2, axis=-1)

        return dist
    # ==================================
    def FE(self):
        # 第一张
        orient = 0
        img = cv2.imread(self.img_list[0])
        self.h = img.shape[0]
        self.w = img.shape[1]

        kpt_1, des_1 = self.generate_feature(img)
        # 获取位置信息
        # pos_pre = self.exif_parse(self.img_list[0])
        # pos_cur = self.exif_parse(self.img_list[1])

        n = len(self.img_list)
        for id in range(1, n):
            next_img = cv2.imread(self.img_list[id])
            # 获取位置信息
            # if id + 1 < n:
            #     pos_next = self.exif_parse(self.img_list[id+1])
            #     cos = (pos_cur[0] - pos_pre[0]) * (pos_next[0] - pos_cur[0]) + \
            #           (pos_cur[1] - pos_pre[1]) * (pos_next[1] - pos_cur[1])
            #     orient = 0 if cos > 0 else 1
            #
            #     pos_pre = pos_cur
            #     pos_cur = pos_next

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
            src_norm = self.normalize(src)
            dts_norm = self.normalize(dts)
            H, inliers = cv2.findHomography(src_norm, dts_norm, cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold1)
            inliers = np.squeeze(inliers, axis=1).astype(np.bool)
            # 计算全局映射误差
            src_in = src_norm[inliers]
            dts_in = dts_norm[inliers]
            warp_err = self.glob_warp_errs(H, src_in, dts_in)
            err_cond = warp_err > self.ransac_threshold2
            # 特征点分区
            grids_cond, centers = self.assign_grid(src_in)
            feature_sets = self.assign_set(grids_cond, err_cond)
            # 计算权重
            w = self.cale_weights(centers, feature_sets, src_in)

            src1 = np.concatenate([src_in, np.ones((src_in.shape[0], 1))], axis=1)
            A = np.zeros((2*src_in.shape[0], 8))
            A[::2, :3] = A[1::2, 3:6] = src1
            A[::2, -2:] = -src_in * dts_in[:, [0]]
            A[1::2, -2:] = -src_in * dts_in[:, [1]]

            col_idx = np.linspace(0, self.w, self.grid_cols+1).astype(np.int32)
            row_idx = np.linspace(0, self.h, self.grid_rows+1).astype(np.int32)
            test = np.zeros((2*next_img.shape[0], next_img.shape[1], 3))
            y = np.zeros(2*src_in.shape[0])
            y[::2] = dts_in[:, 0]
            y[1::2] = dts_in[:, 1]
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    temp = np.zeros(2*src_in.shape[0])
                    temp[::2] = temp[1::2] = w[i, j]

                    temp = np.expand_dims(temp, axis=1)
                    h = np.linalg.lstsq(A*temp, y, rcond=None)[0]
                    h = np.append(h, 1).reshape((3,3))
                    t = cv2.warpPerspective(img[row_idx[i]:row_idx[i+1], col_idx[j]:col_idx[j+1]], \
                                            h, (next_img.shape[1], 2*next_img.shape[0]))
                    self.show(t)
                    # test += t

                    # W_star.append(h)
            self.show(test.astype(np.uint8))
            pass

    def glob_warp_errs(self, H, src, dts):
        # ======================
        ex = np.ones((src.shape[0], 1))
        temp = np.concatenate([src, ex], axis=1).T
        warp = H @ temp
        src = warp[:-1] / warp[-1]
        err = src.T - dts
        dist = np.sqrt(err[:, 0]**2 + err[:, 1]**2)

        return dist

    def assign_grid(self, src):
        space_col = self.w / self.grid_cols
        space_row = self.h / self.grid_rows

        # 计算所属行列
        res = np.zeros_like(src)
        res[:, 0] = src[:, 0] // space_col
        res[:, 1] = src[:, 1] // space_row
        # 计算每个区域中心点
        x = np.arange(start=space_col / 2, stop=self.w, step=space_col)
        y = np.arange(start=space_row / 2, stop=self.h, step=space_row)
        xx, yy = np.meshgrid(x, y)
        xx = np.expand_dims(xx, axis=2)
        yy = np.expand_dims(yy, axis=2)
        centers = np.concatenate([xx, yy], axis=2)

        return res.astype(np.int32), centers

    def assign_set(self, grid_cond, err_cond):
        res = np.zeros((self.grid_rows, self.grid_cols, err_cond.shape[0]))
        for i in range(err_cond.shape[0]):
            res[..., i] = 2 + err_cond[i]
            res[grid_cond[i, 1], grid_cond[i, 0], i] = 0 + err_cond[i]

        return res

    def cale_weights(self, centers, feature_sets, src):
        # 中心点到各特征点集的平均距离
        res = np.zeros((self.grid_rows, self.grid_cols, src.shape[0]))
        sigma = self.h

        mean_dists = np.zeros((self.grid_rows, self.grid_cols, 4))
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                dists = np.linalg.norm(centers[i, j] - src, axis=1)
                for k in range(4):
                    temp = feature_sets[i, j] == k
                    if np.any(temp):
                        mean_dists[i, j, k] = np.mean(dists[temp])

        d = np.sum(mean_dists, axis=-1)
        # mean_dists = np.where(mean_dists==0, self.w, mean_dists)

        temp = -(mean_dists / sigma) ** 2
        k = self.ransac_threshold2 / self.ransac_threshold1

        for t in range(4):
            idx = feature_sets == t
            if np.any(idx):
                if t % 2 == 0:
                    f = (d - mean_dists[..., t]) / d * np.exp(k * temp[..., t])
                else:
                    f = (d - mean_dists[..., t]) / d * np.exp(temp[..., t])
                f = np.tile(np.expand_dims(f, 2), (1,1,res.shape[-1]))
                np.putmask(res, idx, f)
                pass

        return res

    # ========================
    def APAP(self):
        # 第一张
        img = cv2.imread(self.img_list[0])
        self.h = img.shape[0]
        self.w = img.shape[1]

        kpt_1, des_1 = self.generate_feature(img)
        # 获取位置信息
        # pos_pre = self.exif_parse(self.img_list[0])
        # pos_cur = self.exif_parse(self.img_list[1])

        n = len(self.img_list)
        for id in range(1, n):
            next_img = cv2.imread(self.img_list[id])

            # 计算特征点
            kpt_2, des_2 = self.generate_feature(next_img)
            # 特征点匹配、筛选
            matched = self.keypoint_match(des_1, des_2)
            if len(matched) < 4: break
            # 还原特征点坐标
            src, dts = self.reset_kpt_coord(matched, kpt_1, kpt_2)

            # 计算映射矩阵
            src_norm = self.normalize(src)
            dts_norm = self.normalize(dts)
            H, inliers = cv2.findHomography(src, dts, cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold1)
            inliers = np.squeeze(inliers, axis=1).astype(np.bool)
            # 计算全局映射误差
            src_in = src[inliers]
            dts_in = dts[inliers]
            warp_err = self.glob_warp_errs(H, src_in, dts_in)
            err_cond = warp_err > self.ransac_threshold2
            # 特征点分区
            grids_cond, centers = self.assign_grid(src_in)
            feature_sets = self.assign_set(grids_cond, err_cond)
            # 计算权重
            w = self.apap_cale_weights(centers, feature_sets, src_in)

            src1 = np.concatenate([src_in, np.ones((src_in.shape[0], 1))], axis=1)
            A = np.zeros((2*src_in.shape[0], 8))
            A[::2, :3] = A[1::2, 3:6] = src1
            A[::2, -2:] = -src_in * dts_in[:, [0]]
            A[1::2, -2:] = -src_in * dts_in[:, [1]]

            col_idx = np.linspace(0, self.w, self.grid_cols+1).astype(np.int32)
            row_idx = np.linspace(0, self.h, self.grid_rows+1).astype(np.int32)
            test = np.zeros((2*next_img.shape[0], next_img.shape[1], 3))
            y = np.zeros(2*src_in.shape[0])
            y[::2] = dts_in[:, 0]
            y[1::2] = dts_in[:, 1]
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    temp = np.zeros(2*src_in.shape[0])
                    temp[::2] = temp[1::2] = w[i, j]

                    temp = np.expand_dims(temp, axis=1)
                    h = np.linalg.lstsq(A*temp, y, rcond=None)[0]
                    h = np.append(h, 1).reshape((3,3))
                    t = cv2.warpPerspective(img[row_idx[i]:row_idx[i+1], col_idx[i]:col_idx[i+1]], \
                                            h, (next_img.shape[1], 2*next_img.shape[0]))
                    # self.show(t)
                    test += t

                    # W_star.append(h)
            self.show(test.astype(np.uint8))
            pass

    def apap_cale_weights(self, centers, feature_sets, src):
        # 中心点到各特征点集的平均距离
        res = np.zeros((self.grid_rows, self.grid_cols, src.shape[0]))
        sigma = self.h / self.grid_rows / 3

        mean_dists = np.zeros((self.grid_rows, self.grid_cols, 4))
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                dists = np.linalg.norm(centers[i, j] - src, axis=1)
                for k in range(4):
                    temp = feature_sets[i, j] == k
                    if np.any(temp):
                        mean_dists[i, j, k] = np.mean(dists[temp])
                    else:
                        mean_dists[i, j, k] = 0

        d = np.sum(mean_dists, axis=-1)
        mean_dists = np.where(mean_dists==0, self.w, mean_dists)

        temp = -(mean_dists / sigma) ** 2
        k = self.ransac_threshold2 / self.ransac_threshold1

        for t in range(4):
            idx = feature_sets == t
            if np.any(temp):
                if t % 2 == 0:
                    f = (d - mean_dists[..., t]) / d * np.exp(k * temp[..., t])
                else:
                    f = (d - mean_dists[..., t]) / d * np.exp(temp[..., t])

                np.putmask(res, idx, f)

        return res

    def normalize(self, pts):
        pt_mean = np.mean(pts, axis=0)
        norm = np.linalg.norm(pts-pt_mean, axis=1)
        mean = np.mean(norm)
        scale = sqrt(2) / (mean + 1e-8)

        t = np.array([[scale, 0, -scale * pt_mean[0]],
                      [0, scale, -scale * pt_mean[1]],
                      [0, 0, 1]], dtype=np.float)
        ex = np.ones((pts.shape[0], 1))
        temp = np.column_stack([pts, ex])
        res = t @ temp.T

        return res[:-1].T




if __name__ == '__main__':
    root = "G:\\data\\20210817002"
    img_list = os.listdir(root)
    imgs = [os.path.join(root, name) for name in img_list]

    st = myStitcher(imgs[3:5])
    st.start()
    # cv2.imwrite("./img.jpg", res)

