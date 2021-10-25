import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from math import exp, sin, pi
import threading

class Stitch:
    def __init__(self, imgs, isList=False):
        self.mask = np.array([0])
        if isList:
            self.img_list = imgs
        else:
            self.img1 = imgs[0]
            self.img2 = imgs[1]

    def detect_compute(self, img):
        # extractor = cv2.xfeatures2d.SURF_create()
        # extractor = cv2.xfeatures2d_SIFT.create()
        extractor = cv2.SIFT_create(1000)
        # extractor = cv2.ORB_create(500)
        return extractor.detectAndCompute(img, None)

    def show(self, img, name="", t=0):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey(t)
        cv2.destroyAllWindows()

    def match(self):
        scale = 5
        h = int(self.img1.shape[0] / scale)
        w = int(self.img1.shape[1] / scale)
        # 缩放
        src1 = cv2.resize(self.img1, (w, h), interpolation=cv2.INTER_NEAREST)
        src2 = cv2.resize(self.img2, (w, h), interpolation=cv2.INTER_NEAREST)
        # src1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
        # src2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

        # 特征描述
        kpt1, des1 = self.detect_compute(src1)
        kpt2, des2 = self.detect_compute(src2)

        # 匹配
        matcher = cv2.FlannBasedMatcher()

        # matches = matcher.match(des1, des2)
        # # 筛选
        # min_dis = 100
        # max_dis = 0
        # for i in range(len(matches)):
        #     dis = matches[i].distance
        #     if dis < min_dis:
        #         min_dis = dis
        #
        #     if dis > max_dis:
        #         max_dis = dis
        #
        # matched = []
        # threhold = 2 * min_dis
        # for i in range(len(matches)):
        #     if matches[i].distance < threhold:
        #         matched.append(matches[i])

        # knn
        matches = matcher.knnMatch(des1, des2, k=2)
        threhold = 0.4
        matched = []
        for m, n in matches:
            if m.distance < threhold * n.distance:
                matched.append(m)

        # t = cv2.drawMatches(src1, kpt1, src2, kpt2, matched, None)
        # self.show(t)

        # scalex = float(self.img1.shape[1] / src1.shape[1])
        # scaley = float(self.img1.shape[0] / src1.shape[0])
        ori = np.zeros((len(matched), 2))
        target = np.zeros_like(ori)
        for i in range(len(matched)):
            ori[i, 0] = kpt1[matched[i].queryIdx].pt[0] * scale
            ori[i, 1] = kpt1[matched[i].queryIdx].pt[1] * scale
            # ori.append((x, y))

            target[i, 0] = kpt2[matched[i].trainIdx].pt[0] * scale
            target[i, 1] = kpt2[matched[i].trainIdx].pt[1] * scale
            # target.append((x, y))

        # ori = np.array(ori)
        # target = np.array(target)

        self.H, _ = cv2.findHomography(ori, target, cv2.RHO)

        # 重合边界
        pts = np.array([[0, self.img1.shape[1], self.img1.shape[1], 0],
                        [0, 0, self.img1.shape[0], self.img1.shape[0]],
                        [1, 1, 1, 1]])
        pts_H = self.H @ pts
        pts_H /= pts_H[-1, :]

        xy_min = np.round(np.min(pts_H, axis=1)[:-1])
        xy_max = np.round(np.max(pts_H, axis=1)[:-1])
        mins = np.maximum(xy_min, [0, 0]).astype(np.int32)
        maxs = np.minimum(xy_max, [self.img1.shape[1], self.img1.shape[0]]).astype(np.int32)
        self.tl_br = (mins, maxs)
        # # 曝光差异
        # ori = np.round(ori).astype(np.int32)
        # target = np.round(target).astype(np.int32)
        #
        # i1 = cv2.GaussianBlur(self.img1, (5,5), 1)
        # i2 = cv2.GaussianBlur(self.img2, (5,5), 1)
        # bgr1 = i1[ori[:,1], ori[:,0]]
        # bgr2 = i2[target[:,1], target[:,0]]
        #
        # res = np.zeros((3, 2))
        # for i in range(3):
        #     A = np.vstack([bgr2[:, i], np.ones(bgr2.shape[0])]).T
        #     res[i,0], res[i,1] = np.linalg.lstsq(A, bgr1[:, i], rcond=None)[0]
        #
        # self.img2 = self.img2 * res[:, 0] + res[:, 1]
        # self.img2 = self.img2.astype(np.uint8)


    def stitch(self):
        t1 = time.time()
        self.match()
        # self.show(self.img2)
        t2 = time.time()
        # img = cv2.copyMakeBorder(img2, 0, img2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self.img = cv2.warpPerspective(self.img1, self.H, (self.img2.shape[1], 2 * self.img2.shape[0]))

        # self.show(img)
        t3 = time.time()
        # #============================================================
        # # 权重向两边递减
        # cut = self.img[:self.img2.shape[0], :self.img2.shape[1]]
        # mask = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        #
        # # # 像素值判断
        # mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        # mask = cv2.GaussianBlur(mask, (5,5), 1)
        # t4 = time.time()
        #
        # # mask = cv2.copyMakeBorder(mask, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=0)
        # mask = cv2.distanceTransform(mask, cv2.DIST_L1, 3)
        #
        # t5 = time.time()
        #
        # cv2.normalize(mask, mask, alpha=0, beta=1,
        #               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #
        # # s = mask.astype(np.uint8)
        # # self.show(s)
        # # mask = np.where(mask > 0.7, mask, 0)
        # t6 = time.time()
        # # mask = mask[:-1]
        # mask = cv2.merge([mask, mask, mask])
        # t7 = time.time()
        # self.img[:self.img2.shape[0], :self.img2.shape[1]] = mask * cut + (1-mask) * self.img2
        # #============================================================

        # # Laplacian金字塔图像融合
        # self.blend()

        # 最佳缝合线
        self.baseline()

        t8 = time.time()
        # print("matching:%f"%(t2-t1))
        # print("warping:%f"%(t3-t2))
        # # print("fushing:%f cond:%f dis:%f norm:%f expand:%f stitch:%f" % (t8-t3, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7))
        # print("fushing:%f" % (t8-t3))
        print("total:%f"%(t8-t1))
        # self.show(self.img)
        #
        cv2.imwrite("./imgg.jpg", self.img)

    def gauss(self, img, n):
        # 构建高斯金字塔
        Gauss_pyr = [img.astype(np.float32)]
        for i in range(n-1):
            temp = cv2.pyrDown(Gauss_pyr[-1])
            Gauss_pyr.append(temp)

        return Gauss_pyr

    def laplacian(self, img, n=3):
        # 构建高斯金字塔
        Gauss_pyr = self.gauss(img, n)

        # 构建拉普拉斯金字塔
        Laplacian_Pyr = []
        for i in range(n-1):
            s = Gauss_pyr[i].shape[:2]
            expand = cv2.pyrUp(Gauss_pyr[i+1])
            temp = Gauss_pyr[i] - expand
            Laplacian_Pyr.append(temp)

        return Laplacian_Pyr, Gauss_pyr

    def blend(self):
        cut = self.img[:self.img2.shape[0], :self.img2.shape[1]]
        mask = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)

        # # 像素值判断
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.GaussianBlur(mask, (5,5), 1)

        # mask = cv2.copyMakeBorder(mask, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=0)
        # mask = cv2.distanceTransform(mask, cv2.DIST_L1, 5)
        # cv2.normalize(mask, mask, alpha=0, beta=1,
        #                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # mask = mask[:-1]
        mask = cv2.merge([mask, mask, mask])
        # mask = mask.astype(np.float32)
        # mask *= 0.5

        n = 3
        mask_pyr = self.gauss(mask, n)
        img1_pyr, img1_pyr_g = self.laplacian(self.img[:self.img2.shape[0], :self.img2.shape[1]], n)
        img2_pyr, img2_pyr_g = self.laplacian(self.img2, n)

        start = img1_pyr_g[-1] * mask_pyr[-1] + img2_pyr_g[-1] * (1 - mask_pyr[-1])
        for i in range(n-2, -1, -1):
            temp = img1_pyr[i] * mask_pyr[i] + img2_pyr[i] * (1 - mask_pyr[i])
            start = cv2.pyrUp(start) + temp

        self.img[:self.img2.shape[0], :self.img2.shape[1]] = start.astype(np.uint8)

    def baseline_search(self, beg, end, w):
        end = min(end, w)
        for i in range(1, w):
            mids = self.path_row[beg:end, i-1]
            lefts = np.maximum(mids-1, self.min_val)
            rights = np.minimum(mids+1, self.max_val)
            temp = np.vstack([lefts, mids, rights])
            rg = np.arange(temp.shape[1])

            mid_E = self.E[mids, i]
            left_E = self.E[lefts, i]
            right_E = self.E[rights, i]

            temp_E = np.vstack([left_E, mid_E, right_E])
            idx = np.argmin(temp_E, axis=0)
            temp_E = temp_E[idx, rg]

            self.path_energy[beg:end] = self.path_energy[beg:end] + temp_E
            idx = temp[idx, rg]#.reshape((-1, 1))
            self.path_row[beg:end, i] = idx

    def mask_filter(self, w):
        for i in range(self.min_val, self.max_val+1):
            for j in range(w):
                if i < opt_path[j] and self.mask[i, j] > 0:
                    self.mask[i, j] = 0

    def baseline(self):
        cut = self.img[:self.img2.shape[0], :self.img2.shape[1]]
        src1 = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        src2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        # mask = src1 > 0
        # mask = mask.astype(np.uint8)
        # self.mask = cv2.GaussianBlur(mask, (5,5), 1)

        # self.mask = cv2.rectangle(np.zeros_like(src2), tuple(self.tl_br[0]), tuple(self.tl_br[1]), 255, -1)

        # cut[:, :, 1] = 0.5 * cut[:, :, 1] + 0.5 * self.mask * 255
        # self.show(cut)

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
        h, w = src1.shape
        t = int(w/2)
        path_row = [i for i in range(h) if src1[i, t] > 0]
        path_row = np.array(path_row).reshape(-1, 1)
        self.path_row = np.tile(path_row[1:-1], w)
        self.min_val = np.min(path_row) + 1
        self.max_val = np.max(path_row)

        path_row = np.tile(path_row, w)
        self.path_row = path_row[1:-1]
        # 初始能量
        self.path_energy = np.squeeze(self.E[self.path_row[:,0], 0])

        # # 生长(多线程）
        # num_thread = 1
        # width = h // 1
        # beg = 0
        # end = 0
        # worker_thread = []
        #
        # for i in range(num_thread):
        #     beg = end
        #     end = beg + width
        #     t = threading.Thread(target=self.baseline_search, args=(beg, end, w))
        #     worker_thread.append(t)
        #     t.start()
        #
        # for t in worker_thread:
        #     t.join()
        #     print(t.name)

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
        rg_y = np.tile(np.arange(path_row.shape[0]), [w,1]).T
        # temp_E = temp_E[rg_y, rg_x-1, idx]
        idx -= 1
        rg_y[1:-1, 1:] += idx[:, 1:]
        rg_y[0] += 1
        rg_y[-1] -= 1

        path_row[1:-1, 1:] += idx[:, 1:]
        prev = rg_y[1:-1, 0]
        for i in range(1, w):
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

        rg = np.tile(np.arange(h), [w, 1]).T
        mask = np.greater(rg, opt_path).astype(np.uint8)

        # rg = np.arange(h)
        # for j in range(w):
        #     col = rg > opt_path[j]
        #     self.mask[:, j] = self.mask[:, j] * col

        # self.show(self.mask * 255)
        # 权重向两边递减
        blend = 30
        mask = mask.astype(np.float32)
        for i in range(w):
            for j in range(-blend, blend):
                if opt_path[i] < -j or opt_path[i] + j >= h:
                    continue
                hh = opt_path[i] + j
                mask[hh, i] = (blend + j) / (2 * blend)
        # self.mask = np.zeros_like(mask)
        # cv2.normalize(mask, self.mask, 0, 255, cv2.NORM_MINMAX)
        # self.mask = self.mask.astype(np.uint8)
        # self.show(self.mask)
        # # 三角函数递减
        # T = 100
        # k = pi / T
        # for i, r in enumerate(opt_path):
        #     for j in range(-T, T):
        #         mask[r+j, i] = sin(k * (r+j)) / 2 + 0.5

        # mask = mask[:-1]

        self.mask = cv2.merge([mask, mask, mask])
        cut = self.mask * cut + (1-self.mask) * self.img2
        self.img[:self.img2.shape[0], :self.img2.shape[1]] = cut

    def mul_sitich(self):
        if hasattr(self, "img_list"):
            scale = 5
            # 缩放
            img1 = cv2.imread(self.img_list[0])
            h = int(img1.shape[0] / scale)
            w = int(img1.shape[1] / scale)
            src1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_NEAREST)
            kpt1, des1 = self.detect_compute(src1)

            n = len(self.img_list)
            matcher = cv2.FlannBasedMatcher()
            threhold = 0.4
            for i in range(1, n):
                # 计算特征点
                img2 = cv2.imread(self.img_list[i])
                src2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_NEAREST)
                kpt2, des2 = self.detect_compute(src2)
                # 匹配
                matches = matcher.knnMatch(des1, des2, k=2)
                matched = []
                for m, n in matches:
                    if m.distance < threhold * n.distance:
                        matched.append(m)

                ori = np.zeros((len(matched), 2))
                target = np.zeros_like(ori)
                for i in range(len(matched)):
                    ori[i, 0] = kpt1[matched[i].queryIdx].pt[0] * scale
                    ori[i, 1] = kpt1[matched[i].queryIdx].pt[1] * scale
                    # ori.append((x, y))

                    target[i, 0] = kpt2[matched[i].trainIdx].pt[0] * scale
                    target[i, 1] = kpt2[matched[i].trainIdx].pt[1] * scale
                    # target.append((x, y))

                H, _ = cv2.findHomography(ori, target, cv2.RANSAC)

                # img = cv2.copyMakeBorder(img2, 0, img2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img1.shape[0] + img2.shape[0]))
                # self.show(img)

                # 权重向两边递减
                cut = img1[:img2.shape[0], :img2.shape[1]]
                mask = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)

                # # 像素值判断
                mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
                mask = cv2.GaussianBlur(mask, (5,5), 1)

                # mask = cv2.copyMakeBorder(mask, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=0)
                mask = cv2.distanceTransform(mask, cv2.DIST_L1, 3)

                cv2.normalize(mask, mask, alpha=0, beta=1,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # s = mask.astype(np.uint8)
                # self.show(s)
                # mask = np.where(mask > 0.7, mask, 0)
                # mask = mask[:-1]
                mask = cv2.merge([mask, mask, mask])
                img1[:img2.shape[0], :img2.shape[1]] = mask * cut + (1-mask) * img2

                kpt1 = kpt2
                des1 = des2
                self.show(img1)

            self.show(img1)
            cv2.imwrite("./imgg.jpg", img1)
        else:
            print("None")


if __name__ == "__main__":
    path1 = "../../data/20210817002/20210817002_0006.JPG"
    path2 = "../../data/20210817002/20210817002_0007.JPG"
    img_1 = cv2.imread(path1)
    img_2 = cv2.imread(path2)

    imgs = ["../../data/20210817002/20210817002_0005.JPG",
            "../../data/20210817002/20210817002_0006.JPG",
            "../../data/20210817002/20210817002_0007.JPG",
            "../../data/20210817002/20210817002_0008.JPG"]
    s = Stitch([img_1, img_2])
    s.stitch()

    pass
