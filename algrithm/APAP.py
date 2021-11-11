from algrithm.FE import FE
from threading import Thread
from queue import Queue
import numpy as np
from math import sqrt
import cv2
import os

def asnyc_thread(f):
    def warp(*args, **kwargs):
        t = Thread(target=f, args=args, kwargs=kwargs)
        t.start()

    return warp

class APAP(FE):
    # ========================
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
        for id in range(1, n):
            # 获取位置信息
            if id > 1:
                pos_next = self.exif_parse(self.img_list[id])
                a = pos_cur - pos_pre
                b = pos_cur - pos_next
                cos = np.sum(a*b) / (np.linalg.norm(a)*np.linalg.norm(b))
                if cos > -0.95 and cos > 0:
                    orient = 1 - orient

                pos_pre = pos_cur
                pos_cur = pos_next

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
            src += size[0, ::2]
            # 计算映射矩阵

            H, inliers = cv2.findHomography(src, dts, cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold1)
            inliers = np.squeeze(inliers, axis=1).astype(np.bool)
            # 计算全局映射误差
            size = self.get_warp_shape(H, img, 1)
            src_in = src[inliers]
            dts_in = dts[inliers] + size[0, ::2]
            w_expand = size[0, 1] + size[0, 0]
            h_expand = size[0, 2] + size[0, 3]
            # # ===========================================================================
            # t1, src1 = self.centroid_normalize(src_in)
            # t2, dts1 = self.centroid_normalize(dts_in)
            # c1, src2 = self.std_normalize(src1)
            # c2, dts2 = self.std_normalize(dts1)
            #
            # A = self.gen_A(src2, dts2)
            # H = self.gen_h(A, t1, t2, c1, c2)
            #
            # img = cv2.warpPerspective(img, H, (next_img.shape[1], 2*next_img.shape[0]))
            # cut = img[:next_img.shape[0]]
            # mask = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
            # mask = (mask > 0).astype(np.uint8)
            # mask = cv2.GaussianBlur(mask, (5,5), 1)
            # mask = cv2.merge([mask, mask, mask])
            # img[:next_img.shape[0]] = mask * next_img + (1-mask) * cut
            #
            # self.show(img)
            # # ==============================================================================
            warp_err = self.glob_warp_errs(H, src_in, dts_in)
            err_cond = warp_err > self.ransac_threshold2
            # 特征点分区
            grids_cond, centers = self.assign_grid(src_in, img.shape[:-1])
            feature_sets = self.assign_set(grids_cond, err_cond)
            # 计算权重
            w = self.apap_cale_weights(centers, feature_sets, src_in)

            t1, src1 = self.centroid_normalize(src_in)
            t2, dts1 = self.centroid_normalize(dts_in)
            c1, src2 = self.std_normalize(src1)
            c2, dts2 = self.std_normalize(dts1)

            A = self.gen_A(src2, dts2)

            # #　画网格
            # for i in range(1,col_idx.shape[0]):
            #     cv2.line(img, (col_idx[i]-1, 0), (col_idx[i]-1, self.h-1), (0, 0, 255), 2)
            # for j in range(1,row_idx.shape[0]):
            #     cv2.line(img, (0, row_idx[j]-1), (self.w-1, row_idx[j]-1), (0, 0, 255), 2)
            temp = np.zeros((h_expand, w_expand, 3))
            col_idx = np.linspace(0, w_expand, self.grid_cols+1).astype(np.int32)
            row_idx = np.linspace(0, h_expand, self.grid_rows+1).astype(np.int32)
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    W = np.zeros(2*src_in.shape[0])
                    W[::2] = W[1::2] = w[i, j]
                    W = np.expand_dims(W, axis=1)

                    h = self.gen_h(A, t1, t2, c1, c2, W)

                    mask = np.zeros_like(img)
                    mask[row_idx[i]:row_idx[i+1], col_idx[j]:col_idx[j+1]] = 1
                    warp = img * mask

                    temp += cv2.warpPerspective(warp, h, (w_expand, h_expand))
            temp = temp.astype(np.uint8)
            cut = temp[size[0, 2]:self.h+size[0, 2], size[0, 0]:self.w+size[0, 0]]
            temp[size[0, 2]:self.h+size[0, 2], size[0, 0]:self.w+size[0, 0]] = self.opt_seam(cut, next_img, invert)
            img = temp.copy()

            # for next loop
            kpt_1 = kpt_2
            des_1 = des_2

            self.show(img)
        cv2.imwrite("./img_apap10.jpg", temp.astype(np.uint8))
        pass

    def apap_cale_weights(self, centers, feature_sets, src):
        # 中心点到各特征点集的平均距离
        res = np.zeros((self.grid_rows, self.grid_cols, src.shape[0]))
        sigma = self.h / 6

        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                dists = np.linalg.norm(centers[i, j] - src, axis=1)
                res[i, j] = np.maximum(np.exp(-(dists / sigma)**2), self.gamma)

        return res

    def centroid_normalize(self, pts):
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

        return t, res[:-1].T

    def std_normalize(self, pts):
        n = pts.shape[0]
        pts_ex = np.expand_dims(pts, axis=0)
        mean, std = cv2.meanStdDev(pts_ex)
        mean = np.squeeze(mean)
        # 无偏估计
        std *= sqrt(n / (n-1))
        # 非零
        std += std==0
        scale = sqrt(2) / np.squeeze(std)

        t = np.array([[scale[0], 0, -scale[0]*mean[0]],
                      [0, scale[1], -scale[1]*mean[1]],
                      [0, 0, 1]])
        pts_ex = np.column_stack([pts, np.ones((n, 1))])
        res = t @ pts_ex.T

        return t, res[:-1].T

    def gen_A(self, src, dst):
        src11 = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)
        A = np.zeros((2*src.shape[0], 9))
        A[::2, :3] = A[1::2, 3:6] = src11
        A[::2, -3:-1] = -src * dst[:, [0]]
        A[1::2, -3:-1] = -src * dst[:, [1]]
        A[::2, -1] = -dst[:, 0]
        A[1::2, -1] = -dst[:, 1]

        return A

    def gen_h(self, A, t1, t2, c1, c2, w=1):
        t = cv2.SVDecomp(A*w)[-1]
        h = t[-1, :].reshape((3, 3))

        h = np.linalg.inv(c2).dot(h).dot(c1)
        h = np.linalg.inv(t2).dot(h).dot(t1)
        h = h / h[2, 2]

        return h

    def warp_local(self, i, src_in, img, size, prama):
        A, t1, t2, c1, c2, w = prama

        col_idx = np.linspace(0, size[1], self.grid_cols+1).astype(np.int32)
        row_idx = np.linspace(0, size[0], self.grid_rows+1).astype(np.int32)
        test = np.zeros((size[0], size[1], 3), dtype=np.int32)
        for j in range(self.grid_cols):
            temp = np.zeros(2*src_in.shape[0])
            temp[::2] = temp[1::2] = w[i, j]
            temp = np.expand_dims(temp, axis=1)

            h = self.gen_h(A, t1, t2, c1, c2, temp)

            mask = np.zeros_like(img)
            mask[row_idx[i]:row_idx[i+1], col_idx[j]:col_idx[j+1]] = 1
            warp = img * mask

            t = cv2.warpPerspective(warp, \
                                    h, (size[1], size[0]))
            # self.show(t)
            test += t

        return test



if __name__ == '__main__':
    root = "G:\\data\\20210817002"
    img_list = os.listdir(root)
    imgs = [os.path.join(root, name) for name in img_list]

    # imgs = [r"G:\APAP-Image-Stitching-main\images\demo3\prague2.jpg",
    #         r"G:\APAP-Image-Stitching-main\images\demo3\prague1.jpg"]

    st = APAP(imgs[24:44])
    st.start()