from algrithm.FE import FE
import numpy as np
from math import sqrt
import cv2
import os

class APAP(FE):
    # ========================
    def start(self):
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

            H, inliers = cv2.findHomography(src, dts, cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold1)
            inliers = np.squeeze(inliers, axis=1).astype(np.bool)
            # 计算全局映射误差
            src_in = src[inliers]
            dts_in = dts[inliers]
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
            grids_cond, centers = self.assign_grid(src_in)
            feature_sets = self.assign_set(grids_cond, err_cond)
            # 计算权重
            w = self.cale_weights(centers, feature_sets, src_in)

            t1, src1 = self.centroid_normalize(src_in)
            t2, dts1 = self.centroid_normalize(dts_in)
            c1, src2 = self.std_normalize(src1)
            c2, dts2 = self.std_normalize(dts1)

            A = self.gen_A(src2, dts2)

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

                    h = self.gen_h(A, t1, t2, c1, c2, temp)

                    mask = np.zeros_like(img)
                    mask[row_idx[i]:row_idx[i+1], col_idx[j]:col_idx[j+1]] = 1
                    warp = img * mask

                    t = cv2.warpPerspective(warp, \
                                            h, (next_img.shape[1], 2*next_img.shape[0]))
                    # self.show(t)
                    test += t
                    # self.show(test.astype(np.uint8), t=200)

                    # W_star.append(h)
            # self.show(test.astype(np.uint8))
            test[:self.h, :self.w] = next_img
            cv2.imwrite("./img_apap.jpg", test.astype(np.uint8))
            pass

    def apap_cale_weights(self, centers, feature_sets, src):
        # 中心点到各特征点集的平均距离
        res = np.zeros((self.grid_rows, self.grid_cols, src.shape[0]))
        sigma = 8.5

        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                dists = np.linalg.norm(centers[i, j] - src, axis=1)
                res[i, j] = np.maximum(np.exp(-dists / sigma**2), 0.001)

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



if __name__ == '__main__':
    root = "G:\\data\\20210817002"
    img_list = os.listdir(root)
    imgs = [os.path.join(root, name) for name in img_list]

    st = APAP(imgs[3:5])
    st.start()