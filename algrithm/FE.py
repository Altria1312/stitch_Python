from Stitcher import myStitcher
import numpy as np
import cv2
import os

class FE(myStitcher):
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
        sigma = self.h / self.grid_rows

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


if __name__ == '__main__':
    root = "G:\\data\\20210817002"
    img_list = os.listdir(root)
    imgs = [os.path.join(root, name) for name in img_list]

    st = FE(imgs[3:5])
    st.start()