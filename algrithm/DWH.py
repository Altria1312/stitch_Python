from sklearn.cluster import KMeans
from Stitcher import myStitcher
import numpy as np
import time
import cv2
import os

class DWH(myStitcher):
    def start(self):
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

if __name__ == '__main__':
    root = "G:\\data\\20210817002"
    img_list = os.listdir(root)
    imgs = [os.path.join(root, name) for name in img_list]

    st = DWH(imgs[3:5])
    st.start()