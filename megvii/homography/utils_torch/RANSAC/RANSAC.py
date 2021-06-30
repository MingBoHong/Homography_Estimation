import numpy as np


class RANSAC:
    def __init__(self, src_pt, dst_pt, iter=1000, thr=5, seed=66):
        if len(src_pt.shape) == 3:
            src_pt = np.squeeze(src_pt, src_pt.shape.index(1))
        if len(dst_pt.shape) == 3:
            dst_pt = np.squeeze(dst_pt, dst_pt.shape.index(1))

        self.iter_nums = iter
        self.thr = thr
        self.bs_inliner = float('inf')
        self.bs_homo = np.ones((3, 3))
        self.src_pt = src_pt
        self.dst_pt = dst_pt
        self.nums_pt = len(self.src_pt)
        self.order_pt = np.arange(self.nums_pt)
        self.seed = np.random.seed(seed)

    @staticmethod
    def WarpMatrix_4pt(src, dst):

        if len(src.shape) == 3:
            src = np.squeeze(src, src.shape.index(1))
        if len(dst.shape) == 3:
            dst = np.squeeze(dst, dst.shape.index(1))

        A = np.zeros((4 * 2, 8))
        B = np.zeros((4 * 2, 1))

        for i in range(4):
            A[2 * i] = [src[i, 0], src[i, 1], 1, 0, 0, 0, -src[i, 0] * dst[i, 0], -src[i, 1] * dst[i, 0]]
            A[2 * i + 1] = [0, 0, 0, src[i, 0], src[i, 1], 1, -src[i, 0] * dst[i, 1], -src[i, 1] * dst[i, 1]]
            B[2 * i] = dst[i, 0]
            B[2 * i + 1] = dst[i, 1]
        A = np.matrix(A)
        warpMatrix = A.I * B
        # warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0)
        # warp_matrix = warp_matrix.reshape((3, 3))
        warpMatrix = np.array(warpMatrix).T[0]
        warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
        warpMatrix = warpMatrix.reshape((3, 3))
        return warpMatrix

    def CalErr(self, res_idx, H):
        res_pt = self.src_pt[res_idx]
        res_pt = np.matrix(np.hstack((res_pt, np.ones((len(res_pt), 1)))))
        trans_pt = res_pt * H
        trans_pt[:, 0] = trans_pt[:, 0] / trans_pt[:, 2]
        trans_pt[:, 1] = trans_pt[:, 1] / trans_pt[:, 2]
        trans_pt = np.array(trans_pt)[:, :2]
        res_matrix = trans_pt - self.dst_pt[res_idx]
        error = np.linalg.norm(res_matrix)
        return error

    def __call__(self):
        for i in range(self.iter_nums):
            tmp_idx = np.random.choice(self.order_pt, 4, replace=False)
            tmp_src = self.src_pt[tmp_idx]
            tmp_dst = self.dst_pt[tmp_idx]

            try:
                H = RANSAC.WarpMatrix_4pt(tmp_src, tmp_dst)
                tmp_rst = self.CalErr(np.delete(self.order_pt, tmp_idx), H)
                if tmp_rst < self.bs_inliner:
                    self.bs_inliner = tmp_rst
                    self.bs_homo = H
            except:
                print("matrix is irreversible")
        return self.bs_homo
