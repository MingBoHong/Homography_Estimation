import torch
import torch.nn as nn


class DLT(nn.Module):
    def __init__(self, batch_size, nums_pt=4):
        super(DLT, self).__init__()
        self.batch_size = batch_size
        self.nums_pt = nums_pt

    def forward(self, src_pt, dst_pt):
        if not isinstance(dst_pt, type(src_pt)):
            dst_pt = src_pt.new_tensor(dst_pt)

        S = src_pt.new_zeros((self.batch_size, self.nums_pt * 2, 8))
        D = src_pt.new_zeros((self.batch_size, self.nums_pt * 2, 1))

        for i in range(self.nums_pt):
            S[:, 2 * i] = S.new_tensor(torch.stack(
                [src_pt[:, i, 0], src_pt[:, i, 1], S.new_ones(self.batch_size),
                 S.new_zeros(self.batch_size), S.new_zeros(self.batch_size),
                 S.new_zeros(self.batch_size),
                 -src_pt[:, i, 0] * dst_pt[:, i, 0],
                 -src_pt[:, i, 1] * dst_pt[:, i, 0]], 1))
            S[:, 2 * i + 1] = S.new_tensor(torch.stack(
                [S.new_zeros(self.batch_size), S.new_zeros(self.batch_size),
                 S.new_zeros(self.batch_size), src_pt[:, i, 0], src_pt[:, i, 1],
                 S.new_ones(self.batch_size),
                 -src_pt[:, i, 0] * dst_pt[:, i, 1],
                 -src_pt[:, i, 1] * dst_pt[:, i, 1]], 1))

            D[:, 2 * i] = D.new_tensor(torch.stack([dst_pt[:, i, 0]], 1))
            D[:, 2 * i + 1] = D.new_tensor(torch.stack([dst_pt[:, i, 1]], 1))

        Warp_Matix = torch.bmm(torch.inverse(S), D).squeeze(-1)
        Warp_Matix = torch.cat((Warp_Matix, S.new_ones((self.batch_size, 1))), 1)
        return Warp_Matix.view(self.batch_size, 3, 3)


# if __name__ == '__main__':
#     dlt = DLT(batch_size=2)
#     src_pt = torch.Tensor([[[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0], [0, 1], [1, 0], [1, 1]]])
#     dst_pt = torch.Tensor([[[1, 1], [1, 3], [0.5, 4], [3, 7]], [[3, 3], [4, 6], [3, 2], [6, 6]]])
#
#     warp_matrix = dlt(src_pt, dst_pt)


