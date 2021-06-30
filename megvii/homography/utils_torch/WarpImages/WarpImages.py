import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from megvii.homography.utils_torch import DLT


class WarpImages(nn.Module):
    def __init__(self, batch_size, ImgShape):
        super(WarpImages, self).__init__()
        self.batch_size = batch_size
        self.H, self.W, self.C = ImgShape

    def bilinear_interpote(self, batch_img, batch_x, batch_y):
        batch_x = torch.clamp(batch_x, min=0, max=self.W - 1)
        batch_y = torch.clamp(batch_y, min=0, max=self.H - 1)

        # select four points around the interpolated point
        batch_x0 = torch.floor(batch_x).long()
        batch_x1 = batch_x0 + 1
        batch_y0 = torch.floor(batch_y).long()
        batch_y1 = batch_y0 + 1

        batch_x0 = torch.clamp(batch_x0, min=0, max=self.W - 1)
        batch_x1 = torch.clamp(batch_x1, min=0, max=self.W - 1)
        batch_y0 = torch.clamp(batch_y0, min=0, max=self.H - 1)
        batch_y1 = torch.clamp(batch_y1, min=0, max=self.H - 1)

        Ia = batch_y0 * self.W + batch_x0
        Ib = batch_y1 * self.W + batch_x0
        Ic = batch_y0 * self.W + batch_x1
        Id = batch_y1 * self.W + batch_x1

        batch_pa = torch.gather(batch_img, dim=1, index=Ia.unsqueeze(-1).repeat(1, 1, 3))
        batch_pb = torch.gather(batch_img, dim=1, index=Ib.unsqueeze(-1).repeat(1, 1, 3))
        batch_pc = torch.gather(batch_img, dim=1, index=Ic.unsqueeze(-1).repeat(1, 1, 3))
        batch_pd = torch.gather(batch_img, dim=1, index=Id.unsqueeze(-1).repeat(1, 1, 3))

        # computing the weight of the four points
        wa = ((batch_x1 - batch_x) * (batch_y1 - batch_y)).reshape(self.batch_size, -1, 1)
        wb = ((batch_x1 - batch_x) * (batch_y - batch_y0)).reshape(self.batch_size, -1, 1)
        wc = ((batch_x - batch_x0) * (batch_y1 - batch_y)).reshape(self.batch_size, -1, 1)
        wd = ((batch_x - batch_x0) * (batch_y - batch_y0)).reshape(self.batch_size, -1, 1)

        return wa * batch_pa + wb * batch_pb + wc * batch_pc + wd * batch_pd

    def forward(self, batch_img, Warp_Matrix):
        if not isinstance(batch_img, type(Warp_Matrix)):
            batch_img = Warp_Matrix.new_Tensor(batch_img)

        new_batch_img = batch_img.new_zeros(batch_img.shape).view(self.batch_size, -1, 3)
        # transformation xy coordinates
        new_batch_xy = new_batch_img.clone()[:, :, :2]

        batch_x = torch.arange(self.W).repeat(self.H).view(1, -1, 1).repeat(self.batch_size, 1, 1)
        batch_y = torch.repeat_interleave(torch.arange(self.H), repeats=self.W).view(1, -1, 1).repeat(self.batch_size,
                                                                                                      1, 1)
        goal = new_batch_xy.new_tensor(torch.cat([batch_x, batch_y, torch.ones_like(batch_x)], -1)).permute((0, 2, 1))

        # Transform the matrix 'goal'
        img_pt = torch.bmm(torch.inverse(Warp_Matrix), goal).permute(0, 2, 1)

        # Normalizing the transformation coordinates
        new_batch_xy[:, :, 0] = img_pt[:, :, 0] / img_pt[:, :, 2]
        new_batch_xy[:, :, 1] = img_pt[:, :, 1] / img_pt[:, :, 2]

        new_batch_img = self.bilinear_interpote(batch_img.permute(0, 2, 3, 1).view(self.batch_size, -1, 3),
                                                new_batch_xy[:, :, 0],
                                                new_batch_xy[:, :, 1])

        return new_batch_img


if __name__ == '__main__':
    dlt = DLT(batch_size=1)
    src_pt = torch.Tensor([[[284, 241], [287, 747], [547, 285], [593, 649]]])
    dst_pt = torch.Tensor([[[122, 310], [107, 742], [378, 235], [421, 705]]])

    warp_matrix = dlt(src_pt, dst_pt)

    warp_matrix_inv = torch.inverse(warp_matrix)
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    img1 = cv2.imread(r'/home/hongmingbo/PycharmProjects/internship/megvii/img/left.jpg')
    img2 = cv2.imread(r'/home/hongmingbo/PycharmProjects/internship/megvii/img/right.jpg')

    batch_img = torch.stack([transform(img1), transform(img2)])
    H = torch.cat([warp_matrix, warp_matrix_inv])
    # H = torch.Tensor([[[0.5, -0.5, 0], [0.5, 0.5, 0], [0, 0, 1]], [[1, 0, 20], [0, 1, 30], [0, 0, 1]]])
    warp_op = WarpImages(batch_size=2, ImgShape=img1.shape)
    warp_img = warp_op(batch_img, H)
    # warp_img1 = warp_img.numpy().reshape(2, 800, 600, 3)
    # cv2.imshow("img1", warp_img1[0])
    # cv2.imshow("img2", warp_img1[1])
    # cv2.waitKey(0)
