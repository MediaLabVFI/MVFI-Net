import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image as imwrite
from torchvision import transforms
from MAC import Motion_aware_convolution
from optical_flow_Vis import *
def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_path',
        default='',
        type=str,
        help='prefix of input path'
    )

    parser.add_argument(
        '--output_path',
        default='',
        type=str,
        help='save path'
    )

    return parser

def SumFlow(x_1,y_1,x_2,y_2):
    coe = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
        x_1).cuda()
    ones = torch.ones(x_1.size()).float().cuda()
    zeros = torch.zeros(x_1.size()).float().cuda()
    minorOnes = torch.ones(x_1.size()).float().cuda() * (-1.0)

    temWin = torch.where(x_1 >= 1, ones, x_1)
    temWin = torch.where(temWin <= -1, minorOnes, temWin)
    windows_x1 = torch.where(abs(temWin) < 1, zeros, temWin)
    windows_x1 = windows_x1 * coe
    x_1 = windows_x1 + x_1

    temWin = torch.where(y_1 >= 1, ones, x_1)
    temWin = torch.where(temWin <= -1, minorOnes, temWin)
    windows_y1 = torch.where(abs(temWin) < 1, zeros, temWin)
    windows_y1 = windows_y1 * coe
    y_1 = windows_y1 + y_1

    temWin = torch.where(x_2 >= 1, ones, x_1)
    temWin = torch.where(temWin <= -1, minorOnes, temWin)
    windows_x2 = torch.where(abs(temWin) < 1, zeros, temWin)
    windows_x2 = windows_x2 * coe
    x_2 = windows_x2 + x_2

    temWin = torch.where(y_2 >= 1, ones, x_1)
    temWin = torch.where(temWin <= -1, minorOnes, temWin)
    windows_y2 = torch.where(abs(temWin) < 1, zeros, temWin)
    windows_y2 = windows_y2 * coe
    y_2 = windows_y2 + y_2

    x_1 = x_1[0].detach().cpu().numpy().transpose(1, 2, 0)
    x_1 = np.sum(x_1, axis=2, keepdims=True)
    y_1 = y_1[0].detach().cpu().numpy().transpose(1, 2, 0)
    y_1 = np.sum(y_1, axis=2, keepdims=True)

    x_2 = x_2[0].detach().cpu().numpy().transpose(1, 2, 0)
    x_2 = np.sum(x_2, axis=2, keepdims=True)
    y_2 = y_2[0].detach().cpu().numpy().transpose(1, 2, 0)
    y_2 = np.sum(y_2, axis=2, keepdims=True)

    flow10 = np.concatenate([x_1, y_1], axis=2)
    flow12 = np.concatenate([x_2, y_2], axis=2)
    img_flow10 = flow_to_image(flow10)
    img_flow12 = flow_to_image(flow12)

    return img_flow10,img_flow12

def main(arg):

    Kernel_H_1 = torch.from_numpy(np.load(os.path.join(arg.input_path, 'Params/Kernel_H_1.npy'))).cuda()
    Kernel_W_1 = torch.from_numpy(np.load(os.path.join(arg.input_path, 'Params/Kernel_W_1.npy'))).cuda()
    TMV_U_1 = torch.from_numpy(np.load(os.path.join(arg.input_path, 'Params/TMV_U_1.npy'))).cuda()
    TMV_V_1 = torch.from_numpy(np.load(os.path.join(arg.input_path, 'Params/TMV_V_1.npy'))).cuda()

    Kernel_H_2 = torch.from_numpy(np.load(os.path.join(arg.input_path, 'Params/Kernel_H_2.npy'))).cuda()
    Kernel_W_2 = torch.from_numpy(np.load(os.path.join(arg.input_path, 'Params/Kernel_W_2.npy'))).cuda()
    TMV_U_2 = torch.from_numpy(np.load(os.path.join(arg.input_path, 'Params/TMV_U_2.npy'))).cuda()
    TMV_V_2 = torch.from_numpy(np.load(os.path.join(arg.input_path, 'Params/TMV_V_2.npy'))).cuda()

    Mask = torch.from_numpy(np.load(os.path.join(arg.input_path, 'Params/Mask.npy'))).cuda()

    transformer = transforms.ToTensor()
    I0 = transformer(Image.open('images/im1.png')).unsqueeze(0).float().cuda()
    I2 = transformer(Image.open('images/im3.png')).unsqueeze(0).float().cuda()
    padding = ((121//11)*1 - 1)//2  #padding = ((Kernelsize//numMVs)*dilation - 1)//2
    modulePad = torch.nn.ReplicationPad2d((padding, padding, padding, padding))
    I0 = modulePad(I0)
    I2 = modulePad(I2)
    '''
    Eq.(2) MAC (Note that this output only comes from MAC, not the final interpolated frame)
    '''
    #mac = Motion_aware_convolution._FunctionMultiVK.apply()
    I_01 = Motion_aware_convolution._FunctionMultiVK.apply(I0, Kernel_H_1, Kernel_W_1, TMV_U_1, TMV_V_1, 1)
    I_21 = Motion_aware_convolution._FunctionMultiVK.apply(I2, Kernel_H_2, Kernel_W_2, TMV_U_2, TMV_V_2, 1)
    I_bar = I_01 * Mask + I_21 * (1 - Mask)
    imwrite(I_bar,'results/I1_bar.png')
    '''
    Eq.(10) Temporal MVs are weighted summed to obtain sumflow
    '''
    flow10,flow12 = SumFlow(TMV_U_1,TMV_V_1,TMV_U_2,TMV_V_2)
    plt.figure()
    plt.imshow(flow10)
    plt.axis('off')
    plt.savefig('results/Sumflow10.png')

    plt.figure()
    plt.imshow(flow12)
    plt.axis('off')
    plt.savefig('results/Sumflow12.png')
    '''
    end
    '''


if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()
    main(args)


