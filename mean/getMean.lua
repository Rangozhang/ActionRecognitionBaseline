import 'torch'
import 'nn'

local net = torch.load('nin_nobn_final.t7'):unpack():float()
img_mean = net.transform
