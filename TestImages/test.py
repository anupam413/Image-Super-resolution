import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import FENet

scale = 2

# load the model from model
model = torch.load('./final/checkpoint_x2/None_594000.pth.tar')

net = FENet.FENet(multi_scale = False, group=4, scale=2)
net.load_state_dict(model['model_state_dict'])

# read the input image
input_image = Image.open('./final/3.JPG')

input_image = input_image.convert('RGB')

transform = transforms.Compose([
    # transforms.Resize((720,576)),
    transforms.ToTensor(),
])

lr = transform(input_image)

# print(lr)
# print(lr.size())

h, w = lr.size()[1:]
h_half, w_half = int(h / 2), int(w / 2)
h_chop, w_chop = h_half + 20, w_half + 20


# split large image to 4 patch to avoid OOM error
lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop:w])
lr_patch[2].copy_(lr[:, h - h_chop:h, 0:w_chop])
lr_patch[3].copy_(lr[:, h - h_chop:h, w - w_chop:w])
lr_patch = lr_patch.cpu()


# feed each patch to the model and get the output
# pass the tensor through the model
with torch.no_grad():
    output = net(lr_patch, scale)
    # print(output)


h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale

# merge splited patch images
result = torch.FloatTensor(3, h, w).cuda()
result[:, 0:h_half, 0:w_half].copy_(output[0, :, 0:h_half, 0:w_half])
result[:, 0:h_half, w_half:w].copy_(output[1, :, 0:h_half, w_chop - w + w_half:w_chop])
result[:, h_half:h, 0:w_half].copy_(output[2, :, h_chop - h + h_half:h_chop, 0:w_half])
result[:, h_half:h, w_half:w].copy_(output[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])
sr = result

# convert the output tensor to a PIL image
transform_f = transforms.ToPILImage()
output_image = transform_f(sr)

# save the final output image
output_image.save('./final/output.jpg')
