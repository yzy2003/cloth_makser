import os
from PIL import Image
from typing import Union
import numpy as np
import cv2
from diffusers.image_processor import VaeImageProcessor
import torch

from model.SCHP import SCHP  # type: ignore
from model.DensePose import DensePose  # type: ignore

import torch.nn as nn
import torch.nn.functional as F

# 定义DensePose模型中不同人体部位的索引映射
DENSE_INDEX_MAP = {
    "background": [0],
    "torso": [1, 2],
    "right hand": [3],
    "left hand": [4],
    "right foot": [5],
    "left foot": [6],
    "right thigh": [7, 9],
    "left thigh": [8, 10],
    "right leg": [11, 13],
    "left leg": [12, 14],
    "left big arm": [15, 17],
    "right big arm": [16, 18],
    "left forearm": [19, 21],
    "right forearm": [20, 22],
    "face": [23, 24],
    "thighs": [7, 8, 9, 10],
    "legs": [11, 12, 13, 14],
    "hands": [3, 4],
    "feet": [5, 6],
    "big arms": [15, 16, 17, 18],
    "forearms": [19, 20, 21, 22],
}

# 用于服装分割
# 定义ATR模型中的部位映射
ATR_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3, 
    'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 'Dress': 7,
    'Belt': 8, 'Left-shoe': 9, 'Right-shoe': 10, 'Face': 11, 
    'Left-leg': 12, 'Right-leg': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Bag': 16, 'Scarf': 17
}

# 定义LIP模型中的部位映射
LIP_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Glove': 3, 
    'Sunglasses': 4, 'Upper-clothes': 5, 'Dress': 6, 'Coat': 7,
    'Socks': 8, 'Pants': 9, 'Jumpsuits': 10, 'Scarf': 11, 
    'Skirt': 12, 'Face': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Left-leg': 16, 'Right-leg': 17, 'Left-shoe': 18, 'Right-shoe': 19
}

# 定义生成掩码时需要保护的身体部位
PROTECT_BODY_PARTS = {
    'upper': ['Left-leg', 'Right-leg'],
    'lower': ['Right-arm', 'Left-arm', 'Face'],
    'overall': [],
    'inner': ['Left-leg', 'Right-leg'],
    'outer': ['Left-leg', 'Right-leg'],
}

# 定义生成掩码时需要保护的服装部分，使用两个不同模型的映射
PROTECT_CLOTH_PARTS = {
    'upper': {
        'ATR': ['Skirt', 'Pants'],
        'LIP': ['Skirt', 'Pants']
    },
    'lower': {
        'ATR': ['Upper-clothes'],
        'LIP': ['Upper-clothes', 'Coat']
    },
    'overall': {
        'ATR': [],
        'LIP': []
    },
    'inner': {
        'ATR': ['Dress', 'Coat', 'Skirt', 'Pants'],
        'LIP': ['Dress', 'Coat', 'Skirt', 'Pants', 'Jumpsuits']
    },
    'outer': {
        'ATR': ['Dress', 'Pants', 'Skirt'],
        'LIP': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Jumpsuits']
    }
}

# 定义不同类型中需要生成掩码的服装部分
MASK_CLOTH_PARTS = {
    'upper': ['Upper-clothes', 'Coat', 'Dress', 'Jumpsuits'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat',]
}
# 定义了在生成掩码时需要考虑的人体部位
MASK_DENSE_PARTS = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms']
}

# 公共需要保护的服装配饰部分   
schp_public_protect_parts = ['Hat', 'Hair', 'Sunglasses', 'Left-shoe', 'Right-shoe', 'Bag', 'Glove', 'Scarf']

schp_protect_parts = {
    'upper': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits'],  
    'lower': ['Left-arm', 'Right-arm', 'Upper-clothes', 'Coat'],
    'overall': [],
    'inner': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits', 'Coat'],
    'outer': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits', 'Upper-clothes']
}

# 定义使用SCHP模型时生成掩码的服装部分
schp_mask_parts = {
    'upper': ['Upper-clothes', 'Dress', 'Coat', 'Jumpsuits'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits', 'socks'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits', 'socks'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat',]
}

# 定义使用DensePose模型时生成掩码的人体部位
dense_mask_parts = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms']
}

def vis_mask(image, mask):
    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    # 掩码二值化，将掩码转换为纯黑白图像
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    # 掩码归一化
    mask = mask / 255
    return Image.fromarray((image * (1 - mask)).astype(np.uint8))

def part_mask_of(part: Union[str, list],
                 parse: np.ndarray, mapping: dict):
    if isinstance(part, str):
        part = [part]
    mask = np.zeros_like(parse)
    for _ in part:
        if _ not in mapping:
            continue
        if isinstance(mapping[_], list):
            for i in mapping[_]:
                mask += (parse == i)
        else:
            mask += (parse == mapping[_])
    return mask

# 返回该掩码区域的凸包掩码
def hull_mask(mask_area: np.ndarray):
    # 掩码二值化：使用 cv2.threshold 将 mask_area 转换为二值图像，像素值大于127的部分设为255，其他部分设为0。
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    # 轮廓检测：使用 cv2.findContours 在二值图像中检测轮廓，只保留外部轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 生成凸包掩码：
    # 初始化一个与 mask_area 形状相同的全零数组 hull_mask。
    # 对于每个检测到的轮廓 c，计算它的凸包，并将凸包区域填充为白色（255），再将该区域与 hull_mask 进行或运算。
    hull_mask = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        hull_mask = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255) | hull_mask
    return hull_mask
    
# Unet网络
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # 定义 U-Net 的卷积块
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.middle = self.conv_block(512, 1024)
        
        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder part
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        
        # Middle part
        m = self.middle(F.max_pool2d(e4, 2))
        
        # Decoder part
        d4 = self.decoder4(torch.cat([F.upsample(m, scale_factor=2, mode='bilinear'), e4], 1))
        d3 = self.decoder3(torch.cat([F.upsample(d4, scale_factor=2, mode='bilinear'), e3], 1))
        d2 = self.decoder2(torch.cat([F.upsample(d3, scale_factor=2, mode='bilinear'), e2], 1))
        d1 = self.decoder1(torch.cat([F.upsample(d2, scale_factor=2, mode='bilinear'), e1], 1))
        
        return torch.sigmoid(self.final(d1))

class AutoMasker:
    # 初始化AutoMaster的实例，接受DensePose和SCHP模型的检查点路径以及设备
    def __init__(
        self, 
        densepose_ckpt='./Models/DensePose', 
        schp_ckpt='./Models/SCHP', 
        device='cuda'):
        # 初始化Unet模型
        self.unet_model = UNet(in_channels =3,out_channels=1).to(device)

        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        self.densepose_processor = DensePose(densepose_ckpt, device)
        self.schp_processor_atr = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908301523-atr.pth'), device=device)
        self.schp_processor_lip = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908261155-lip.pth'), device=device)
        
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
    # DensePose处理，将图像大小调整为1024*1024
    def process_densepose(self, image_or_path):
        return self.densepose_processor(image_or_path, resize=1024)
    # SCHP处理：使用LIP和ATR变体处理输入图像
    def process_schp_lip(self, image_or_path):
        return self.schp_processor_lip(image_or_path)

    def process_schp_atr(self, image_or_path):
        return self.schp_processor_atr(image_or_path)
    # 预处理图像：调用DensePose和SCHP模型对输入图像进行处理，并返回处理后的结果字典，包括DensePose结果、SCHP ATR结果和SCHP LIP结果    
    def preprocess_image(self, image_or_path):
        return {
            'densepose': self.densepose_processor(image_or_path, resize=1024),
            'schp_atr': self.schp_processor_atr(image_or_path),
            'schp_lip': self.schp_processor_lip(image_or_path)
        }
    
    @staticmethod
    def cloth_agnostic_mask(
        densepose_mask: Image.Image,
        schp_lip_mask: Image.Image,    # SCHP模型中生成的LIP掩码，用于分割图像中的服装部位
        schp_atr_mask: Image.Image,    # 用于进一步分割衣物种类。
        part: str='overall',
        unet_mask=None,
        **kwargs
    ):
        assert part in ['upper', 'lower', 'overall', 'inner', 'outer'], f"part should be one of ['upper', 'lower', 'overall', 'inner', 'outer'], but got {part}"
        # 获取densepose_mask图像的宽度和高度
        w, h = densepose_mask.size
        
        # 计算膨胀核的大小。
        # 通过将图像的最大边长除以 250 来确定膨胀核的尺寸。这种比例因子通常基于经验选择，用于确保膨胀核的大小适合图像的分辨率
        dilate_kernel = max(w, h) // 250
        # 确保膨胀核大小是奇数，保证在图像处理操作中，确保处理效果的对称性
        dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
        # 创建一个全为1的矩阵，用于膨胀操作的核
        dilate_kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        
        # 计算高斯模糊操作的核大小，高斯模糊核的大小要大一些，以提供足够的模糊效果
        kernal_size = max(w, h) // 25
        kernal_size = kernal_size if kernal_size % 2 == 1 else kernal_size + 1
        
        # 将图像掩码转换为Numpy数组，densepose_mask、schp_lip_mask、schp_atr_mask输入的掩码图像
        densepose_mask = np.array(densepose_mask)
        schp_lip_mask = np.array(schp_lip_mask)
        schp_atr_mask = np.array(schp_atr_mask)
        
        # 生成强保护区域掩码
        # Strong Protect Area (Hands, Face, Accessory, Feet)
        # 生成手和脚的保护区域，使用part_mask_of函数生成手和脚区域的掩码
        hands_protect_area = part_mask_of(['hands', 'feet'], densepose_mask, DENSE_INDEX_MAP)
        # 使用膨胀操作扩大hands_protect_area掩码中的手和脚的区域
        hands_protect_area = cv2.dilate(hands_protect_area, dilate_kernel, iterations=1)
        # 将手和脚的保护区域与手臂和腿部区域相交，以限制保护区域，得到一个更精确的hands_protect_area掩码
        hands_protect_area = hands_protect_area & \
            (part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_atr_mask, ATR_MAPPING) | \
             part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_lip_mask, LIP_MAPPING))
        # 生成脸部保护区域
        # 使用part_mask_of函数生成一个掩码，该掩码对应schp_lip_mask中表示脸部的区域。
        face_protect_area = part_mask_of('Face', schp_lip_mask, LIP_MAPPING)

        # 合并保护区域，将手脚和脸保护区域合并，生成最终请保护区域掩码
        strong_protect_area = hands_protect_area | face_protect_area 

        # 生成弱保护区域掩码，用于保护头发、不相关衣物、身体部分即配饰等区域
        # Weak Protect Area (Hair, Irrelevant Clothes, Body Parts)
        # 生成 body_protect_area 掩码，用于保护与身体部位相关的区域
        body_protect_area = part_mask_of(PROTECT_BODY_PARTS[part], schp_lip_mask, LIP_MAPPING) | part_mask_of(PROTECT_BODY_PARTS[part], schp_atr_mask, ATR_MAPPING)
        # 生成头发的保护区域
        hair_protect_area = part_mask_of(['Hair'], schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(['Hair'], schp_atr_mask, ATR_MAPPING)
        # 生成衣物的保护区域
        cloth_protect_area = part_mask_of(PROTECT_CLOTH_PARTS[part]['LIP'], schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(PROTECT_CLOTH_PARTS[part]['ATR'], schp_atr_mask, ATR_MAPPING)
        # 生成配饰的保护区域
        accessory_protect_area = part_mask_of((accessory_parts := ['Hat', 'Glove', 'Sunglasses', 'Bag', 'Left-shoe', 'Right-shoe', 'Scarf', 'Socks']), schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(accessory_parts, schp_atr_mask, ATR_MAPPING) 
        # 合并所有的弱保护区域
        weak_protect_area = body_protect_area | cloth_protect_area | hair_protect_area | strong_protect_area | accessory_protect_area
        
        # Mask Area
        # 生成strong_mask_area，用于标识图像中需要特别处理的区域。
        # 从 schp_lip_mask 和 schp_atr_mask 中提取对应的衣物部分，并进行逻辑或运算 (|)，将结果合并为 strong_mask_area 掩码
        strong_mask_area = part_mask_of(MASK_CLOTH_PARTS[part], schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(MASK_CLOTH_PARTS[part], schp_atr_mask, ATR_MAPPING)
        # 生成background_area，标识图像中的背景部分
        # 从 schp_lip_mask 和 schp_atr_mask 中提取背景区域，并进行逻辑与运算 (&)，确保只有两个掩码都标记为背景的部分才会被包含在 background_area 中。
        background_area = part_mask_of(['Background'], schp_lip_mask, LIP_MAPPING) & part_mask_of(['Background'], schp_atr_mask, ATR_MAPPING)
        # 生成mask_dense_area，用于标识需要重点处理的密集区域
        # 提取 DensePose 掩码中的密集区域，生成 mask_dense_area。
        mask_dense_area = part_mask_of(MASK_DENSE_PARTS[part], densepose_mask, DENSE_INDEX_MAP)
        # 将掩码缩小至原尺寸的 1/4，以减少计算量，同时保留重要的细节。
        mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        # 使用膨胀操作扩展这些区域，防止在后续处理中丢失边缘信息。
        mask_dense_area = cv2.dilate(mask_dense_area, dilate_kernel, iterations=2)
        # 最后将掩码恢复至原始尺寸的 4 倍，以适应图像的原始分辨率。
        mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

        # 生成最终的掩码区域
        # 使用 np.ones_like(densepose_mask) 创建一个与 densepose_mask 大小相同的全 1 数组，表示全图范围。
        # (~weak_protect_area) & (~background_area) 表示将弱保护区域和背景区域从全图中排除。
        # | mask_dense_area 表示将密集区域掩码合并到初步掩码中。
        # 处理 U-Net 生成的掩码
        if unet_mask is not None:
            # 将 U-Net 掩码与其他掩码结合
            mask_area = unet_mask & (~weak_protect_area)
        else:
            # 如果没有 U-Net 掩码，使用原有的 mask 生成逻辑
            mask_area = (np.ones_like(densepose_mask) & (~weak_protect_area) & (~background_area)) | mask_dense_area

        # 扩展和模糊掩码区域
        # 使用 hull_mask 函数应用凸包操作，将掩码区域扩展，以覆盖更大的目标区域。
        mask_area = hull_mask(mask_area * 255) // 255  # Convex Hull to expand the mask area
        # 再次排除弱保护区域 (weak_protect_area)。
        mask_area = mask_area & (~weak_protect_area)
        # 应用高斯模糊 (cv2.GaussianBlur) 来平滑掩码边缘，以去除可能的噪声和不连续的部分。
        mask_area = cv2.GaussianBlur(mask_area * 255, (kernal_size, kernal_size), 0)
        # 将模糊后的掩码进行阈值处理：将小于 25 的值设为 0，大于等于 25 的值设为 1，以生成二值化掩码。
        mask_area[mask_area < 25] = 0
        mask_area[mask_area >= 25] = 1
        # 生成最终的掩码区域
        # 将生成的 mask_area 与 strong_mask_area 合并，并排除 strong_protect_area。
        mask_area = (mask_area | strong_mask_area) & (~strong_protect_area) 
        # 最后进行一次膨胀操作，进一步扩展掩码区域，以确保重要区域被正确覆盖。
        mask_area = cv2.dilate(mask_area, dilate_kernel, iterations=1)

        return Image.fromarray(mask_area * 255)
        
    def __call__(
        self,
        image: Union[str, Image.Image],
        mask_type: str = "upper",
    ):
        assert mask_type in ['upper', 'lower', 'overall', 'inner', 'outer'], f"mask_type should be one of ['upper', 'lower', 'overall', 'inner', 'outer'], but got {mask_type}"
        preprocess_results = self.preprocess_image(image)
        # 将预处理图像出入Unet模型
        unet_input = preprocess_results['schp_lip']  # 需调整
        unet_output = self.unet_model(unet_input)
        # 后处理Unet输出
        threshold = 0.5
        binary_mask = (unet_output > threshold).float()
        binary_mask = binary_mask.cpu().detach().numpy() * 255
        binary_mask = cv2.dilate(binary_mask, self.dilate_kernel, iterations=1)
        # 生成衣物无关的掩码
        mask = self.cloth_agnostic_mask(
            preprocess_results['densepose'], 
            preprocess_results['schp_lip'], 
            preprocess_results['schp_atr'], 
            part=mask_type,
            unet_mask=binary_mask # 将Unet生成的掩码传递给cloth_agnostic_mask
        )
        # 将生成的掩码和预处理结果返回给调用者
        return {
            'mask': mask,
            'densepose': preprocess_results['densepose'],
            'schp_lip': preprocess_results['schp_lip'],
            'schp_atr': preprocess_results['schp_atr']
        }


if __name__ == '__main__':
    pass
