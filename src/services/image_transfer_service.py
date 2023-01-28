import os
import uuid

import numpy as np
import torch
from PIL import Image
from aiogram.types import PhotoSize
from torch.autograd import Variable

from src.model import style_model


class ImageTransferService:
    @staticmethod
    async def transfer(
            photo_to_transfer: PhotoSize | None,
            photo_to_get_style: PhotoSize | None,
            img_size: int = 512):

        content_root = uuid.uuid4().hex + '.png'
        style_root = uuid.uuid4().hex + '.png'
        result_root = uuid.uuid4().hex + '.png'

        await photo_to_transfer.download(destination_file=content_root)
        await photo_to_get_style.download(destination_file=style_root)

        content_image = ImageTransferService.tensor_load_rgbimage(content_root, size=img_size,
                                                                  keep_asp=True).unsqueeze(0)
        style = ImageTransferService.tensor_load_rgbimage(style_root, size=img_size).unsqueeze(0)
        style = ImageTransferService.preprocess_batch(style)
        style_v = Variable(style)
        content_image = Variable(ImageTransferService.preprocess_batch(content_image))
        style_model.setTarget(style_v)
        output = style_model(content_image)
        os.remove(content_root)
        os.remove(style_root)
        ImageTransferService.tensor_save_bgrimage(output.data[0], result_root)
        return result_root

    @staticmethod
    def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
        img = Image.open(filename).convert('RGB')
        if size is not None:
            if keep_asp:
                size2 = int(size * 1.0 / img.size[0] * img.size[1])
                img = img.resize((size, size2), Image.LANCZOS)
            else:
                img = img.resize((size, size), Image.LANCZOS)

        elif scale is not None:
            img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)),
                             Image.LANCZOS)
        img = np.array(img).transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

    @staticmethod
    def tensor_save_rgbimage(tensor, filename, cuda=False):
        if cuda:
            img = tensor.clone().cpu().clamp(0, 255).numpy()
        else:
            img = tensor.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype('uint8')
        img = Image.fromarray(img)
        img.save(filename)

    @staticmethod
    def tensor_save_bgrimage(tensor, filename, cuda=False):
        (b, g, r) = torch.chunk(tensor, 3)
        tensor = torch.cat((r, g, b))
        ImageTransferService.tensor_save_rgbimage(tensor, filename, cuda)

    @staticmethod
    def preprocess_batch(batch):
        batch = batch.transpose(0, 1)
        (r, g, b) = torch.chunk(batch, 3)
        batch = torch.cat((b, g, r))
        batch = batch.transpose(0, 1)
        return batch
