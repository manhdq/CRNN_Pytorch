from PIL import Image

import torch
from torch.autograd import Variable

from datasets import resizeNormalize
from models import CRNN
from utils.utils import strLabelConverter


def main():
    model_path = './datasets/crnn_pretrained.pth'
    img_path = './datasets/demo.jpg'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

    model = CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path)['model_state'])

    converter = strLabelConverter(alphabet)

    transformer = resizeNormalize((100, 32))
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('%-20s => %-20s' % (raw_pred, sim_pred))


if __name__ == '__main__':
    main()
