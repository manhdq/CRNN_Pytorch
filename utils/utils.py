# encoding: utf-8

import os
import collections
import torch
import torch.nn as nn
from torch.autograd import Variable


class strLabelConverter(object):
    """
    Convert between str and label.
    
    NOTE:
        insert 'blank' to the alphabet for CTC.
    
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for '-1' index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Suport batch single str.
        
        Args:
            text (str or list of str): texts to convert.
            
        Returns:
            torch.IntTensor [length_0 + length_1 + ... + length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [self.dict[char.lower() if self._ignore_case else char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """
        Decode encoded texts back into strs
        
        Args:
            torch.IntTensor [length_0 + length_1 + ... + length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text
            
        Raises:
            AssertionError: when the texts and its length does not match
            
        Returns:
            text (str of list of str): texts to convert
        """
        t = t.cpu()
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(int(length)):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

    
class averager(object):
    """Compute average for torch.Variable and torch.Tensor."""
    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)