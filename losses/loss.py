import sys
import torch
import torch.nn as nn
import torch.backends.cudnn


class CTCLoss(nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        super(CTCLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
        target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
        dt = log_probs.dtype
        log_probs = log_probs.double()  # we need the accuracy as we are not in logspace
        targets = targets.long()
        cum_target_lengths = target_lengths.cumsum(0)
        losses = []
        for i in range(log_probs.size(1)):
            input_length = input_lengths[i].item()
            target_length = target_lengths[i].item()
            cum_target_length = cum_target_lengths[i].item()
            # =============================================================================================
            targets_prime = targets.new_full((2 * target_length + 1,), self.blank)
            if targets.dim() == 2:
                targets_prime[1::2] = targets[i, : target_length]
            else:
                targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
            # ==============================================================================================
            probs = log_probs[:input_length, i].exp()
            # ==============================================================================================
            alpha = log_probs.new_zeros((target_length * 2 + 1))
            alpha[0] = probs[0, self.blank]
            alpha[1] = probs[0, targets_prime[1]]
            mask_third = (targets_prime[:-2] != targets_prime[2:])
            for t in range(1, input_length):
                alpha_next = alpha.clone()
                alpha_next[1:] += alpha[:-1]
                alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
                alpha = probs[t, targets_prime] * alpha_next
            # ==============================================================================================
            losses.append(-alpha[-2:].sum().log()[None])
        output = torch.cat(losses, 0)
        if self.reduction == 'mean':
            return (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()
        elif self.reduction == 'sum':
            return output.sum()
        output = output.to(dt)
        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_lengths = [30, 25, 20]
    input_lengths = [50, 50, 50]
    targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int)
    log_probs = torch.randn(50, 3, 16, dtype=torch.float, device=device).log_softmax(2)

    ctc_ref = CTCLoss()
    ctc_loss = nn.CTCLoss()

    res = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    res_ref = ctc_ref(log_probs, targets.to(device), input_lengths, target_lengths).float()

    print("Built-in CTC loss:", res)
    print("Reference CTC loss:", res_ref)
    print(torch.all(res.eq(res_ref)))