# python peripherals
import itertools

# torch
import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, mu=1):
        super(ContrastiveLoss, self).__init__()
        self._mu = mu

    def forward(self, output, batch_data):
        v = output[:, 0, :]
        v2 = v.unsqueeze(dim=1)
        v3 = v2 - output
        v4 = v3.abs().squeeze(dim=2)
        v5 = v4[:, 1:].squeeze(dim=1)
        labels = batch_data['labels'].squeeze(dim=1)
        penalties = labels * v5 + (1 - labels) * torch.max(torch.zeros_like(v5), self._mu - v5)
        return penalties.mean()


class TupletLoss(torch.nn.Module):
    def __init__(self):
        super(TupletLoss, self).__init__()

    def forward(self, output, batch_data):
        v = output[:, 0, :]
        v2 = v.unsqueeze(dim=1)
        v3 = v2 - output
        v4 = v3.abs().squeeze(dim=2)
        v5 = v4[:, 1:]
        v6 = v5[:, 0]
        v7 = v6.unsqueeze(dim=1)
        v8 = v7 - v5
        v9 = v8[:, 1:]
        v10 = v9.exp()
        v11 = v10.sum(dim=1)
        v12 = v11 + 1
        v13 = v12.log()
        return v13.mean(dim=0)


class CurvatureLoss(torch.nn.Module):
    def __init__(self):
        super(CurvatureLoss, self).__init__()

    def forward(self, output, batch_data):
        samples_count = 4
        positives = []
        positives_flipped = []
        negatives = []

        for i in range(samples_count):
            positives.append(output[:, i, :])
            positives_flipped.append(output[:, i + samples_count, :])
            negatives.append(output[:, i + 2*samples_count, :])

        positive_pairs = list(itertools.combinations(positives, 2))

        positive_diffs = []
        for positive_pair in positive_pairs:
            positive_diffs.append((positive_pair[0] - positive_pair[1]).abs())

        for i in range(samples_count):
            positive_diffs.append((positives[i] + positives_flipped[i]).abs())

        objective = torch.cat(positive_diffs, dim=1).mean(dim=1)

        negative_diffs = []
        for i in range(samples_count):
            for j in range(samples_count):
                negative_diffs.append((positives[i] - negatives[j]).abs().neg())

        regularization = torch.cat(negative_diffs, dim=1).exp().mean(dim=1)

        loss = (objective + regularization).mean(dim=0)

        return loss


class ArcLengthLoss(torch.nn.Module):
    def __init__(self, exact_examples_count=1):
        self._exact_examples_count = exact_examples_count
        super(ArcLengthLoss, self).__init__()

    def forward(self, output, batch_data):
        losses = []
        first = True
        for [orig_short, orig_long, trans_short, trans_long] in output:
            orig_short_sample = orig_short[:, :-2, :].squeeze(dim=2)
            orig_long_sample = orig_long.squeeze(dim=2)
            orig_short_sample_first = orig_short[:, 0, :].squeeze(dim=1)
            orig_short_sample_last = orig_short[:, -2, :].squeeze(dim=1)
            orig_short_sample_stretch = orig_short[:, -1, :].squeeze(dim=1)

            trans_short_sample = trans_short[:, :-2, :].squeeze(dim=2)
            trans_long_sample = trans_long.squeeze(dim=2)
            trans_short_sample_first = trans_short[:, 0, :].squeeze(dim=1)
            trans_short_sample_last = trans_short[:, -2, :].squeeze(dim=1)
            trans_short_sample_stretch = trans_short[:, -1, :].squeeze(dim=1)

            orig_length_diff = orig_long_sample - orig_short_sample
            orig_length_diff_sum = orig_length_diff.sum(dim=1)

            trans_length_diff = trans_long_sample - trans_short_sample
            trans_length_diff_sum = trans_length_diff.sum(dim=1)

            # loss1 = (orig_length_diff_sum - orig_short_sample_last).abs().exp().mean().log()
            # loss1 = (orig_length_diff_sum - trans_length_diff_sum).abs().exp().mean().log()
            loss2 = ((orig_short_sample - trans_short_sample) + (orig_long_sample - trans_long_sample)).abs().exp().mean().log()
            loss3 = ((orig_short_sample - orig_long_sample) + (trans_short_sample - trans_long_sample)).exp().mean(dim=1).mean(dim=0)
            # loss3 = (orig_short_sample_first + orig_short_sample_last - orig_short_sample_stretch).abs().exp().mean().log()
            if first is True:
                # acc_loss1 = loss1
                acc_loss2 = loss2
                acc_loss3 = loss3
                first = False
            else:
                # acc_loss1 = acc_loss1 + loss1
                acc_loss2 = acc_loss2 + loss2
                acc_loss3 = acc_loss3 + loss3

        return acc_loss2 + acc_loss3
        # return acc_loss1 + acc_loss2 + acc_loss3


class NegativeLoss(torch.nn.Module):
    def __init__(self, factor):
        self._factor = float(factor)
        super(NegativeLoss, self).__init__()

    def forward(self, output, batch_data):
        v = torch.max(-output, torch.zeros_like(output))
        return v.reshape([-1, 1]).mean(dim=0) * self._factor
