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


# class ArcLengthLoss(torch.nn.Module):
#     def __init__(self, exact_examples_count=1):
#         self._exact_examples_count = exact_examples_count
#         super(ArcLengthLoss, self).__init__()
#
#     def forward(self, output, batch_data):
#         # barrier_dim0 = int(output.shape[0] / 2)
#         # barrier_dim1 = int((output.shape[1] - 1) / 2)
#         # orig_curve =        output[:barrier_dim0,                   :barrier_dim1,                  :].squeeze(dim=2)
#         # trans_curve =       output[ barrier_dim0:2*barrier_dim0,     barrier_dim1:2*barrier_dim1,   :].squeeze(dim=2)
#         # orig_curve_add =    output[:barrier_dim0,                    (barrier_dim1 + 1):,           :].squeeze(dim=2)
#         # trans_curve_add =   output[ barrier_dim0:2*barrier_dim0,     (barrier_dim1 + 1):,           :].squeeze(dim=2)
#         # orig_curve_last =   output[:barrier_dim0,                    barrier_dim1,                  :].squeeze(dim=1)
#         # trans_curve_last =  output[ barrier_dim0:2*barrier_dim0,     barrier_dim1,                  :].squeeze(dim=1)
#
#         orig_curve =        output[0][:, :-1, :].squeeze(dim=2)
#         trans_curve =       output[1][:, :-1, :].squeeze(dim=2)
#         orig_curve_add =    output[2].squeeze(dim=2)
#         trans_curve_add =   output[3].squeeze(dim=2)
#         orig_curve_last =   output[0][:, -1, :].squeeze(dim=1)
#         trans_curve_last =  output[1][:, -1, :].squeeze(dim=1)
#
#         orig_diff = orig_curve_add - orig_curve
#         trans_diff = trans_curve_add - trans_curve
#         orig_sum = orig_diff.sum(dim=1)
#         trans_sum = trans_diff.sum(dim=1)
#         length_diff1 = (orig_sum - trans_sum).abs().exp().mean()
#         # length_diff2 = (orig_sum - orig_curve_last).abs().exp().mean()
#         # length_diff3 = (orig_sum - trans_curve_last).abs().exp().mean()
#         # length_diff4 = (trans_sum - orig_curve_last).abs().exp().mean()
#         # length_diff5 = (trans_sum - trans_curve_last).abs().exp().mean()
#         # loss1 = (length_diff2).log()
#         loss1 = (length_diff1).log()
#         # loss1 = (length_diff3).log()
#
#         # orig_diff = orig_curve_add - orig_curve
#         # trans_diff = trans_curve_add - trans_curve
#         # orig_sum = orig_diff.sum(dim=1)
#         # trans_sum = trans_diff.sum(dim=1)
#         # length_diff = (orig_sum - trans_sum).abs()
#         # loss1 = length_diff.mean(dim=0)
#
#         orig_diff2 = orig_curve - orig_curve_add
#         trans_diff2 = trans_curve - trans_curve_add
#         orig_diff2_exp = orig_diff2.exp()
#         trans_diff2_exp = trans_diff2.exp()
#         diff2_sum = orig_diff2_exp.mean(dim=1) + trans_diff2_exp.mean(dim=1)
#         loss2 = diff2_sum.mean(dim=0)
#
#         # orig_diff3 = orig_curve - trans_curve
#         # trans_diff3 = orig_curve_add - trans_curve_add
#         # orig_sum3 = orig_diff3.sum(dim=1)
#         # trans_sum3 = trans_diff3.sum(dim=1)
#         # length_diff3 = (orig_sum3 - trans_sum3).abs()
#         # loss3 = length_diff3.mean(dim=0)
#
#         return loss1 + loss2


class ArcLengthLoss(torch.nn.Module):
    def __init__(self, exact_examples_count=1):
        self._exact_examples_count = exact_examples_count
        super(ArcLengthLoss, self).__init__()

    def forward(self, output, batch_data):
        v_1_3 = output[:, 0, :].unsqueeze(dim=1)  # 1:3
        v_2_4 = output[:, 1, :].unsqueeze(dim=1)  # 2:4
        v_3_5 = output[:, 2, :].unsqueeze(dim=1)  # 3:5
        v_1_4 = output[:, 3, :].unsqueeze(dim=1)  # 1:4
        v_2_5 = output[:, 4, :].unsqueeze(dim=1)  # 2:5
        v_1_5 = output[:, 5, :].unsqueeze(dim=1)  # 1:5

        v_1_2 = output[:, 6, :].unsqueeze(dim=1)  # 1:2
        v_2_3 = output[:, 7, :].unsqueeze(dim=1)  # 2:3
        v_3_4 = output[:, 8, :].unsqueeze(dim=1)  # 3:4
        v_4_5 = output[:, 9, :].unsqueeze(dim=1)  # 4:5

        # output2 = output[:, 10:, :]

        v_1_3_sum = v_1_2 + v_2_3
        v_2_4_sum = v_2_3 + v_3_4
        v_3_5_sum = v_3_4 + v_4_5
        v_1_4_sum = v_1_2 + v_2_3 + v_3_4
        v_2_5_sum = v_2_3 + v_3_4 + v_4_5
        v_1_5_sum = v_1_2 + v_2_3 + v_3_4 + v_4_5

        v_1_2_sub = v_1_3 - v_2_3
        v_2_3_sub = v_2_4 - v_3_4
        v_3_4_sub = v_3_5 - v_4_5
        v_4_5_sub = v_3_5 - v_3_4
        v_2_3_sub2 = v_1_3 - v_1_2
        v_3_4_sub2 = v_2_4 - v_2_3
        v_1_3_sub = v_1_4 - v_3_4
        v_2_4_sub = v_2_5 - v_4_5
        v_3_5_sub = v_2_5 - v_2_3
        v_2_5_sub = v_1_5 - v_1_2
        v_1_4_sub = v_1_5 - v_4_5

        diff = torch.cat((v_1_2 - v_1_3,
                          v_1_3 - v_1_4,
                          v_1_4 - v_1_5,
                          v_2_3 - v_2_4,
                          v_2_4 - v_2_5,
                          v_3_4 - v_3_5),
                         dim=1)

        v10 = torch.cat((v_1_3, v_1_3_sum,
                         v_2_4, v_2_4_sum,
                         v_3_5, v_3_5_sum,
                         v_1_4, v_1_4_sum,
                         v_2_5, v_2_5_sum,
                         v_1_5, v_1_5_sum,
                         v_1_2, v_1_2_sub,
                         v_2_3, v_2_3_sub,
                         v_3_4, v_3_4_sub,
                         v_4_5, v_4_5_sub,
                         v_2_3, v_2_3_sub2,
                         v_3_4, v_3_4_sub2,
                         v_1_3, v_1_3_sub,
                         v_2_4, v_2_4_sub,
                         v_3_5, v_3_5_sub,
                         v_2_5, v_2_5_sub,
                         v_1_4, v_1_4_sub),
                        dim=1).abs()

        v11 = v10[:, 0::2, :]
        v12 = v10[:, 1::2, :]
        v13 = (v11 - v12).abs()
        v15 = v13.mean(dim=1)

        diff_2 = diff.mean(dim=1)
        diff_3 = diff_2.exp()

        # anchor_points_count = int(batch_data['metadata']['anchor_points_count'][0])
        # supporting_points_count = int(batch_data['metadata']['supporting_points_count'][0])
        # sections_count = anchor_points_count - 1
        # sections = [v_1_2, v_2_3, v_3_4, v_4_5]
        # loss3 = None
        # for i in range(sections_count):
        #     base_index = (2*(supporting_points_count-1)) * i
        #     section_diff1 = output2[:, base_index + 1, :].unsqueeze(dim=1) - output2[:, base_index, :].unsqueeze(dim=1)
        #     section_diff2 = output2[:, base_index + 3, :].unsqueeze(dim=1) - output2[:, base_index + 2, :].unsqueeze(dim=1)
        #     # section_diff3 = output2[:, base_index + 5, :].unsqueeze(dim=1) - output2[:, base_index + 4, :].unsqueeze(dim=1)
        #     # section_diff4 = output2[:, base_index + 7, :].unsqueeze(dim=1) - output2[:, base_index + 6, :].unsqueeze(dim=1)
        #     # loss3_tmp = (sections[i] - section_diff1 - section_diff2 - section_diff3 - section_diff4).abs()
        #     loss3_tmp = (sections[i] - section_diff1 - section_diff2).abs()
        #     if loss3 is None:
        #         loss3 = loss3_tmp
        #     else:
        #         loss3 = loss3 + loss3_tmp

        # return v15.mean(dim=0) + diff_3.mean(dim=0) + loss3.mean(dim=1).mean(dim=0)
        return v15.mean(dim=0) + diff_3.mean(dim=0)


class NegativeLoss(torch.nn.Module):
    def __init__(self, factor):
        self._factor = float(factor)
        super(NegativeLoss, self).__init__()

    def forward(self, output, batch_data):
        v = torch.max(-output, torch.zeros_like(output))
        return v.reshape([-1, 1]).mean(dim=0) * self._factor
