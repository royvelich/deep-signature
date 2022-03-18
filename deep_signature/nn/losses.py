# python peripherals
import itertools

# torch
import torch


class CurvatureLoss(torch.nn.Module):
    def __init__(self):
        super(CurvatureLoss, self).__init__()

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


class ArcLengthLoss(torch.nn.Module):
    def __init__(self, anchor_points_count):
        self._anchor_points_count = anchor_points_count
        super(ArcLengthLoss, self).__init__()

    def forward(self, output):
        sample_index = 0
        section_approx = {}
        indices = list(range(self._anchor_points_count))

        for i in range(2, self._anchor_points_count):
            for index1, index2 in zip(indices, indices[i:]):
                section_approx[(index1, index2)] = output[:, sample_index, :]
                sample_index = sample_index + 1

        for index1, index2 in zip(indices, indices[1:]):
            section_approx[(index1, index2)] = output[:, sample_index, :]
            sample_index = sample_index + 1

        A = []
        B = []

        for index1 in range(self._anchor_points_count - 2):
            for index2 in range(index1+1, self._anchor_points_count - 1):
                # print(f'[{index1 + 1}, {index2 + 1}] - [{index1 + 1}, {index2 + 2}]')
                A.append((section_approx[(index1, index2)] - section_approx[(index1, index2 + 1)]))

        for i in range(3, self._anchor_points_count+1):
            for endpoints in itertools.combinations(indices, i):
                B_aux = []
                for index1, index2 in zip(endpoints, endpoints[1:]):
                    B_aux.append(section_approx[(index1, index2)])
                    # print(f'[{index1 + 1}, {index2 + 1}]', end='')

                B_aux_sum = torch.sum(torch.cat(B_aux, dim=1), dim=1).unsqueeze(dim=1)
                B.append(section_approx[(endpoints[0], endpoints[-1])])
                B.append(B_aux_sum)
                # print(f'[{endpoints[0] + 1}, {endpoints[-1] + 1}]')

        A_eval = torch.cat(A, dim=1)
        A_eval_mean = A_eval.mean(dim=1)
        A_eval_mean_exp = A_eval_mean.exp()

        B_eval = torch.cat(B, dim=1)
        B_eval1 = B_eval[:, 0::2]
        B_eval2 = B_eval[:, 1::2]
        B_eval_diff = (B_eval1 - B_eval2).abs()
        B_eval_diff_mean = B_eval_diff.mean(dim=1)

        loss_eval = (A_eval_mean_exp + B_eval_diff_mean).mean(dim=0)

        return loss_eval



# class ArcLengthLoss(torch.nn.Module):
#     def __init__(self, anchor_points_count):
#         self._anchor_points_count = anchor_points_count
#         super(ArcLengthLoss, self).__init__()
#
#     def forward(self, output):
#         v_1_3 = output[:, 0, :].unsqueeze(dim=1)  # 1:3
#         v_2_4 = output[:, 1, :].unsqueeze(dim=1)  # 2:4
#         v_3_5 = output[:, 2, :].unsqueeze(dim=1)  # 3:5
#         v_1_4 = output[:, 3, :].unsqueeze(dim=1)  # 1:4
#         v_2_5 = output[:, 4, :].unsqueeze(dim=1)  # 2:5
#         v_1_5 = output[:, 5, :].unsqueeze(dim=1)  # 1:5
#
#         v_1_2 = output[:, 6, :].unsqueeze(dim=1)  # 1:2
#         v_2_3 = output[:, 7, :].unsqueeze(dim=1)  # 2:3
#         v_3_4 = output[:, 8, :].unsqueeze(dim=1)  # 3:4
#         v_4_5 = output[:, 9, :].unsqueeze(dim=1)  # 4:5
#
#         v_1_3_sum = v_1_2 + v_2_3
#         v_2_4_sum = v_2_3 + v_3_4
#         v_3_5_sum = v_3_4 + v_4_5
#         v_1_4_sum = v_1_2 + v_2_3 + v_3_4
#         v_2_5_sum = v_2_3 + v_3_4 + v_4_5
#         v_1_5_sum = v_1_2 + v_2_3 + v_3_4 + v_4_5
#
#         v_1_2_sub = v_1_3 - v_2_3
#         v_2_3_sub = v_2_4 - v_3_4
#         v_3_4_sub = v_3_5 - v_4_5
#         v_4_5_sub = v_3_5 - v_3_4
#         v_2_3_sub2 = v_1_3 - v_1_2
#         v_3_4_sub2 = v_2_4 - v_2_3
#         v_1_3_sub = v_1_4 - v_3_4
#         v_2_4_sub = v_2_5 - v_4_5
#         v_3_5_sub = v_2_5 - v_2_3
#         v_2_5_sub = v_1_5 - v_1_2
#         v_1_4_sub = v_1_5 - v_4_5
#
#         diff = torch.cat((v_1_2 - v_1_3,
#                           v_1_3 - v_1_4,
#                           v_1_4 - v_1_5,
#                           v_2_3 - v_2_4,
#                           v_2_4 - v_2_5,
#                           v_3_4 - v_3_5),
#                          dim=1)
#
#         v10 = torch.cat((v_1_3, v_1_3_sum,
#                          v_2_4, v_2_4_sum,
#                          v_3_5, v_3_5_sum,
#                          v_1_4, v_1_4_sum,
#                          v_2_5, v_2_5_sum,
#                          v_1_5, v_1_5_sum,
#                          v_1_2, v_1_2_sub,
#                          v_2_3, v_2_3_sub,
#                          v_3_4, v_3_4_sub,
#                          v_4_5, v_4_5_sub,
#                          v_2_3, v_2_3_sub2,
#                          v_3_4, v_3_4_sub2,
#                          v_1_3, v_1_3_sub,
#                          v_2_4, v_2_4_sub,
#                          v_3_5, v_3_5_sub,
#                          v_2_5, v_2_5_sub,
#                          v_1_4, v_1_4_sub),
#                         dim=1).abs()
#
#         v11 = v10[:, 0::2, :]
#         v12 = v10[:, 1::2, :]
#         v13 = (v11 - v12).abs()
#         v15 = v13.mean(dim=1)
#
#         diff_2 = diff.mean(dim=1)
#         diff_3 = diff_2.exp()
#
#         return v15.mean(dim=0) + diff_3.mean(dim=0)
