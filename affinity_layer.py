import torch
import torch.nn as nn
import numpy as np
class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d=512):
        super(Affinity, self).__init__()
        self.d = d

        self.fc_M = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)

        )

        self.project_sr = nn.Linear(256, 256,bias=False)
        self.project_tg = nn.Linear(256, 256,bias=False)

        self.reset_parameters()


    def reset_parameters(self):

        for i in self.fc_M:
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, std=0.01)
                nn.init.constant_(i.bias, 0)


        nn.init.normal_(self.project_sr.weight, std=0.01)
        nn.init.normal_(self.project_tg.weight, std=0.01)

        # The common GM design doesn;t work!!
        # stdv = 1. / math.sqrt(self.d)
        # self.A.data.uniform_(-stdv, stdv)
        # self.A.data += torch.eye(self.d).cuda()
        # nn.init.normal_(self.project_2.weight, std=0.01)
        # nn.init.normal_(self.project2.weight, std=0.01)
        # nn.init.constant_(i.bias, 0)
    def forward(self, X, Y):
        X, Y = X.cpu(), Y.cpu()
        X = self.project_sr(X)
        Y = self.project_tg(Y)

        N1, C = X.size()
        N2, C = Y.size()

        X_k = X.unsqueeze(1).expand(N1, N2, C)
        Y_k = Y.unsqueeze(0).expand(N1, N2, C)
        M = torch.cat([X_k, Y_k], dim=-1)
        M = self.fc_M(M).squeeze()

        # The common GM design doesn;t work!!

        # M = self.affinity_pred(M[None,]).squeeze()
        # M_r = self.fc_M(M_r).squeeze()
        # M = torch.matmul(X, (self.A + self.A.transpose(0, 1).contiguous()) / 2)
        # M = torch.matmul(M, Y.transpose(0, 1).contiguous())

        return M
def sinkhorn_rpm(log_alpha, n_iters=5, slack=True, eps=-1):
    ''' Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)

        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    '''
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                    dim=1)
            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                    dim=2)
            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()
        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()
    return log_alpha
def one_hot(x, num_class=None):
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((len(x), num_class))
    ohx[range(len(x)), x] = 1
    return ohx

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        # pt = torch.clamp(torch.sigmoid(_input), min=0.00001)
        # pt = torch.clamp(pt, max=0.99999)
        pt = _input
        alpha = self.alpha

        # pos = torch.nonzero(target[:,1] > 0).squeeze(1).numel()
        # print(pos)

        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction =='pos':
            loss = torch.sum(loss) / (2*pos)


        return loss

class MultiHeadAttention(nn.Module):
    """ 多头自注意力"""
    def __init__(self, model_dim=256, num_heads=4, dropout=0.0, version='v2'):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim//num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.version  = version

    def forward(self, key, value, query, attn_mask=None):

        if self.version == 'v2':

            B =1
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            query = query.unsqueeze(1)
            residual = query


            dim_per_head = self.dim_per_head
            num_heads = self.num_heads


            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)
            key = key.view(key.size(0), B * num_heads, dim_per_head).transpose(0,1)
            value = value.view(value.size(0), B * num_heads, dim_per_head).transpose(0,1)
            query = query.view(query.size(0), B * num_heads, dim_per_head).transpose(0,1)

            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, key.size(1), num_heads, scale, attn_mask)
            # (query, key, value, scale, attn_mask)
            context = context.transpose(0, 1).contiguous().view(query.size(1), B, dim_per_head * num_heads)
            output = self.linear_final(context)
            # dropout
            output = self.dropout(output)

            output = self.layer_norm(residual + output)
            # output = residual + output

        elif self.version == 'v1':

            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            query = query.unsqueeze(0)
            residual = query
            B, L, C = key.size()
            dim_per_head = self.dim_per_head
            num_heads = self.num_heads
            batch_size = key.size(0)

            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            key = key.view(batch_size * num_heads, -1, dim_per_head)
            value = value.view(batch_size * num_heads, -1, dim_per_head)
            query = query.view(batch_size * num_heads, -1, dim_per_head)


            if attn_mask:
                attn_mask = attn_mask.repeat(num_heads, 1, 1)

            # 缩放点击注意力机制
            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

            context = context.view(batch_size, -1, dim_per_head * num_heads)

            output = self.linear_final(context)


            output = self.dropout(output)
            output = self.layer_norm(residual + output)


        return output.squeeze(), attention.squeeze()

class dot_attention(nn.Module):
    """ 点积注意力机制"""

    def __init__(self, attention_dropout=0.0, topk=30):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.topk = topk
    def forward(self, q, k, v, B, num_heads, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        # 是否设置缩放
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     # 给需要mask的地方设置一个负无穷。


        mask = torch.zeros(attention.size(0), B, B)
        index = torch.topk(attention, k=self.topk, dim=-1, largest=True)[1]
        # print(mask.size())
        # print(index.size())

        mask.scatter_(-1, index, 1.)
        attention = torch.where(mask > 0, attention, torch.full_like(attention, float('-inf')))

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


