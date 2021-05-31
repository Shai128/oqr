import torch
from helper import compute_coverages_and_avg_interval_len, pearsons_corr2d, HSIC



def independence_penalty(y, pred_l, pred_u, pearsons_corr_multiplier=1, hsic_multiplier=0, y_multiplier=100):

    """
    Computes the independence penalty given the true label and the prediced upper and lower quantiles.
    Parameters
    ----------
    y - the true label of a feature vector.
    pred_l - the predicted lower bound
    pred_u - the prediced upper bound
    pearsons_corr_multiplier - multiplier of R_corr
    hsic_multiplier - multiplier of R_HSIC
    y_multiplier - multiplier of y for numeric stability

    Returns
    -------
    The independence penalty R
    """

    if pearsons_corr_multiplier == 0 and hsic_multiplier == 0:
        return 0
    
    is_in_interval, interval_sizes = compute_coverages_and_avg_interval_len(y.view(-1) * y_multiplier,
                                                                            pred_l * y_multiplier,
                                                                            pred_u * y_multiplier)
    partial_interval_sizes = interval_sizes[abs(torch.min(is_in_interval, dim=1)[0] -
                                                torch.max(is_in_interval, dim=1)[0]) > 0.05, :]
    partial_is_in_interval = is_in_interval[abs(torch.min(is_in_interval, dim=1)[0] -
                                                torch.max(is_in_interval, dim=1)[0]) > 0.05, :]
    
    if partial_interval_sizes.shape[0] > 0 and pearsons_corr_multiplier != 0:
        corrs = pearsons_corr2d(partial_interval_sizes, partial_is_in_interval)
        pearsons_corr_loss = torch.mean((torch.abs(corrs)))
        if pearsons_corr_loss.isnan().item():
            pearsons_corr_loss = 0
    else:
        pearsons_corr_loss = 0
    
    hsic_loss = 0
    if partial_interval_sizes.shape[0] > 0 and hsic_multiplier != 0:
        n = partial_is_in_interval.shape[1]
        data_size_for_hsic = 512
        for i in range(partial_is_in_interval.shape[0]):

            v = partial_is_in_interval[i, :].reshape((n,1))
            l = partial_interval_sizes[i, :].reshape((n,1))
            v = v[:data_size_for_hsic]
            l = l[:data_size_for_hsic]
            if torch.max(v) - torch.min(v) > 0.05:  # in order to not get hsic = 0
                curr_hsic = torch.abs(torch.sqrt(HSIC(v, l)))
            else:
                curr_hsic = 0

            hsic_loss += curr_hsic
        hsic_loss = hsic_loss / partial_interval_sizes.shape[0]
    
    penalty = pearsons_corr_loss * pearsons_corr_multiplier + hsic_loss * hsic_multiplier

    return penalty


def batch_qr_loss(model, y, x, q_list, device, args):
    num_pts = y.size(0)

    with torch.no_grad():
        l_list = torch.min(torch.stack([q_list, 1 - q_list], dim=1), dim=1)[0].to(device)
        u_list = 1.0 - l_list

    q_list = torch.cat([l_list, u_list],dim=0)
    num_q = q_list.shape[0]

    l_rep = l_list.view(-1, 1).repeat(1, num_pts).view(-1, 1)
    num_l = l_rep.size(0)

    q_rep = q_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    y_stacked = y.repeat(num_q, 1)

    if x is None:
        model_in = torch.cat([l_list, u_list], dim=0)
    else:
        x_stacked = x.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, q_rep], dim=1)

    pred_y = model(model_in)

    diff = pred_y - y_stacked
    mask = (diff.ge(0).float() - q_rep).detach()  # / q_rep

    pinball_loss = ((mask * diff).mean())

    pearsons_corr_multiplier = args.corr_mult * 0.1
    hsic_multiplier = args.hsic_mult
    pred_l = pred_y[:num_l].view(num_q // 2, num_pts)
    pred_u = pred_y[num_l:].view(num_q // 2, num_pts)
    independence_loss = independence_penalty(y, pred_l, pred_u, pearsons_corr_multiplier, hsic_multiplier)

    loss = pinball_loss + independence_loss

    return loss



def batch_interval_loss(model, y, x, q_list, device, args):
    """
    implementation of interval score, for batch of quantiles
    """
    num_pts = y.size(0)
    num_q = q_list.size(0)

    with torch.no_grad():
        l_list = torch.min(torch.stack([q_list, 1 - q_list], dim=1), dim=1)[0].to(device)
        u_list = 1.0 - l_list

    l_rep = l_list.view(-1, 1).repeat(1, num_pts).view(-1, 1)
    u_rep = u_list.view(-1, 1).repeat(1, num_pts).view(-1, 1)
    num_l = l_rep.size(0)

    if x is None:
        model_in = torch.cat([l_list, u_list], dim=0)
    else:
        x_stacked = x.repeat(num_q, 1)
        l_in = torch.cat([x_stacked, l_rep], dim=1)
        u_in = torch.cat([x_stacked, u_rep], dim=1)
        model_in = torch.cat([l_in, u_in], dim=0)

    pred_y = model(model_in)
    pred_l = pred_y[:num_l].view(num_q, num_pts)
    pred_u = pred_y[num_l:].view(num_q, num_pts)

    below_l = (pred_l - y.view(-1)).gt(0)
    above_u = (y.view(-1) - pred_u).gt(0)

    interval_score_loss = (pred_u - pred_l) + \
                          (1.0 / l_list).view(-1, 1).to(device) * (pred_l - y.view(-1)) * below_l + \
                          (1.0 / l_list).view(-1, 1).to(device) * (y.view(-1) - pred_u) * above_u

    interval_score_loss = torch.mean(interval_score_loss)

    pearsons_corr_multiplier = args.corr_mult
    hsic_multiplier = args.hsic_mult * 10
    independence_loss = independence_penalty(y, pred_l, pred_u, pearsons_corr_multiplier, hsic_multiplier)

    loss = interval_score_loss + independence_loss

    return loss
