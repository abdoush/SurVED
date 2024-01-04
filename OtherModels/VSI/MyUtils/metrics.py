import numpy as np


def c_index_decomposition(t, y, e, debug=False):
    '''
    Args:
        t: t true
        y: t pred (the lower the riskier)
        e: event
        debug:

    Returns: cindex decomp
    '''

    e = e.astype(int)
    ete = (e[np.newaxis, :] + e[:, np.newaxis])

    is_ee_pair = (ete == 2)
    is_ec_pair = (ete == 1)

    tdiff = t[np.newaxis, :] - t[:, np.newaxis]
    tij = (tdiff > 0) + (tdiff == 0) * is_ec_pair

    is_valid_pair = (tij * e[:, np.newaxis]).astype(bool)

    ydiff = (y[np.newaxis, :] - y[:, np.newaxis])
    yij_greater = (ydiff > 0)
    yij_tieds = (ydiff == 0)

    num_pairs_ee = np.sum(is_valid_pair * is_ee_pair)
    num_pairs_ec = np.sum(is_valid_pair * is_ec_pair)
    num_pairs = num_pairs_ee + num_pairs_ec

    correctij = (yij_greater * is_valid_pair)
    num_correct_ee = np.sum(correctij * is_ee_pair)
    num_correct_ec = np.sum(correctij * is_ec_pair)

    tiedij = (yij_tieds * is_valid_pair)
    num_tied_ee = np.sum(tiedij * is_ee_pair)
    num_tied_ec = np.sum(tiedij * is_ec_pair)

    # ======================================================================
    pp_ee = num_correct_ee
    pn_ee = num_pairs_ee - num_correct_ee - num_tied_ee
    pp_ec = num_correct_ec
    pn_ec = num_pairs_ec - num_correct_ec - num_tied_ec

    alpha = (pp_ee + 0.5 * num_tied_ee) / (pp_ee + pp_ec + 0.5 * (num_tied_ee + num_tied_ec))
    if num_pairs_ee > 0:
        c_ee = (pp_ee + 0.5 * num_tied_ee) / num_pairs_ee
    else:
        c_ee = 0

    if num_pairs_ec > 0:
        c_ec = (pp_ec + 0.5 * num_tied_ec) / num_pairs_ec
    else:
        c_ec = 0

    if c_ee == 0:
        c = c_ec
    elif c_ec == 0:
        c = c_ee
    else:
        c = (c_ee * c_ec) / ((alpha * c_ec) + ((1 - alpha) * c_ee))

    alpha_opt = num_pairs_ee / num_pairs
    alpha_deviation = alpha - alpha_opt

    if debug:
        return pp_ee, pn_ee, pp_ec, pn_ec, num_tied_ee, num_tied_ec, c_ee, c_ec, alpha, alpha_deviation, c
    else:
        return c_ee, c_ec, alpha, alpha_deviation, c
