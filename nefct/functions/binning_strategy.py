import numpy as np

pertile = np.percentile


def binning_strategy(v_data: np.ndarray, f_data: np.ndarray, nbin: int):
    bin_gap = 95 / nbin
    bin_inds = [0] * nbin
    bin_inds[0] = np.where(v_data > pertile(v_data, 100 - bin_gap))[0]
    for i in range(1, nbin // 2):
        bin_inds[i] = np.where((v_data > pertile(v_data, 100 - bin_gap * (i * 2 + 1))) &
                               (v_data <= pertile(v_data, 100 - bin_gap * (i * 2 - 1))) &
                               (f_data <= 0))[0]
    bin_inds[nbin // 2] = \
        np.where((v_data <= pertile(v_data, 5 + bin_gap)) & (v_data >= pertile(v_data, 5)))[0]
    for i in range(nbin // 2 + 1, nbin):
        i2 = nbin - i
        bin_inds[i] = np.where((v_data > pertile(v_data, 100 - bin_gap * (i2 * 2 + 1))) &
                               (v_data <= pertile(v_data, 100 - bin_gap * (i2 * 2 - 1))) &
                               (f_data > 0))[0]

    total_bin_inds = np.zeros(0, dtype = np.int32)
    v0_data = np.zeros(nbin, dtype = np.float32)
    f0_data = np.zeros(nbin, dtype = np.float32)
    num_in_bin = np.zeros(nbin + 1, dtype = np.int32)
    for i in range(nbin):
        total_bin_inds = np.hstack((total_bin_inds, bin_inds[i]))
        v0_data[i] = np.mean(v_data[bin_inds[i]])
        f0_data[i] = np.mean(f_data[bin_inds[i]])
        num_in_bin[(i + 1):] += bin_inds[i].size

    return total_bin_inds, v0_data, f0_data, num_in_bin
