import numpy as np
from scipy.stats import multivariate_normal as mvn


def explicit_kalman_smoothing(ms, Ps, Fs, Qs, bs):
    """ Explicit Kalman smoother implementation for testing purposes.
    """
    T, _ = ms.shape
    ms_smoothed = np.zeros_like(ms)
    Ps_smoothed = np.zeros_like(Ps)
    ms_smoothed[-1] = ms[-1]
    Ps_smoothed[-1] = Ps[-1]
    for t in range(T - 2, -1, -1):
        m, P, F, Q, b = ms[t], Ps[t], Fs[t], Qs[t], bs[t]
        m_next = ms_smoothed[t + 1]
        P_next = Ps_smoothed[t + 1]

        P_pred = F @ P @ F.T + Q
        m_pred = F @ m + b

        K = P @ F.T @ np.linalg.inv(P_pred)
        ms_smoothed[t] = m + K @ (m_next - m_pred)
        Ps_smoothed[t] = P + K @ (P_next - P_pred) @ K.T
    return ms_smoothed, Ps_smoothed


def explicit_kalman_filter(ys, m0, P0, Hs, Rs, cs, Fs, Qs, bs):
    """ Explicit Kalman filter implementation for testing purposes.
    Computes the marginal likelihood and the marginals of the state for a model
    X_0 ~ N(m0, P0)
    X_t = F_{t-1} X_{t-1} + b_{t-1} + N(0, Q_{t-1}), t = 1, ..., T
    Y_t = H_t X_t + c_t + N(0, R_t), t = 0, ..., T
    """
    T = len(ys)
    dx, dy = Hs.shape[2], Hs.shape[1]

    # Initialize
    ms = np.zeros((T, dx))
    Ps = np.zeros((T, dx, dx))

    # Initial update
    S0 = Hs[0] @ P0 @ Hs[0].T + Rs[0]
    y0_hat = Hs[0] @ m0 + cs[0]
    ell0 = mvn.logpdf(ys[0], y0_hat, S0)
    K0 = P0 @ Hs[0].T @ np.linalg.inv(S0)
    m0 = m0 + K0 @ (ys[0] - y0_hat)
    P0 = P0 - K0 @ S0 @ K0.T

    ms[0] = m0
    Ps[0] = P0
    ell = ell0

    for t in range(1, T):
        # Prediction
        m = Fs[t-1] @ ms[t - 1] + bs[t-1]
        P = Fs[t-1] @ Ps[t - 1] @ Fs[t-1].T + Qs[t-1]

        # Update
        S = Hs[t] @ P @ Hs[t].T + Rs[t]
        y_hat = Hs[t] @ m + cs[t]
        ell += mvn.logpdf(ys[t], y_hat, S)
        K = P @ Hs[t].T @ np.linalg.inv(S)
        ms[t] = m + K @ (ys[t] - y_hat)
        Ps[t] = P - K @ Hs[t] @ P

    return ms, Ps, ell
