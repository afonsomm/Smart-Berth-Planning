import numpy as np
from scipy.linalg import sqrtm

class KalmanFilter:

    def __init__(self, x, F, B, H, Q, R, P):
        self.x = x
        self.nx = self.x.shape[0]

        self.F = F
        self.B = B
        self.H = H

        self.Q = Q
        self.R = R
        self.P = P

        self.y = None

    def predict_x(self, u):
        if self.B is not None and u is not None:
            return np.dot(self.F, self.x) + np.dot(self.B, u)

        return np.dot(self.F, self.x)

    def predict_z(self):
        return np.dot(self.H, self.x)

    def predict(self, u=None):
        self.x = self.predict_x(u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        self.y = z - self.predict_z()
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x += np.dot(K, self.y)
        I = np.eye(self.nx) - np.dot(K, self.H)
        self.P = np.dot(np.dot(I, self.P), I.T) + np.dot(np.dot(K, self.R), K.T)


class ExtendedKalmanFilter:

    def __init__(self, x, F, B, Jf, H, Jh, Q, R, P):
        self.x = x
        self.nx = self.x.shape[0]

        self.F = F
        self.B = B
        self.Jf = Jf
        self.H = H
        self.Jh = Jh

        self.Q = Q
        self.R = R
        self.P = P

        self.y = None

    def predict_x(self, u):
        if self.B is not None and u is not None:
            return np.dot(self.F, self.x) + np.dot(self.B, u)

        return np.dot(self.F, self.x)

    def predict_z(self):
        return np.dot(self.H, self.x)

    def predict(self, u=None):
        self.x = self.predict_x(u)
        self.P = np.dot(np.dot(self.Jf, self.P), self.Jf.T) + self.Q

    def update(self, z):
        self.y = z - self.predict_z()
        S = np.dot(np.dot(self.Jh, self.P), self.Jh.T) + self.R
        K = np.dot(np.dot(self.P, self.Jh.T), np.linalg.inv(S))

        self.x += np.dot(K, self.y)
        I = np.eye(self.nx) - np.dot(K, self.Jh)
        self.P = np.dot(np.dot(I, self.P), I.T) + np.dot(np.dot(K, self.R), K.T)


class UnscentedKalmanFilter:

    def __init__(self, x, F, B, H, Q, R, P, alpha, beta, kappa):

        self.x = x
        self.nx = self.x.shape[0]
        self.sigma_x = None
        self.sigma_z = None

        self.F = F
        self.B = B
        self.H = H

        self.Q = Q
        self.R = R
        self.P = P

        self.y = None

        self.wm, self.wc, self.gamma = self.setup_ukf(self.nx, alpha, beta, kappa)

    def setup_ukf(self, nx, alpha, beta, kappa):
        # calculate lambda
        lamb = alpha ** 2 * (nx + kappa) - nx
        # calculate the weights
        # w^(0)
        wm = [lamb / (lamb + nx)]  # wm corresponds to w of the UKF Algorithm
        # wc^(0)
        wc = [(lamb / (lamb + nx)) + (1 - alpha ** 2 + beta)]
        for i in range(2 * nx):
            # w^(+-i)
            wm.append(1.0 / (2 * (nx + lamb)))
            # wc^(+-i)
            wc.append(1.0 / (2 * (nx + lamb)))

        # define gamma
        gamma = np.sqrt(nx + lamb)

        wm = np.array([wm])
        wc = np.array([wc])

        return wm, wc, gamma

    def predict_z(self):
        return np.dot(self.H, self.x)

    def get_sigma_points(self):

        Psqrt = sqrtm(self.P)

        sigma_plus = self.x + self.gamma * Psqrt
        sigma_minus = self.x - self.gamma * Psqrt

        self.sigma_x = np.hstack((self.x, sigma_minus, sigma_plus))

    def predict_sigma(self, sigma, u):

        if self.B is not None and u is not None:
            return np.dot(self.F, sigma) + np.dot(self.B, u)

        return np.dot(self.F, sigma)

    def propagate_sigma_points(self, u=None):
        self.sigma_x = np.apply_along_axis(self.predict_sigma, axis=0, arr=self.sigma_x, u=u)

    def obs_sigma(self, sigma):
        return np.dot(self.H, sigma)

    def predict_sigma_observation(self):
        self.sigma_z = np.apply_along_axis(self.obs_sigma, axis=0, arr=self.sigma_x)

    def predict_sigma_x_covariance(self):

        d = self.sigma_x - self.x

        def calc_covariance(sigma):
            return np.dot(sigma[:, np.newaxis], sigma[:, np.newaxis].T)

        cov = np.apply_along_axis(calc_covariance, axis=0, arr=d).T
        cov *= self.wc.T[:, :, np.newaxis]
        self.P = self.Q + np.sum(cov, axis=0)

    def predict_sigma_z_covariance(self, zb):

        d = self.sigma_z - zb

        def calc_covariance(d):
            return np.dot(d[:, np.newaxis], d[:, np.newaxis].T)

        cov = np.apply_along_axis(calc_covariance, axis=0, arr=d).T
        cov *= self.wc.T[:, :, np.newaxis]
        return self.R + np.sum(cov, axis=0)

    def calc_pxz(self, zb):

        d = self.sigma_x - self.x
        dz = self.sigma_z - zb
        d = np.vstack((d, dz))

        def calc_covariance(d):
            return np.dot(d[:self.nx, np.newaxis], d[self.nx:, np.newaxis].T)

        cov = np.apply_along_axis(calc_covariance, axis=0, arr=d).T
        cov *= self.wc.T[:, :, np.newaxis]

        return np.sum(cov, axis=0).T

    def predict(self, u=None):

        self.get_sigma_points()
        self.propagate_sigma_points(u)

        self.x = np.dot(self.wm, self.sigma_x.T).T
        self.predict_sigma_x_covariance()

    def update(self, z):

        self.y = z - self.predict_z()
        self.get_sigma_points()
        zb = np.dot(self.wm, self.sigma_x.T).T[[0, 2]]

        self.predict_sigma_observation()

        S = self.predict_sigma_z_covariance(zb)
        Pxz = self.calc_pxz(zb)
        K = np.dot(Pxz, np.linalg.inv(S))

        self.x += np.dot(K, self.y)
        I = np.eye(self.nx) - np.dot(K, self.H)
        self.P = np.dot(np.dot(I, self.P), I.T) + np.dot(np.dot(K, self.R), K.T)