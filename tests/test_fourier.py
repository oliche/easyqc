import unittest
import numpy as np
import scipy.signal

import dsp.fourier as ft
from dsp.utils import fcn_cosine


class TestDspMisc(unittest.TestCase):

    def test_dsp_cosine_func(self):
        x = np.linspace(0, 40)
        fcn = fcn_cosine(bounds=[20, 30])
        y = fcn(x)
        self.assertTrue(y[0] == 0 and y[-1] == 1 and np.all(np.diff(y) >= 0))


class TestPhaseRegression(unittest.TestCase):

    def test_fit_phase1d(self):
        w = np.zeros(500)
        w[1] = 1
        self.assertTrue(np.isclose(ft.fit_phase(w, .002), .002))

    def test_fit_phase2d(self):
        w = np.zeros((500, 2))
        w[1, 0], w[2, 1] = (1, 1)
        self.assertTrue(np.all(np.isclose(ft.fit_phase(w, .002, axis=0), np.array([.002, .004]))))
        self.assertTrue(np.all(np.isclose(
            ft.fit_phase(w.transpose(), .002), np.array([.002, .004]))))


class TestShift(unittest.TestCase):

    def test_shift_1d(self):
        ns = 500
        w = scipy.signal.ricker(ns, 10)
        self.assertTrue(np.all(np.isclose(ft.shift(w, 1), np.roll(w, 1))))

    def test_shift_2d(self):
        ns = 500
        w = scipy.signal.ricker(ns, 10)
        w = np.tile(w, (100, 1)).transpose()
        self.assertTrue(np.all(np.isclose(ft.shift(w, 1, axis=0), np.roll(w, 1, axis=0))))
        self.assertTrue(np.all(np.isclose(ft.shift(w, 1, axis=1), np.roll(w, 1, axis=1))))


class TestFFT(unittest.TestCase):

    def test_spectral_convolution(self):
        sig = np.random.randn(20, 500)
        w = np.hanning(25)
        c = ft.convolve(sig, w)
        s = np.convolve(sig[0, :], w)
        self.assertTrue(np.all(np.isclose(s, c[0, :-1])))

        c = ft.convolve(sig, w, mode='same')
        s = np.convolve(sig[0, :], w, mode='same')
        self.assertTrue(np.all(np.isclose(c[0, :], s)))

        c = ft.convolve(sig, w[:-1], mode='same')
        s = np.convolve(sig[0, :], w[:-1], mode='same')
        self.assertTrue(np.all(np.isclose(c[0, :], s)))

    def test_nech_optim(self):
        self.assertTrue(ft.ns_optim_fft(2048) == 2048)
        self.assertTrue(ft.ns_optim_fft(65532) == 65536)

    def test_freduce(self):
        # test with 1D arrays
        fs = np.fft.fftfreq(5)
        self.assertTrue(np.all(ft.freduce(fs) == fs[:-2]))
        fs = np.fft.fftfreq(6)
        self.assertTrue(np.all(ft.freduce(fs) == fs[:-2]))

        # test 2D arrays along both dimensions
        fs = np.tile(ft.fscale(500, 0.001), (4, 1))
        self.assertTrue(ft.freduce(fs).shape == (4, 251))
        self.assertTrue(ft.freduce(np.transpose(fs), axis=0).shape == (251, 4))

    def test_fexpand(self):
        # test odd input
        res = np.random.rand(11)
        X = ft.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(ft.fexpand(X, 11)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test even input
        res = np.random.rand(12)
        X = ft.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(ft.fexpand(X, 12)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test with a 2 dimensional input along last dimension
        res = np.random.rand(2, 12)
        X = ft.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(ft.fexpand(X, 12)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test with a 3 dimensional input along last dimension
        res = np.random.rand(3, 5, 12)
        X = ft.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(ft.fexpand(X, 12)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test with 2 dimensional input along first dimension
        fs = np.transpose(np.tile(ft.fscale(500, 0.001, one_sided=True), (4, 1)))
        self.assertTrue(ft.fexpand(fs, 500, axis=0).shape == (500, 4))

    def test_fscale(self):
        # test for an even number of samples
        res = [0, 100, 200, 300, 400, 500, -400, -300, -200, -100],
        self.assertTrue(np.all(np.abs(ft.fscale(10, 0.001) - res) < 1e-6))
        # test for an odd number of samples
        res = [0, 90.9090909090909, 181.818181818182, 272.727272727273, 363.636363636364,
               454.545454545455, -454.545454545455, -363.636363636364, -272.727272727273,
               -181.818181818182, -90.9090909090909],
        self.assertTrue(np.all(np.abs(ft.fscale(11, 0.001) - res) < 1e-6))

    def test_filter_lp_hp(self):
        # test 1D time serie: subtracting lp filter removes DC
        ts1 = np.random.rand(500)
        out1 = ft.lp(ts1, 1, [.1, .2])
        self.assertTrue(np.mean(ts1 - out1) < 0.001)
        # test 2D case along the last dimension
        ts = np.tile(ts1, (11, 1))
        out = ft.lp(ts, 1, [.1, .2])
        self.assertTrue(np.allclose(out, out1))
        # test 2D case along the first dimension
        ts = np.tile(ts1[:, np.newaxis], (1, 11))
        out = ft.lp(ts, 1, [.1, .2], axis=0)
        self.assertTrue(np.allclose(np.transpose(out), out1))
        # test 1D time serie: subtracting lp filter removes DC
        out2 = ft.hp(ts1, 1, [.1, .2])
        self.assertTrue(np.allclose(out1, ts1 - out2))

    def test_dft(self):
        # test 1D complex
        x = np.array([1, 2 - 1j, -1j, -1 + 2j])
        X = ft.dft(x)
        assert np.all(np.isclose(X, np.fft.fft(x)))
        # test 1D real
        x = np.random.randn(7)
        X = ft.dft(x)
        assert np.all(np.isclose(X, np.fft.rfft(x)))
        # test along the 3 dimensions of a 3D array
        x = np.random.rand(10, 11, 12)
        for axis in np.arange(3):
            X_ = np.fft.rfft(x, axis=axis)
            assert np.all(np.isclose(X_, ft.dft(x, axis=axis)))
        # test 2D irregular grid
        _n0, _n1, nt = (10, 11, 30)
        x = np.random.rand(_n0 * _n1, nt)
        X_ = np.fft.fft(np.fft.fft(x.reshape(_n0, _n1, nt), axis=0), axis=1)
        r, c = [v.flatten() for v in np.meshgrid(np.arange(
            _n0) / _n0, np.arange(_n1) / _n1, indexing='ij')]
        nk, nl = (_n0, _n1)
        X = ft.dft2(x, r, c, nk, nl)
        assert np.all(np.isclose(X, X_))
