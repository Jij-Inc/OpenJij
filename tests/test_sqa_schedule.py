import itertools
import unittest

import numpy as np

import openjij as oj
import openjij.cxxjij as cxxjij
from openjij.sampler.sa_sampler import geometric_ising_beta_schedule
from openjij.sampler.sqa_sampler import (
    geometric_sqa_schedule,
    trotter_coupling_range,
)


def _trotter_coupling(beta, s, gamma, trotter):
    return -0.5 * np.log(np.tanh(beta * gamma * (1 - s) / trotter))


class TestSQASchedule(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.h = {i: float(rng.normal()) for i in range(12)}
        self.J = {(i, j): float(rng.normal()) for i in range(12) for j in range(i + 1, 12) if rng.random() < 0.4}
        self.bqm = oj.model.model.BinaryQuadraticModel(self.h, self.J, 0.0, "SPIN", sparse=True)
        self.graph, _ = self.bqm.get_cxxjij_ising_graph()

    def test_schedule_follows_sa_beta_and_coupling(self):
        trotter, num_sweeps, gamma = 6, 200, 2.5
        schedule, info = geometric_sqa_schedule(
            self.graph, trotter, num_sweeps, gamma=gamma, coupling_min=0.03, coupling_max=0.7
        )
        classical, beta_range = geometric_ising_beta_schedule(self.graph, num_sweeps=num_sweeps)

        beta = np.array([x.updater_parameter.beta for x in schedule])
        s = np.array([x.updater_parameter.s for x in schedule])
        steps = [x.one_mc_step for x in schedule]
        self.assertEqual(sum(steps), num_sweeps)
        self.assertTrue(np.all((0 <= s) & (s <= 1)))

        # classical inverse temperature of a slice is the SA schedule
        np.testing.assert_allclose(s * beta / trotter, [x.updater_parameter.beta for x in classical], rtol=1e-9)
        # the Trotter coupling grows geometrically between the endpoints
        coupling = _trotter_coupling(beta, s, gamma, trotter)
        self.assertAlmostEqual(coupling[0], 0.03, places=6)
        self.assertAlmostEqual(coupling[-1], 0.7, places=6)
        self.assertTrue(np.all(np.diff(coupling) > 0))
        self.assertEqual(info["beta_min"], beta_range[1])
        self.assertEqual(info["beta_max"], beta_range[0])
        self.assertNotIn("dominance", info)

    def test_default_coupling_from_landscape(self):
        schedule, info = geometric_sqa_schedule(self.graph, 4, 100)
        (coupling_min, coupling_max), stats = trotter_coupling_range(self.graph)
        self.assertEqual((info["coupling_min"], info["coupling_max"]), (coupling_min, coupling_max))
        self.assertEqual(info["dominance"], stats["dominance"])
        self.assertEqual(info["pair_move_fraction"], stats["pair_move_fraction"])
        self.assertEqual(info["zero_mode_fraction"], stats["zero_mode_fraction"])
        beta = np.array([x.updater_parameter.beta for x in schedule])
        s = np.array([x.updater_parameter.s for x in schedule])
        coupling = _trotter_coupling(beta, s, 1.0, 4)
        self.assertAlmostEqual(coupling[0], coupling_min, places=6)
        self.assertAlmostEqual(coupling[-1], coupling_max, places=6)
        # one end can be fixed while the other is estimated
        _, info = geometric_sqa_schedule(self.graph, 4, 100, coupling_max=3.0)
        self.assertEqual((info["coupling_min"], info["coupling_max"]), (coupling_min, 3.0))

    def test_sampler_uses_default_schedule(self):
        res = oj.SQASampler().sample(self.bqm, num_sweeps=50, num_reads=3, seed=1)
        info = res.info["schedule"]
        self.assertEqual(info["trotter"], 8)
        (coupling_min, coupling_max), _ = trotter_coupling_range(self.graph)
        self.assertEqual(info["coupling_min"], coupling_min)
        self.assertEqual(info["coupling_max"], coupling_max)

        res = oj.SQASampler().sample(
            self.bqm, num_sweeps=50, beta_min=0.1, beta_max=7.0, coupling_min=0.05, coupling_max=1.5
        )
        info = res.info["schedule"]
        self.assertEqual((info["beta_min"], info["beta_max"]), (0.1, 7.0))
        self.assertEqual((info["coupling_min"], info["coupling_max"]), (0.05, 1.5))

    def test_legacy_schedule_with_beta_or_gamma(self):
        res = oj.SQASampler().sample(self.bqm, beta=3.0, num_sweeps=20)
        self.assertEqual(res.info["schedule"], {"beta": 3.0, "gamma": 1.0, "num_sweeps": 20})
        res = oj.SQASampler().sample(self.bqm, gamma=2.0, num_sweeps=20)
        self.assertEqual(res.info["schedule"], {"beta": 5.0, "gamma": 2.0, "num_sweeps": 20})
        with self.assertRaises(ValueError):
            oj.SQASampler().sample(self.bqm, beta=3.0, beta_min=0.1)
        with self.assertRaises(ValueError):
            oj.SQASampler().sample(self.bqm, gamma=2.0, coupling_max=1.0)

    def test_seed_reproducible_and_independent_of_threads(self):
        sampler = oj.SQASampler()
        kwargs = dict(num_sweeps=30, num_reads=7, seed=12345)
        a = sampler.sample(self.bqm, num_threads=1, **kwargs)
        b = sampler.sample(self.bqm, num_threads=3, **kwargs)
        np.testing.assert_array_equal(a.record.sample, b.record.sample)
        np.testing.assert_array_equal(a.record.energy, b.record.energy)
        for x, y in zip(a.info["system"], b.info["system"]):
            np.testing.assert_array_equal(x["trotter_state"], y["trotter_state"])
        # every read has its own seed
        states = {tuple(map(tuple, x["trotter_state"])) for x in a.info["system"]}
        self.assertGreater(len(states), 1)

    def test_sequential_reads_without_reinitialization(self):
        sampler = oj.SQASampler()
        init = {i: 1 for i in self.h}
        res = sampler.sample(
            self.bqm, num_sweeps=10, num_reads=3, initial_state=init, reinitialize_state=False, seed=3
        )
        self.assertEqual(len(res.record), 3)

    def test_invalid_arguments(self):
        sampler = oj.SQASampler()
        with self.assertRaises(ValueError):
            sampler.sample(self.bqm, num_threads=0)
        with self.assertRaises(ValueError):
            sampler.sample(self.bqm, trotter=1)
        with self.assertRaises(ValueError):
            sampler.sample(self.bqm, updater="unknown")
        with self.assertRaises(ValueError):
            geometric_sqa_schedule(self.graph, 4, coupling_min=0.0)
        with self.assertRaises(ValueError):
            geometric_sqa_schedule(self.graph, 4, coupling_max=np.inf)

    def test_trotter_cluster_flip_finds_ground_state(self):
        h = {0: -1, 1: -1, 2: 1, 3: 1}
        J = {(0, 1): -1, (3, 4): -1}
        res = oj.SQASampler().sample_ising(h, J, updater="trotter cluster flip", num_reads=5, seed=2)
        self.assertEqual(res.first.energy, -6.0)


def _graph(h, J, vartype="SPIN", sparse=True):
    bqm = oj.model.model.BinaryQuadraticModel(h, J, 0.0, vartype, sparse=sparse)
    return bqm.get_cxxjij_ising_graph()[0]


class TestTrotterCouplingRange(unittest.TestCase):
    """The coupling range follows the structure of the energy landscape."""

    def test_spin_glass_keeps_replicas_weakly_coupled(self):
        rng = np.random.default_rng(1)
        n = 60
        J = {(i, j): float(rng.normal()) / np.sqrt(n) for i in range(n) for j in range(i + 1, n)}
        (coupling_min, coupling_max), stats = trotter_coupling_range(_graph({}, J))
        self.assertLess(stats["dominance"], 0.4)
        self.assertLess(stats["pair_move_fraction"], 0.3)
        self.assertEqual((coupling_min, coupling_max), (0.01, 0.5))

    def test_dominant_fields_couple_replicas_early(self):
        rng = np.random.default_rng(2)
        n = 40
        h = {i: float(rng.choice([-3.0, 3.0])) for i in range(n)}
        J = {(i, (i + 1) % n): float(rng.normal()) * 0.3 for i in range(n)}
        (coupling_min, _), stats = trotter_coupling_range(_graph(h, J))
        self.assertGreater(stats["dominance"], 0.9)
        self.assertAlmostEqual(coupling_min, 0.2)

    def test_global_constraint_couples_replicas_at_the_end(self):
        # balanced bisection: a single move breaks the balance, a swap keeps it
        n = 30
        J = {(i, j): 1.0 for i in range(n) for j in range(i + 1, n)}
        for i in range(n):
            J[(i, (i + 1) % n) if i + 1 < n else (0, n - 1)] -= 0.2
        (_, coupling_max), stats = trotter_coupling_range(_graph({}, J))
        self.assertGreater(stats["pair_move_fraction"], 0.8)
        self.assertAlmostEqual(coupling_max, 2.0)

    def test_degenerate_lattice_weakens_coupling(self):
        # 2D +-1 spin glass: many spins of low-energy states flip at no cost
        rng = np.random.default_rng(4)
        L = 16
        J = {}
        for x in range(L):
            for y in range(L):
                J[(x * L + y, ((x + 1) % L) * L + y)] = float(rng.choice([-1.0, 1.0]))
                J[(x * L + y, x * L + (y + 1) % L)] = float(rng.choice([-1.0, 1.0]))
        (coupling_min, coupling_max), stats = trotter_coupling_range(_graph({}, J))
        self.assertGreater(stats["zero_mode_fraction"], 0.03)
        self.assertAlmostEqual(coupling_max, 0.05)
        self.assertLess(coupling_min, 0.01)

    def test_deterministic_and_storage_independent(self):
        rng = np.random.default_rng(3)
        h = {i: float(rng.normal()) for i in range(20)}
        J = {(i, j): float(rng.normal()) for i in range(20) for j in range(i + 1, 20) if rng.random() < 0.3}
        sparse = trotter_coupling_range(_graph(h, J))
        self.assertEqual(sparse, trotter_coupling_range(_graph(h, J)))
        dense = trotter_coupling_range(_graph(h, J, sparse=False))
        self.assertAlmostEqual(sparse[1]["dominance"], dense[1]["dominance"])

    def test_degenerate_problems(self):
        # fields only: no pair moves
        (_, coupling_max), stats = trotter_coupling_range(_graph({0: 1.0, 1: -2.0}, {}))
        self.assertEqual(stats["pair_move_fraction"], 0.0)
        self.assertEqual(coupling_max, 0.5)
        # every coefficient zero
        (coupling_min, coupling_max), _ = trotter_coupling_range(_graph({0: 0.0, 1: 0.0}, {(0, 1): 0.0}))
        self.assertTrue(np.isfinite(coupling_min) and np.isfinite(coupling_max))
        # single spin
        res = oj.SQASampler().sample_ising({0: -1.0}, {}, num_reads=2, seed=0)
        self.assertEqual(res.first.energy, -1.0)


class TestTransverseIsingUpdaters(unittest.TestCase):
    """Both updaters sample the Trotterized Boltzmann distribution."""

    def setUp(self):
        self.h = {0: 0.3, 1: -0.2, 2: 0.1}
        self.J = {(0, 1): -1.0, (1, 2): 0.7, (0, 2): 0.4}
        bqm = oj.model.model.BinaryQuadraticModel(self.h, self.J, 0.0, "SPIN", sparse=True)
        self.graph, _ = bqm.get_cxxjij_ising_graph()

    def _exact(self, beta, s, trotter):
        n = 3
        states = np.array(list(itertools.product([-1, 1], repeat=n * trotter))).reshape(-1, trotter, n)
        energy = states @ np.array([self.h[i] for i in range(n)])
        for (i, j), v in self.J.items():
            energy = energy + v * states[:, :, i] * states[:, :, j]
        classical = s * beta / trotter
        coupling = -0.5 * np.log(np.tanh(beta * (1 - s) / trotter))
        weight = -classical * energy.sum(1) + coupling * (states * np.roll(states, 1, axis=1)).sum((1, 2))
        weight = np.exp(weight - weight.max())
        weight /= weight.sum()
        corr_space = (states[:, :, 0] * states[:, :, 1]).mean(1)
        corr_time = (states[:, 0, :] * states[:, 1, :]).mean(1)
        return weight @ corr_space, weight @ corr_time

    def _sample(self, updater, beta, s, trotter, num_systems=3000, num_sweeps=60):
        rng = np.random.default_rng(0)
        systems = [
            cxxjij.system.make_transverse_ising(rng.choice([-1, 1], (trotter, 3)).tolist(), self.graph, 1.0)
            for _ in range(num_systems)
        ]
        run = getattr(cxxjij.algorithm, f"Algorithm_{updater}_run_parallel")
        run(systems, list(range(num_systems)), [((beta, s), num_sweeps)])
        spins = np.array([np.asarray(x.trotter_spins)[:-1].T for x in systems])
        return (spins[:, :, 0] * spins[:, :, 1]).mean(), (spins[:, 0, :] * spins[:, 1, :]).mean()

    def test_equilibrium(self):
        for updater in ["SingleSpinFlip", "TrotterClusterFlip"]:
            for beta, s in [(2.0, 0.5), (4.0, 0.8)]:
                with self.subTest(updater=updater, beta=beta, s=s):
                    exact = self._exact(beta, s, 4)
                    np.testing.assert_allclose(self._sample(updater, beta, s, 4), exact, atol=0.04)

    def test_cluster_flip_at_s_equal_one(self):
        # J_perp is infinite: replicas move as a whole and stay aligned.
        exact = self._exact(3.0, 1 - 1e-12, 4)
        np.testing.assert_allclose(self._sample("TrotterClusterFlip", 3.0, 1.0, 4), exact, atol=0.04)

    def test_single_spin_flip_does_not_freeze_at_s_equal_one(self):
        # A copy between disagreeing neighbours has no Trotter cost at s = 1;
        # it must be allowed to flip so that the slices can align.
        _, corr_time = self._sample("SingleSpinFlip", 3.0, 1.0, 4, num_systems=200, num_sweeps=200)
        self.assertGreater(corr_time, 0.99)


if __name__ == "__main__":
    unittest.main()
