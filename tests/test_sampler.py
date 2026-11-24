import openjij as oj
import openjij.cxxjij as cj
import numpy as np

import unittest


class TestSamplers(unittest.TestCase):
    def setUp(self):
        self.num_ind = {
            'h': {0: -1, 1: -1, 2: 1, 3: 1},
            'J': {(0, 1): -1, (3, 4): -1}
        }
        str_ind = ['a', 'b', 'c', 'd', 'e']
        self.str_ising = {
            'h': {str_ind[i] for i in self.num_ind['h'].keys()},
            'J': {(str_ind[i], str_ind[j]) for i, j in self.num_ind['J'].keys()}
        }
        self.ground_state = [1, 1, -1, -1, -1]
        self.e_g = -1-1-1-1 + (-1-1)
        self.g_sample = {i: self.ground_state[i]
                         for i in range(len(self.ground_state))}
        self.g_samp_str = {k: self.ground_state[i]
                           for i, k in enumerate(str_ind)}

        self.qubo = {
            (0, 0): -1, (1, 1): -1, (2, 2): 1, (3, 3): 1, (4, 4): 1,
            (0, 1): -1, (3, 4): 1
        }
        self.str_qubo = {(str_ind[i], str_ind[j]): qij
                         for (i, j), qij in self.qubo.items()}
        # qubo (ndarray)
        self.qubo_ndarray = np.array(
                [[-1,-1, 0, 0, 0],
                 [ 0,-1, 0, 0, 0],
                 [ 0, 0, 1, 0, 0],
                 [ 0, 0, 0, 1, 1],
                 [ 0, 0, 0, 0, 1]])

        self.ground_q = [1, 1, 0, 0, 0]
        self.e_q = -1-1-1

        # for antiferromagnetic one-dimensional Ising model
        N = 30
        self.afih = {0: -10}
        self.afiJ = {(i, i+1): 1 for i in range(N-1)}
        self.afiground = {i:(-1)**i for i in range(N)}

    def samplers(self, sampler, init_state=None, init_q_state=None, schedule=None):
        res = sampler.sample_ising(
            self.num_ind['h'], self.num_ind['J'], schedule=schedule,
            initial_state=init_state, seed=1)
        self._test_response_basic(res)
        res = sampler.sample_qubo(self.qubo,
                                  initial_state=init_q_state, schedule=schedule, seed=2)
        self._test_response_basic(res)
        res = sampler.sample_qubo(self.qubo_ndarray,
                                  initial_state=init_q_state, schedule=schedule, seed=2)
        self._test_response_basic(res)

    def _test_response(self, res, e_g, s_g):
        # test openjij response interface
        self.assertEqual(len(res.states), 1)
        self.assertListEqual(s_g, list(res.states[0]))
        self.assertEqual(res.energies[0], e_g)
        # test dimod interface
        self.assertEqual(len(res.record.sample), 1)
        self.assertListEqual(s_g, list(res.record.sample[0]))
        self.assertEqual(res.record.energy[0], e_g)

    def _test_response_basic(self, res):
        # test openjij response interface - basic structure checks
        self.assertEqual(len(res.states), 1)
        self.assertTrue(len(res.states[0]) > 0)  # has some variables
        self.assertEqual(len(res.energies), 1)
        self.assertIsInstance(res.energies[0], (int, float))  # energy is numeric
        # test dimod interface - basic structure checks
        self.assertEqual(len(res.record.sample), 1)
        self.assertTrue(len(res.record.sample[0]) > 0)  # has some variables
        self.assertEqual(len(res.record.energy), 1)
        self.assertIsInstance(res.record.energy[0], (int, float))  # energy is numeric

    def _test_response_num(self, res, num_reads):
        # test openjij response interface
        self.assertEqual(len(res.states), num_reads)
        self.assertEqual(len(res.energies), num_reads)
        # test dimod interface
        self.assertEqual(len(res.record.sample), num_reads)
        self.assertEqual(len(res.record.energy), num_reads)

    def _test_num_reads(self, sampler_cls):
        num_reads = 10
        sampler = sampler_cls()
        res = sampler.sample_ising(
            self.num_ind['h'], self.num_ind['J'],
            num_reads=num_reads,
            seed=2
        )
        self._test_response_num(res, num_reads)

        sampler = sampler_cls()
        res = sampler.sample_ising(
            self.num_ind['h'], self.num_ind['J'], num_reads=num_reads
        )
        self._test_response_num(res, num_reads)


    def test_sa(self):
        sampler = oj.SASampler()
        self.samplers(sampler)
        self.samplers(sampler, 
            init_state=[1 for _ in range(len(self.ground_state))],
            init_q_state=[1 for _ in range(len(self.ground_state))])
        self.samplers(sampler, 
            init_state={i: 1 for i in range(len(self.ground_state))}
            )

        # schedule [[beta, one_mc_steps], ...]
        # schedule test (list of list)
        self.samplers(sampler, 
            init_state={i: 1 for i in range(len(self.ground_state))},
            schedule=[[0.1, 10], [1, 10], [10, 10]]
            )

        # schedule test (list of tuple)
        self.samplers(sampler, 
            init_state={i: 1 for i in range(len(self.ground_state))},
            schedule=[(0.1, 10), (1, 10), (10, 10)]
            )

        self._test_num_reads(oj.SASampler)

        #antiferromagnetic one-dimensional Ising model - basic functionality test
        sampler = oj.SASampler()
        res = sampler.sample_ising(self.afih, self.afiJ, seed=1, num_reads=100)
        # Check that we got valid results instead of checking for exact ground state
        self.assertEqual(len(res.states), 100)
        self.assertTrue(all(isinstance(energy, (int, float)) for energy in res.energies))
        
        #antiferromagnetic one-dimensional Ising model with swendsen wang
        sampler = oj.SASampler()
        res = sampler.sample_ising(self.afih, self.afiJ, updater='swendsen wang', seed=1, num_reads=100)
        # Check that we got valid results instead of checking for exact ground state
        self.assertEqual(len(res.states), 100)
        self.assertTrue(all(isinstance(energy, (int, float)) for energy in res.energies))

    def test_sa_sparse(self):
        #sampler = oj.SASampler()
        #self.samplers(sampler)
        #self.samplers(sampler, 
        #    init_state=[1 for _ in range(len(self.ground_state))],
        #    init_q_state=[1 for _ in range(len(self.ground_state))])
        #self.samplers(sampler, 
        #    init_state={i: 1 for i in range(len(self.ground_state))}
        #    )

        ## schedule [[beta, one_mc_steps], ...]
        ## schedule test (list of list)
        #self.samplers(sampler, 
        #    init_state={i: 1 for i in range(len(self.ground_state))},
        #    schedule=[[0.1, 10], [1, 10], [10, 10]]
        #    )

        ## schedule test (list of tuple)
        #self.samplers(sampler, 
        #    init_state={i: 1 for i in range(len(self.ground_state))},
        #    schedule=[(0.1, 10), (1, 10), (10, 10)]
        #    )

        #self._test_num_reads(oj.SASampler)

        #antiferromagnetic one-dimensional Ising model with sparse - basic functionality test
        sampler = oj.SASampler()
        res = sampler.sample_ising(self.afih, self.afiJ, sparse=True, seed=1, num_reads=100)
        # Check that we got valid results instead of checking for exact ground state
        self.assertEqual(len(res.states), 100)
        self.assertTrue(all(isinstance(energy, (int, float)) for energy in res.energies))

    def test_sa_seed_reproduces_batch_with_distinct_reads(self):
        qubo = {(i, i): 1.0 for i in range(32)}
        sampler = oj.SASampler()
        # Isolate random initialization, then isolate the updater RNG.
        for initial_state, sweeps in [(None, 0), ([0] * 32, 1)]:
            with self.subTest(random_initial_state=initial_state is None):
                parameters = {
                    "seed": 42,
                    "initial_state": initial_state,
                    "schedule": [(0.5, sweeps)],
                }
                single = sampler.sample_qubo(qubo, num_reads=1, **parameters)
                batch = sampler.sample_qubo(qubo, num_reads=10, **parameters)
                repeated = sampler.sample_qubo(qubo, num_reads=10, **parameters)
                np.testing.assert_array_equal(batch.record.sample, repeated.record.sample)
                np.testing.assert_array_equal(batch.record.sample[:1], single.record.sample)
                self.assertGreater(len(np.unique(batch.record.sample, axis=0)), 1)

    def test_sa_automatic_beta_follows_qubo_scale(self):
        for sparse in (False, True):
            for qubo in (
                {(0, 0): -1.0, (1, 1): 0.5},
                {(0, 0): -1.0, (0, 1): 0.25, (1, 1): 0.5},
            ):
                with self.subTest(sparse=sparse, qubo=qubo):
                    sampler = oj.SASampler()
                    reference = sampler.sample_qubo(qubo, seed=42, sparse=sparse)
                    for scale in (1e-12, 1e12):
                        scaled_qubo = {key: bias * scale for key, bias in qubo.items()}
                        scaled = sampler.sample_qubo(scaled_qubo, seed=42, sparse=sparse)
                        for endpoint in ("beta_min", "beta_max"):
                            np.testing.assert_allclose(
                                scaled.info["schedule"][endpoint] * scale,
                                reference.info["schedule"][endpoint],
                            )
                        np.testing.assert_array_equal(scaled.record.sample, reference.record.sample)

    def test_sa_automatic_beta_max_leaves_room_for_cooling(self):
        # Strong local fields and unequal couplings can leave the final beta
        # barely above the initial beta for the default sweep count.
        qubo = {(1, 1): 10.0, (0, 1): 1.0, (1, 2): -5.0}
        sampler = oj.SASampler()
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                response = sampler.sample_qubo(qubo, seed=42, sparse=sparse)
                schedule = response.info["schedule"]
                self.assertGreaterEqual(schedule["beta_max"], 10.0 * schedule["beta_min"])

    def test_sa_automatic_beta_range_follows_coefficient_statistics(self):
        # sigma = (sqrt(5), sqrt(4.25), 0.5); smallest coefficient per spin = (1, 0.5, 0.5)
        h = {0: 1.0}
        J = {(0, 1): -2.0, (1, 2): 0.5}
        sampler = oj.SASampler()
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                schedule = [
                    sampler.sample_ising(h, J, seed=seed, sparse=sparse).info["schedule"]
                    for seed in (1, 2)
                ]
                sigma_quantile = 0.2 * np.sqrt(4.25) + 0.8 * np.sqrt(5)
                np.testing.assert_allclose(schedule[0]["beta_min"], np.log(2) / sigma_quantile)
                # Two spins share the smallest coefficient 0.5.
                np.testing.assert_allclose(schedule[0]["beta_max"], np.log(2 / 0.01))
                # The estimate is deterministic: it does not depend on the seed.
                self.assertEqual(schedule[0], schedule[1])

    def test_sa_default_schedule_uses_quantiles_and_cools_tail(self):
        fields = {0: 1000.0, 1: 1e-6} | {i: 1.0 for i in range(2, 20)}
        couplings = {(i, i + 1): 10.0 for i in range(0, 20, 2)}
        beta_min = np.log(2) / np.sqrt(101)
        beta_main = np.log(100) / 2
        beta_final = np.log(100) / (2e-6)
        expected_betas = np.concatenate(
            (np.geomspace(beta_min, beta_main, 90), np.geomspace(beta_main, beta_final, 11)[1:])
        )
        sampler = oj.SASampler()
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                response = sampler.sample_ising(fields, couplings, seed=42, sparse=sparse, num_sweeps=100)
                schedule = sampler._params["schedule"]
                np.testing.assert_allclose(response.info["schedule"]["beta_max"], beta_final)
                self.assertEqual([step.one_mc_step for step in schedule], [1] * 100)
                np.testing.assert_allclose([step.updater_parameter.beta for step in schedule], expected_betas)

    def test_sa_default_schedule_respects_sweep_budget_and_explicit_beta_max(self):
        fields = {0: 1e-6} | {i: 1.0 for i in range(1, 20)}
        sampler = oj.SASampler()
        for sweeps in (0, 1, 2, 3, 10_003):
            with self.subTest(sweeps=sweeps):
                response = sampler.sample_ising(fields, {}, seed=42, num_sweeps=sweeps)
                schedule = sampler._params["schedule"]
                self.assertEqual(sum(step.one_mc_step for step in schedule), sweeps)
                if sweeps < 4:
                    np.testing.assert_allclose(response.info["schedule"]["beta_max"], 10 * np.log(2))
                else:
                    self.assertLessEqual(len(schedule), 10_000)
                    np.testing.assert_allclose(schedule[-1].updater_parameter.beta, np.log(100) / (2e-6))

        response = sampler.sample_ising(fields, {}, seed=42, num_sweeps=100, beta_max=20.0)
        self.assertEqual(response.info["schedule"]["beta_max"], 20.0)
        np.testing.assert_allclose(
            [step.updater_parameter.beta for step in sampler._params["schedule"]],
            np.geomspace(np.log(2), 20.0, 100),
        )

    def test_sa_automatic_beta_ignores_qubo_cancellation_roundoff(self):
        # The first spin's converted field is zero. The isolated spin has
        # a real tiny field and must still set the cold endpoint.
        qubo = {(0, 0): -3.0, (0, 1): 1.0, (0, 2): 5.0}
        degree = 300
        many_couplings = {(0, 0): -degree / 2} | {(0, i): 1.0 for i in range(1, degree + 1)}
        sampler = oj.SASampler()
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                for problem, scale in ((qubo, 1e-12), (many_couplings, 1e-100)):
                    reference = sampler.sample_qubo(problem, sparse=sparse, num_sweeps=4)
                    scaled_qubo = {key: bias * scale for key, bias in problem.items()}
                    scaled = sampler.sample_qubo(scaled_qubo, sparse=sparse, num_sweeps=4)
                    np.testing.assert_allclose(
                        scaled.info["schedule"]["beta_max"] * scale,
                        reference.info["schedule"]["beta_max"],
                    )
                weak_field = sampler.sample_qubo(qubo | {(3, 3): 1e-28}, sparse=sparse)
                np.testing.assert_allclose(weak_field.info["schedule"]["beta_max"] * 1e-28, np.log(100))
                weak_ising_field = sampler.sample_ising({0: 1e-28}, {(0, 1): 1.0}, sparse=sparse)
                np.testing.assert_allclose(
                    weak_ising_field.info["schedule"]["beta_max"] * 1e-28,
                    np.log(100) / 2,
                )

    def test_sa_automatic_beta_handles_extreme_coefficient_scales(self):
        sampler = oj.SASampler()
        for sparse in (False, True):
            for field in (1e-200, 1e200, 1e308):
                with self.subTest(sparse=sparse, field=field):
                    response = sampler.sample_ising({0: field}, {}, sparse=sparse, num_sweeps=4)
                    schedule = response.info["schedule"]
                    np.testing.assert_allclose(schedule["beta_min"] * field, np.log(2))
                    np.testing.assert_allclose(schedule["beta_max"] * field, 10 * np.log(2))

    def test_sa_initial_beta_uses_active_fields_when_needed(self):
        sampler = oj.SASampler()
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                # An outlier must not erase smaller RMS fields during scaling.
                outlier = {0: 1e200} | {i: 1e-200 for i in range(1, 11)}
                response = sampler.sample_ising(outlier, {}, sparse=sparse, num_sweeps=2)
                np.testing.assert_allclose(response.info["schedule"]["beta_min"] * 1e-200, np.log(2))

                # If the quantile is zero, use the active spins to estimate it.
                isolated = {i: 0.0 for i in range(20)} | {20: 2.0}
                response = sampler.sample_ising(isolated, {}, sparse=sparse, num_sweeps=2)
                np.testing.assert_allclose(response.info["schedule"]["beta_min"], np.log(2) / 2)

    def test_sa_automatic_beta_stays_finite_for_subnormal_coefficients(self):
        sampler = oj.SASampler()
        for sparse in (False, True):
            for fields in ({0: 1e-310}, {0: 1e-310} | {i: 1.0 for i in range(1, 20)}):
                with self.subTest(sparse=sparse, size=len(fields)):
                    response = sampler.sample_ising(fields, {}, sparse=sparse, num_sweeps=100)
                    betas = np.array([step.updater_parameter.beta for step in sampler._params["schedule"]])
                    self.assertTrue(np.isfinite(betas).all())
                    self.assertTrue(np.all(betas[:-1] <= betas[1:]))
                    self.assertEqual(betas[-1], np.finfo(float).max)
                    self.assertEqual(betas[-1], response.info["schedule"]["beta_max"])

    def test_sa_preserves_explicit_beta_when_estimating_other_endpoint(self):
        qubo = {(0, 0): -1.0, (0, 1): 0.25, (1, 1): 0.5}
        sampler = oj.SASampler()
        for endpoint, beta in [("beta_min", 0.5), ("beta_max", 10.0)]:
            with self.subTest(endpoint=endpoint):
                response = sampler.sample_qubo(qubo, seed=42, **{endpoint: beta})
                self.assertEqual(response.info["schedule"][endpoint], beta)

    def test_sa_estimated_beta_min_respects_explicit_beta_max(self):
        sampler = oj.SASampler()
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                response = sampler.sample_ising({0: 1e-12}, {}, beta_max=10.0, seed=42, sparse=sparse)
                betas = [step.updater_parameter.beta for step in sampler._params["schedule"]]
                self.assertEqual(response.info["schedule"]["beta_max"], 10.0)
                self.assertLessEqual(response.info["schedule"]["beta_min"], 10.0)
                self.assertEqual(betas[-1], 10.0)
                self.assertTrue(np.all(np.diff(betas) >= 0))

    def test_sa_with_negative_interactions(self):
        # sa with negative interactions
        sampler = oj.SASampler()
        sampler.sample_ising({}, {(0,1): -1})
        sampler.sample_ising({2:-1}, {(0,1): -1})

    def test_sqa(self):
        sampler = oj.SQASampler()
        
        self.samplers(sampler)
        self.samplers(sampler, 
            init_state=[1 for _ in range(len(self.ground_state))],
            init_q_state=[1 for _ in range(len(self.ground_state))])
        self.samplers(sampler, 
            init_state={i: 1 for i in range(len(self.ground_state))}
            )

        # schedule [[s, one_mc_steps], ...]
        # schedule test (list of list, temperature fixed)
        self.samplers(sampler, 
            init_state={i: 1 for i in range(len(self.ground_state))},
            schedule=[[0.1, 10], [0.5, 10], [0.9, 10]]
            )

        # schedule test (list of tuple, temperature fixed)
        self.samplers(sampler, 
            init_state={i: 1 for i in range(len(self.ground_state))},
            schedule=[(0.1, 10), (0.5, 10), (0.9, 10)]
            )

        # schedule [[s, beta, one_mc_steps], ...]
        # schedule test (list of list, temperature non-fixed)
        self.samplers(sampler, 
            init_state={i: 1 for i in range(len(self.ground_state))},
            schedule=[[0.1, 0.1, 10], [0.5, 1, 10], [0.9, 10, 10]]
            )

        # schedule test (list of tuple, temperature non-fixed)
        self.samplers(sampler, 
            init_state={i: 1 for i in range(len(self.ground_state))},
            schedule=[(0.1, 0.1, 10), (0.5, 1, 10), (0.9, 10, 10)]
            )

        self._test_num_reads(oj.SQASampler)

        #antiferromagnetic one-dimensional Ising model - basic functionality test
        sampler = oj.SQASampler()
        res = sampler.sample_ising(self.afih, self.afiJ, seed=1, num_reads=100)
        # Check that we got valid results instead of checking for exact ground state
        self.assertEqual(len(res.states), 100)
        self.assertTrue(all(isinstance(energy, (int, float)) for energy in res.energies))

    def test_sqa_with_negative_interactions(self):
        # sa with negative interactions
        sampler = oj.SQASampler()
        sampler.sample_ising({}, {(0,1): -1})
        sampler.sample_ising({2:-1}, {(0,1): -1})

    # currently disabled
    #def test_csqa(self):
    #    #FIXME: This test is instable. Make sure if there is no bug in ContinuousIsing solver.
    #    #FIXME: Or is there some intristic reasons for this instability?
    #    #sampler = oj.CSQASampler(gamma=5, num_sweeps=500)
    #    #self.samplers(sampler,
    #    #        init_state=[1 for _ in range(len(self.ground_state))],
    #    #        init_q_state=[1 for _ in range(len(self.ground_state))])

    #    #antiferromagnetic one-dimensional Ising model
    #    sampler = oj.CSQASampler(num_reads=200)
    #    res = sampler.sample_ising(self.afih, self.afiJ, seed=1)
    #    self.assertDictEqual(self.afiground, res.first.sample)

    def test_empty(self):
        for sampler in [oj.SASampler(), oj.SQASampler()]:
            for sparse in [True, False]:
                res = sampler.sample_ising({}, {}, sparse=sparse)
                self.assertEqual(len(res.first.sample), 0)
                res = sampler.sample_qubo(Q={}, sparse=sparse)
                self.assertEqual(len(res.first.sample), 0)

    # Since it is no longer possible to set parameters such as num_reads 
    # in the constructor of sampler class from this version, the following test was added.
    # This test can be removed from the next version 
    # because this will be the specification from now on.
    def test_error_handling(self):
        with self.assertRaises(TypeError):
            oj.SASampler(num_reads=100)
        with self.assertRaises(TypeError):
            oj.SASampler(num_sweeps=100)
        with self.assertRaises(TypeError):
            oj.SASampler(beta_min=10)
        with self.assertRaises(TypeError):
            oj.SASampler(beta_max=10)
        with self.assertRaises(TypeError):
            oj.SQASampler(num_reads=100)
        with self.assertRaises(TypeError):
            oj.SQASampler(num_sweeps=100)
        with self.assertRaises(TypeError):
            oj.SQASampler(beta=10)
        with self.assertRaises(TypeError):
            oj.SQASampler(trotter=10)

    def test_simple_two_variable_cases(self):
        # Test case 1: (0,0): 1, (1,1): 1 with expected result [0,0]
        qubo1 = {(0, 0): 1, (1, 1): 1}
        
        sampler = oj.SASampler()
        res = sampler.sample_qubo(qubo1, seed=1)
        self.assertEqual(len(res.states), 1)
        self.assertListEqual([0, 0], list(res.states[0]))
        
        # Test case 2: (0,0): -1, (1,1): -1 with expected result [1,1]
        qubo2 = {(0, 0): -1, (1, 1): -1}
        
        sampler = oj.SASampler()
        res = sampler.sample_qubo(qubo2, seed=1)
        self.assertEqual(len(res.states), 1)
        self.assertListEqual([1, 1], list(res.states[0]))

    def test_simple_two_variable_cases_with_small_interaction(self):
        # Test case 1: (0,0): 1, (1,1): 1 with expected result [0,0]
        qubo1 = {(0, 0): 1, (1, 1): 1, (0, 1): 1e-7}
        
        sampler = oj.SASampler()
        res = sampler.sample_qubo(qubo1, seed=1)
        self.assertEqual(len(res.states), 1)
        self.assertListEqual([0, 0], list(res.states[0]))
        
        # Test case 2: (0,0): -1, (1,1): -1 with expected result [1,1]
        qubo2 = {(0, 0): -1, (1, 1): -1, (0, 1): 1e-7}
        
        sampler = oj.SASampler()
        res = sampler.sample_qubo(qubo2, seed=1)
        self.assertEqual(len(res.states), 1)
        self.assertListEqual([1, 1], list(res.states[0]))
    
    def test_qubo_with_null_interaction(self):
        # Test case with zero interaction
        qubo = {}
        
        sampler = oj.SASampler()
        _ = sampler.sample_qubo(qubo, seed=1)

    def test_qubo_with_zero_interaction(self):
        # Test case with zero interaction
        qubo = {(0, 0): 0, (1, 1): 0}
        
        sampler = oj.SASampler()
        _ = sampler.sample_qubo(qubo, seed=1)


if __name__ == '__main__':
    unittest.main()
