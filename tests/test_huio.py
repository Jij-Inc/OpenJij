import unittest
import random
import openjij as oj
import numpy as np
from itertools import product

class HUIOTest(unittest.TestCase):

    def setUp(self):
        self.upd = ["METROPOLIS", "OPT_METROPOLIS", "HEAT_BATH", "SUWA_TODO"]
        self.ran = ["XORSHIFT", "MT", "MT_64"]
        self.sch = ["GEOMETRIC", "LINEAR"]
        self.seed = 42
        random.seed(self.seed)

    def test_qubo_1(self):
        Q = {}
        n = 5
        for i in range(n):
            for j in range(i, n):
                Q[(i, j)] = random.randint(-10, 10)
        bound_list = {i: (0, 1) for i in range(n)}

        r1 = oj.SASampler().sample_hubo(Q, num_reads=30, seed=self.seed, vartype="BINARY")

        for x, y, z in product(self.upd, self.ran, self.sch):
            r2 = oj.SASampler().sample_huio(Q, bound_list=bound_list, num_reads=30, 
                                             updater=x, random_number_engine=y, 
                                             temperature_schedule=z, seed=self.seed)
            self.assertAlmostEqual(r1.first.energy, r2.first.energy)
            self.assertEqual(r1.first.sample, r2.first.sample)

    def test_qubo_2(self):
        Q = {}
        n = 5
        for i in range(n):
            for j in range(i, n):
                Q[(str(i), str(j))] = random.randint(-10, 10)
        bound_list = {str(i): (0, 1) for i in range(n)}

        r1 = oj.SASampler().sample_hubo(Q, num_reads=30, seed=self.seed, vartype="BINARY")
        for x, y, z in product(self.upd, self.ran, self.sch):
            r2 = oj.SASampler().sample_huio(Q, bound_list=bound_list, num_reads=30, 
                                             updater=x, random_number_engine=y, 
                                             temperature_schedule=z,
                                             seed=self.seed)
            self.assertAlmostEqual(r1.first.energy, r2.first.energy)
            self.assertEqual(r1.first.sample, r2.first.sample)


    def test_hubo_1(self):
        Q = {}
        n = 5
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    Q[(i, j, k)] = random.randint(-10, 10)
        bound_list = {i: (0, 1) for i in range(n)}

        r1 = oj.SASampler().sample_hubo(Q, num_reads=30, vartype="BINARY", seed=self.seed)

        for x, y, z in product(self.upd, self.ran, self.sch):
            r2 = oj.SASampler().sample_huio(Q, bound_list=bound_list, num_reads=30, 
                                             updater=x, random_number_engine=y, 
                                             temperature_schedule=z,
                                             seed=self.seed)
            self.assertAlmostEqual(r1.first.energy, r2.first.energy)
            self.assertEqual(r1.first.sample, r2.first.sample)

    def test_hubo_2(self):
        Q = {}
        n = 5
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    Q[(str(i), str(j), str(k))] = random.randint(-10, 10)
        bound_list = {str(i): (0, 1) for i in range(n)}

        r1 = oj.SASampler().sample_hubo(Q, num_reads=30, vartype="BINARY", seed=self.seed)

        for x, y, z in product(self.upd, self.ran, self.sch):
            r2 = oj.SASampler().sample_huio(Q, bound_list=bound_list, num_reads=30, 
                                             updater=x, random_number_engine=y, 
                                             temperature_schedule=z,
                                             seed=self.seed)
            self.assertAlmostEqual(r1.first.energy, r2.first.energy)
            self.assertEqual(r1.first.sample, r2.first.sample)

    def test_quio_integer_optimal_1(self):
        Q = {(0, 0, 0): 1.0, (1, 1): 1.0, (0, 1): 2.0, (0,): -6.0, (1,): -4.0, (): 1.0}
        bound_list = {0: (0, 3), 1: (0, 3)}

        for x, y, z in product(self.upd, self.ran, self.sch):
            r = oj.SASampler().sample_huio(Q, bound_list=bound_list, num_reads=30,
                                            updater=x, random_number_engine=y,
                                            temperature_schedule=z,
                                             seed=self.seed)
            self.assertAlmostEqual(r.first.energy, -5)
            self.assertEqual(r.first.sample, {0: 1, 1: 1})

    def test_quio_integer_optimal_2(self):
        Q = {(0, 0, 0): 1.0, (1, 1): 2.0, (2,): 3.0, (0, 1, 2): 3.0, (0, 0, 1): 1.0, (0,): -4.0, (1,): -5.0, (): 1.0}
        bound_list = {0: (0, 2), 1: (0, 2), 2: (0, 2)}

        for x, y, z in product(self.upd, self.ran, self.sch):
            r = oj.SASampler().sample_huio(Q, bound_list=bound_list, num_reads=30,
                                            updater=x, random_number_engine=y,
                                            temperature_schedule=z,
                                             seed=self.seed)
            self.assertAlmostEqual(r.first.energy, -4)
            self.assertEqual(r.first.sample, {0: 1, 1: 1, 2: 0})

    def test_huio_integer_corner_case_1(self):
        Q = {}
        bound_list = {}

        for x, y, z in product(self.upd, self.ran, self.sch):
            with self.assertRaises(ValueError):
                _ = oj.SASampler().sample_huio(Q, bound_list=bound_list, num_reads=30,
                                            updater=x, random_number_engine=y,
                                            temperature_schedule=z,
                                             seed=self.seed)
                
    def test_huio_integer_corner_case_2(self):
        Q = {(0, 0): 1.0, (1, 1): 1.0, (0, 1): -4.0}
        bound_list = {0: (2, 2), 1: (0, 2)}

        for x, y, z in product(self.upd, self.ran, self.sch):
            with self.assertRaises(ValueError):
                _ = oj.SASampler().sample_huio(Q, bound_list=bound_list, num_reads=30, 
                                            updater=x, random_number_engine=y, 
                                            temperature_schedule=z,
                                             seed=self.seed)