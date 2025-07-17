# Copyright 2023 Jij Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
from __future__ import annotations
try:
    from typing import Optional, Union, Any
except ImportError:
    from typing_extensions import Optional, Union, Any

import cimod
import dimod
import math
import time
import numpy as np

from collections import defaultdict
from dimod import BINARY, SPIN

import openjij
import openjij as oj
import openjij.cxxjij as cxxjij

from openjij.sampler.sampler import BaseSampler
from openjij.utils.graph_utils import qubo_to_ising
from openjij.sampler.base_sa_sample_hubo import base_sample_hubo, to_oj_response
from openjij.utils.cxx_cast import (
    cast_to_cxx_update_method,
    cast_to_cxx_random_number_engine,
    cast_to_cxx_temperature_schedule
)


"""This module contains Simulated Annealing sampler."""


class SASampler(BaseSampler):
    """Sampler with Simulated Annealing (SA).

    Args:
        beta_min (float):
            Minmum beta (inverse temperature).
            You can overwrite in methods .sample_*.

        beta_max (float):
            Maximum beta (inverse temperature).
            You can overwrite in methods .sample_*.

        num_reads (int):
            number of sampling (algorithm) runs. defaults None.
            You can overwrite in methods .sample_*.

        num_sweeps (int):
            number of MonteCarlo steps during SA. defaults None.
            You can overwrite in methods .sample_*.

        schedule_info (dict):
            Information about an annealing schedule.

    Raises:
        ValueError: If schedules or variables violate as below.
        - not list or numpy.array.
        - not list of tuple (beta : float, step_length : int).
        - beta is less than zero.
    """

    @property
    def parameters(self):
        return {
            "beta_min": ["parameters"],
            "beta_max": ["parameters"],
        }

    def __init__(self):

        # Set default parameters
        num_sweeps = 1000
        num_reads = 1
        beta_min = None
        beta_max = None
        schedule = None

        self._default_params = {
            "beta_min": beta_min,
            "beta_max": beta_max,
            "num_sweeps": num_sweeps,
            "schedule": schedule,
            "num_reads": num_reads,
        }

        self._params = self._default_params.copy()

        self._make_system = {
            "singlespinflip": cxxjij.system.make_classical_ising,
            "singlespinflippolynomial": cxxjij.system.make_classical_ising_polynomial,
            "swendsenwang": cxxjij.system.make_classical_ising,
        }
        self._algorithm = {
            "singlespinflip": cxxjij.algorithm.Algorithm_SingleSpinFlip_run,
            "singlespinflippolynomial": cxxjij.algorithm.Algorithm_SingleSpinFlip_run,
            "swendsenwang": cxxjij.algorithm.Algorithm_SwendsenWang_run,
        }

    def _convert_validation_schedule(self, schedule):
        """Checks if the schedule is valid and returns cxxjij schedule."""
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

        if isinstance(schedule[0], cxxjij.utility.ClassicalSchedule):
            return schedule

        if len(schedule[0]) != 2:
            raise ValueError(
                "schedule is list of tuple or list (beta : float, step_length : int)"
            )

        # schedule validation  0 <= beta
        beta = np.array(schedule).T[0]
        if not np.all(0 <= beta):
            raise ValueError("schedule beta range is '0 <= beta'.")

        # convert to cxxjij.utility.ClassicalSchedule
        cxxjij_schedule = []
        for beta, step_length in schedule:
            _schedule = cxxjij.utility.ClassicalSchedule()
            _schedule.one_mc_step = step_length
            _schedule.updater_parameter.beta = beta
            cxxjij_schedule.append(_schedule)

        return cxxjij_schedule

    def sample(
        self,
        bqm: Union[
            "openj.model.model.BinaryQuadraticModel", dimod.BinaryQuadraticModel
        ],
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
        num_sweeps: Optional[int] = None,
        num_reads: Optional[int] = None,
        schedule: Optional[list] = None,
        initial_state: Optional[Union[list, dict]] = None,
        updater: Optional[str] = None,
        sparse: Optional[bool] = None,
        reinitialize_state: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> "oj.sampler.response.Response":
        """Sample Ising model.

        Args:
            bqm (openjij.model.model.BinaryQuadraticModel) binary quadratic model
            beta_min (float): minimal value of inverse temperature
            beta_max (float): maximum value of inverse temperature
            num_sweeps (int): number of sweeps
            num_reads (int): number of reads
            schedule (list): list of inverse temperature
            initial_state (dict): initial state
            updater(str): updater algorithm
            sparse (bool): use sparse matrix or not.
            reinitialize_state (bool): if true reinitialize state for each run
            seed (int): seed for Monte Carlo algorithm
        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples:

            for Ising case::

                >>> h = {0: -1, 1: -1, 2: 1, 3: 1}
                >>> J = {(0, 1): -1, (3, 4): -1}
                >>> sampler = openj.SASampler()
                >>> res = sampler.sample_ising(h, J)

            for QUBO case::

                >>> Q = {(0, 0): -1, (1, 1): -1, (2, 2): 1, (3, 3): 1, (4, 4): 1, (0, 1): -1, (3, 4): 1}
                >>> sampler = openj.SASampler()
                >>> res = sampler.sample_qubo(Q)
        """

        # Set default parameters
        if updater is None:
            updater = "single spin flip"
        if sparse is None:
            sparse = True
        if reinitialize_state is None:
            reinitialize_state = True

        _updater_name = updater.lower().replace("_", "").replace(" ", "")
        # swendsen wang algorithm runs only on sparse ising graphs.
        if _updater_name == "swendsenwang" or sparse:
            sparse = True
        else:
            sparse = False

        if isinstance(bqm, dimod.BinaryQuadraticModel):
            bqm = oj.model.model.BinaryQuadraticModel(
                dict(bqm.linear),
                dict(bqm.quadratic),
                bqm.offset,
                bqm.vartype,
                sparse=sparse,
            )

        if sparse == True and bqm.sparse == False:
            # convert to sparse bqm
            bqm = oj.model.model.BinaryQuadraticModel(
                bqm.linear, bqm.quadratic, bqm.offset, bqm.vartype, sparse=True
            )

        # alias
        model = bqm

        ising_graph, offset = model.get_cxxjij_ising_graph()

        self._set_params(
            beta_min=beta_min,
            beta_max=beta_max,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            schedule=schedule,
        )

        # set annealing schedule -------------------------------
        if self._params["schedule"] is None:
            self._params["schedule"], beta_range = geometric_ising_beta_schedule(
                cxxgraph=ising_graph,
                beta_max=self._params["beta_max"],
                beta_min=self._params["beta_min"],
                num_sweeps=self._params["num_sweeps"],
                seed=seed,
            )
            self.schedule_info = {
                "beta_max": beta_range[0],
                "beta_min": beta_range[1],
                "num_sweeps": self._params["num_sweeps"],
            }
        else:
            self._params["schedule"] = self._convert_validation_schedule(
                self._params["schedule"]
            )
            self.schedule_info = {"schedule": "custom schedule"}
        # ------------------------------- set annealing schedule

        # make init state generator --------------------------------
        if initial_state is None:

            def _generate_init_state():
                return (
                    ising_graph.gen_spin(seed)
                    if seed is not None
                    else ising_graph.gen_spin()
                )

        else:
            temp_initial_state = []
            if isinstance(initial_state, dict):
                if model.vartype == BINARY:
                    for k in model.variables:
                        v = initial_state[k]
                        if v != 0 and v != 1:
                            raise RuntimeError("The initial variables must be 0 or 1.")
                        temp_initial_state.append(2 * v - 1)
                elif model.vartype == SPIN:
                    for k in model.variables:
                        v = initial_state[k]
                        if v != -1 and v != 1:
                            raise RuntimeError(
                                "The initial variables must be -1 or +1."
                            )
                        temp_initial_state.append(v)
                else:
                    raise RuntimeError("Unknown vartype detected.")
            elif isinstance(initial_state, (list, tuple)):
                if model.vartype == BINARY:
                    for k in range(len(model.variables)):
                        v = initial_state[k]
                        if v != 0 and v != 1:
                            raise RuntimeError("The initial variables must be 0 or 1.")
                        temp_initial_state.append(2 * v - 1)
                elif model.vartype == SPIN:
                    for k in range(len(model.variables)):
                        v = initial_state[k]
                        if v != -1 and v != 1:
                            raise RuntimeError(
                                "The initial variables must be -1 or +1."
                            )
                        temp_initial_state.append(v)
                else:
                    raise RuntimeError("Unknown vartype detected.")
            else:
                raise RuntimeError("Unsupported type of initial_state.")

            _init_state = np.array(temp_initial_state)

            # validate initial_state size
            if len(initial_state) != ising_graph.size():
                raise ValueError(
                    "the size of the initial state should be {}".format(
                        ising_graph.size()
                    )
                )

            def _generate_init_state():
                return np.array(_init_state)

        # -------------------------------- make init state generator

        # choose updater -------------------------------------------
        _updater_name = updater.lower().replace("_", "").replace(" ", "")
        if _updater_name not in self._make_system:
            raise ValueError('updater is one of "single spin flip or swendsen wang"')
        algorithm = self._algorithm[_updater_name]
        sa_system = self._make_system[_updater_name](
            _generate_init_state(), ising_graph
        )
        # ------------------------------------------- choose updater
        response = self._cxxjij_sampling(
            model, _generate_init_state, algorithm, sa_system, reinitialize_state, seed
        )

        response.info["schedule"] = self.schedule_info

        return response
    
    def _sample_hubo_old(
        self,
        J: Union[
            dict, "openj.model.model.BinaryPolynomialModel", cimod.BinaryPolynomialModel
        ],
        vartype: Optional[str] = None,
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
        num_sweeps: Optional[int] = None,
        num_reads: Optional[int] = None,
        schedule: Optional[list] = None,
        initial_state: Optional[Union[list, dict]] = None,
        updater: Optional[str] = None,
        reinitialize_state: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> "openjij.sampler.response.Response":
        """Sampling from higher order unconstrainted binary optimization.

        Args:
            J (dict): Interactions.
            vartype (str, openjij.VarType): "SPIN" or "BINARY".
            beta_min (float, optional): Minimum beta (initial inverse temperature). Defaults to None.
            beta_max (float, optional): Maximum beta (final inverse temperature). Defaults to None.
            schedule (list, optional): schedule list. Defaults to None.
            num_sweeps (int, optional): number of sweeps. Defaults to None.
            num_reads (int, optional): number of reads. Defaults to 1.
            init_state (list, optional): initial state. Defaults to None.
            reinitialize_state (bool): if true reinitialize state for each run
            seed (int, optional): seed for Monte Carlo algorithm. Defaults to None.

        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples::
            for Ising case::
                >>> sampler = openjij.SASampler()
                >>> J = {(0,): -1, (0, 1): -1, (0, 1, 2): 1}
                >>> response = sampler.sample_hubo(J, "SPIN")

            for Binary case::
                >>> sampler = ooenjij.SASampler()
                >>> J = {(0,): -1, (0, 1): -1, (0, 1, 2): 1}
                >>> response = sampler.sample_hubo(J, "BINARY")
        """

        # Set default parameters
        if reinitialize_state is None:
            reinitialize_state = True

        # Set model
        if str(type(J)) == str(type(oj.model.model.BinaryPolynomialModel({}, "SPIN"))):
            if vartype is not None:
                raise ValueError("vartype must not be specified")
            model = J
        elif str(type(J)) == str(type(cimod.BinaryPolynomialModel({}, "SPIN"))):
            if vartype is not None:
                raise ValueError("vartype must not be specified")
            model = J
        else:
            model = oj.model.model.BinaryPolynomialModel(J, vartype)

        # make init state generator --------------------------------
        if initial_state is None:
            if model.vartype == SPIN:

                def _generate_init_state():
                    return (
                        cxxjij.graph.Polynomial(model.num_variables).gen_spin(seed)
                        if seed is not None
                        else cxxjij.graph.Polynomial(model.num_variables).gen_spin()
                    )

            elif model.vartype == BINARY:

                def _generate_init_state():
                    return (
                        cxxjij.graph.Polynomial(model.num_variables).gen_binary(seed)
                        if seed is not None
                        else cxxjij.graph.Polynomial(model.num_variables).gen_binary()
                    )

            else:
                raise ValueError("Unknown vartype detected")
        else:
            if isinstance(initial_state, dict):
                initial_state = [initial_state[k] for k in model.indices]

            def _generate_init_state():
                return np.array(initial_state)

        # -------------------------------- make init state generator

        # determine system class and algorithm --------------------------------
        if model.vartype == SPIN:
            if updater is None or updater == "single spin flip":
                sa_system = cxxjij.system.make_classical_ising_polynomial(
                    _generate_init_state(), model.to_serializable()
                )
                algorithm = cxxjij.algorithm.Algorithm_SingleSpinFlip_run
            elif updater == "k-local":
                raise ValueError(
                    "k-local update is only supported for binary variables"
                )
            else:
                raise ValueError("Unknown updater name")
        elif model.vartype == BINARY:
            if updater == "k-local":
                sa_system = cxxjij.system.make_k_local_polynomial(
                    _generate_init_state(), model.to_serializable()
                )
                algorithm = cxxjij.algorithm.Algorithm_KLocal_run
            elif updater is None or updater == "single spin flip":
                sa_system = cxxjij.system.make_classical_ising_polynomial(
                    _generate_init_state(), model.to_serializable()
                )
                algorithm = cxxjij.algorithm.Algorithm_SingleSpinFlip_run
            else:
                raise ValueError("Unknown updater name")
        else:
            raise ValueError("Unknown vartype detected")
        # -------------------------------- determine system class and algorithm

        self._set_params(
            beta_min=beta_min,
            beta_max=beta_max,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            schedule=schedule,
        )

        # set annealing schedule -------------------------------
        if self._params["schedule"] is None:
            self._params["schedule"], beta_range = geometric_hubo_beta_schedule(
                sa_system,
                self._params["beta_max"],
                self._params["beta_min"],
                self._params["num_sweeps"],
                seed=seed,
            )
            self.schedule_info = {
                "beta_max": beta_range[0],
                "beta_min": beta_range[1],
                "num_sweeps": self._params["num_sweeps"],
            }
        else:
            self.schedule_info = {"schedule": "custom schedule"}
        # ------------------------------- set annealing schedule

        response = self._cxxjij_sampling(
            model, _generate_init_state, algorithm, sa_system, reinitialize_state, seed
        )

        response.info["schedule"] = self.schedule_info

        return response
    
    def sample_hubo(
        self,
        J: dict[tuple, float],
        vartype: Optional[str] = None,
        num_sweeps: int = 1000,
        num_reads: int = 1,
        num_threads: int = 1,
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
        updater: str = "METROPOLIS",
        random_number_engine: str = "XORSHIFT",
        seed: Optional[int] = None,
        temperature_schedule: str = "GEOMETRIC",
    ):  
        """Sampling from higher order unconstrained binary optimization.

        Args:
            J (dict): Interactions.
            vartype (str): "SPIN" or "BINARY".
            num_sweeps (int, optional): The number of sweeps. Defaults to 1000.
            num_reads (int, optional): The number of reads. Defaults to 1.
            num_threads (int, optional): The number of threads. Parallelized for each sampling with num_reads > 1. Defaults to 1.
            beta_min (float, optional): Minimum beta (initial inverse temperature). Defaults to None.
            beta_max (float, optional): Maximum beta (final inverse temperature). Defaults to None.
            updater (str, optional): Updater. One can choose "METROPOLIS", "HEAT_BATH", or "k-local". Defaults to "METROPOLIS".
            random_number_engine (str, optional): Random number engine. One can choose "XORSHIFT", "MT", or "MT_64". Defaults to "XORSHIFT".            
            seed (int, optional): seed for Monte Carlo algorithm. Defaults to None.
            temperature_schedule (str, optional): Temperature schedule. One can choose "LINEAR", "GEOMETRIC". Defaults to "GEOMETRIC".

        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples::
            for Ising case::
                >>> sampler = openjij.SASampler()
                >>> J = {(0,): -1, (0, 1): -1, (0, 1, 2): 1}
                >>> response = sampler.sample_hubo(J, "SPIN")

            for Binary case::
                >>> sampler = openjij.SASampler()
                >>> J = {(0,): -1, (0, 1): -1, (0, 1, 2): 1}
                >>> response = sampler.sample_hubo(J, "BINARY")
        """


        if updater=="k-local" or not isinstance(J, dict):
            # To preserve the correspondence with the old version.
            if updater=="METROPOLIS":
                updater="single spin flip"
            return self._sample_hubo_old(
                J=J,
                vartype=vartype,
                beta_min=beta_min,
                beta_max=beta_max,
                num_sweeps=num_sweeps,
                num_reads=num_reads,
                #schedule,
                #initial_state,
                updater=updater,
                #reinitialize_state,
                seed=seed
            )
        else:
            # To preserve the correspondence with the old version.
            if updater=="single spin flip":
                updater="METROPOLIS"
            return base_sample_hubo(
                hubo=J,
                vartype=vartype,
                num_sweeps=num_sweeps,
                num_reads=num_reads,
                num_threads=num_threads,
                beta_min=beta_min,
                beta_max=beta_max,
                update_method=updater,
                random_number_engine=random_number_engine,
                seed=seed,
                temperature_schedule=temperature_schedule
            )
    
    def _base_integer_sampler(
        self,
        J: dict[tuple, float],
        bound_list: dict[Any, tuple[int, int]],
        include_higher_order: bool,
        num_sweeps: int = 1000,
        num_reads: int = 1,
        num_threads: int = 1,
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
        updater: str = "OPT_METROPOLIS",
        random_number_engine: str = "XORSHIFT",
        seed: Optional[int] = None,
        temperature_schedule: str = "GEOMETRIC",
        log_history: bool = False,
    ) -> "oj.sampler.response.Response":

        start_solving = time.perf_counter()

        if not isinstance(J, dict):
            raise TypeError("J must be a dictionary of interactions.")
        if len(J) == 0:
            raise ValueError("J must not be an empty dictionary.")

        # Summarize interactions
        summarize_interactions = defaultdict(float)
        index_set = set()
        for key, value in J.items():
            if not include_higher_order and len(key) > 2:
                raise ValueError(
                    "Only pairwise interactions are supported. Please use `sample_huio` for higher-order interactions with integer variables."
                )
            key_list = tuple(sorted(list(key), key=lambda x: (isinstance(x, str), x)))
            summarize_interactions[key_list] += value
            index_set.update(key)
        
        self.index_list = sorted(index_set, key=lambda x: (isinstance(x, str), x))
        self.num_variables = len(self.index_list)

        # Create a mapping from the original indices to integer indices
        self.index_map = {index: i for i, index in enumerate(self.index_list)}

        # Convert keys to integer indices
        int_key_list = []
        int_value_list = []
        int_bound_list = []
        for key, value in summarize_interactions.items():
            int_key = [self.index_map[i] for i in key]
            int_key_list.append(int_key)
            int_value_list.append(value)

        for i in range(self.num_variables):
            index = self.index_list[i]
            if index not in bound_list:
                raise ValueError(f"Index {index} not found in bound_list.")
            if bound_list[index][0] >= bound_list[index][1]:
                raise ValueError(f"Index {index} has no variable range.")
            int_bound_list.append(
                (bound_list[index][0], bound_list[index][1])
            )

        if include_higher_order:
            cxx_model = cxxjij.graph.IntegerPolynomialModel(
                key_list=int_key_list,
                value_list=int_value_list,
                bound_list=int_bound_list,
            )
        else:
            cxx_model = cxxjij.graph.IntegerQuadraticModel(
                key_list=int_key_list,
                value_list=int_value_list,
                bound_list=int_bound_list,
            )

        if beta_min is None or beta_max is None:
            max_coeff, min_coeff = cxx_model.get_max_min_terms()
            if beta_min is None:
                max_T = max_coeff / math.log(2)
            if beta_max is None:
                min_T = min_coeff / math.log(100)
        if beta_min is not None:
            max_T = 1.0/beta_min
        if beta_max is not None:
            min_T = 1.0/beta_max

        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)

        preprocess_time = time.perf_counter() - start_solving

        # Start sampling
        start_sample = time.perf_counter()
        cxx_sampler = cxxjij.sampler.sample_by_integer_sa_polynomial if include_higher_order else cxxjij.sampler.sample_by_integer_sa_quadratic
        cxx_result_list = cxx_sampler(
            model=cxx_model,
            num_sweeps=num_sweeps,
            update_method=cast_to_cxx_update_method(updater),
            rand_type=cast_to_cxx_random_number_engine(random_number_engine),
            schedule=cast_to_cxx_temperature_schedule(temperature_schedule),
            num_reads=num_reads,
            seed=seed,
            num_threads=num_threads,
            min_T=min_T,
            max_T=max_T,
            log_history=log_history,
        )
        sample_time = time.perf_counter() - start_sample

        # Make openjij response
        start_make_oj_response = time.perf_counter()
        oj_response = to_oj_response(
            variables=[r.solution for r in cxx_result_list], 
            index_list=self.index_list,
            energies=[r.energy for r in cxx_result_list],
            vartype=oj.Vartype.DISCRETE
        )

        oj_response.info["schedule"] = {
            "num_sweeps": num_sweeps,
            "num_reads": num_reads,
            "num_threads": num_threads,
            "beta_min": 1.0 / max_T,
            "beta_max": 1.0 / min_T,
            "update_method": updater,
            "random_number_engine": random_number_engine,
            "temperature_schedule": temperature_schedule,
            "seed": seed,
        }

        oj_response.info["log"] = {
            "energy_history": np.array([r.energy_history for r in cxx_result_list]),
            "temperature_history": np.array([r.temperature_history for r in cxx_result_list]),
        }

        oj_response.info["time"] = {
            "preprocess": preprocess_time,
            "sample": sample_time,
            "make_oj_response": time.perf_counter() - start_make_oj_response,
            "total": time.perf_counter() - start_solving,
        }

        return oj_response
    
    def sample_quio(
        self,
        J: dict[tuple, float],
        bound_list: dict[Any, tuple[int, int]],
        num_sweeps: int = 1000,
        num_reads: int = 1,
        num_threads: int = 1,
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
        updater: str = "OPT_METROPOLIS",
        random_number_engine: str = "XORSHIFT",
        seed: Optional[int] = None,
        temperature_schedule: str = "GEOMETRIC",
        log_history: bool = False,
    ) -> "oj.sampler.response.Response":
        """Sampling from quadratic unconstrained integer optimization (QUIO).
        This method solves integer optimization problems with interactions up to quadratic order (linear and quadratic terms only).

        Args:
            J (dict): Interactions. Keys are tuples of variable indices, values are interaction coefficients.
            bound_list (dict): Variable bounds. Keys are variable indices, values are tuples of (lower_bound, upper_bound) for integer variables.
            num_sweeps (int, optional): The number of sweeps. Defaults to 1000.
            num_reads (int, optional): The number of reads. Defaults to 1.
            num_threads (int, optional): The number of threads. Parallelized for each sampling with num_reads > 1. Defaults to 1.
            beta_min (float, optional): Minimum beta (initial inverse temperature). Defaults to None.
            beta_max (float, optional): Maximum beta (final inverse temperature). Defaults to None.
            updater (str, optional): Updater. One can choose "METROPOLIS", "OPT_METROPOLIS", "HEAT_BATH", and "SUWA_TODO". Defaults to "OPT_METROPOLIS".
            random_number_engine (str, optional): Random number engine. One can choose "XORSHIFT", "MT", or "MT_64". Defaults to "XORSHIFT".            
            seed (int, optional): Seed for Monte Carlo algorithm. Defaults to None.
            temperature_schedule (str, optional): Temperature schedule. One can choose "LINEAR", "GEOMETRIC". Defaults to "GEOMETRIC".
            log_history (bool, optional): If True, logs the energy and temperature history. Defaults to False.

        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples::
            To solve f(x) = -x_0 - x_0*x_1 + x_1*x_2 with bounds:
            x_0 in [-10, 10], x_1 in [3, 10], use the following code:
            >>> import openjij
            >>> sampler = openjij.SASampler()
            >>> J = {(0,): -1, (0, 1): -1, (1, 2): 1}
            >>> bound_list = {0: (-10, 10), 1: (3, 10), 2: (-2, -1)}
            >>> response = sampler.sample_quio(J, bound_list)
        """
        return self._base_integer_sampler(
            J=J,
            bound_list=bound_list,
            include_higher_order=False,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            num_threads=num_threads,
            beta_min=beta_min,
            beta_max=beta_max,
            updater=updater,
            random_number_engine=random_number_engine,
            seed=seed,
            temperature_schedule=temperature_schedule,
            log_history=log_history
        )
        
    
    def sample_huio(
        self,
        J: dict[tuple, float],
        bound_list: dict[Any, tuple[int, int]],
        num_sweeps: int = 1000,
        num_reads: int = 1,
        num_threads: int = 1,
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
        updater: str = "OPT_METROPOLIS",
        random_number_engine: str = "XORSHIFT",
        seed: Optional[int] = None,
        temperature_schedule: str = "GEOMETRIC",
        log_history: bool = False,
    ) -> "oj.sampler.response.Response":
        """Sampling from higher-order unconstrained integer optimization (HUIO).
        This method solves integer optimization problems that can include variable interactions of any order (linear, quadratic, cubic, and higher).
        
        Args:
            J (dict): Interactions. Keys are tuples of variable indices, values are interaction coefficients.
            bound_list (dict): Variable bounds. Keys are variable indices, values are tuples of (lower_bound, upper_bound) for integer variables.
            num_sweeps (int, optional): The number of sweeps. Defaults to 1000.
            num_reads (int, optional): The number of reads. Defaults to 1.
            num_threads (int, optional): The number of threads. Parallelized for each sampling with num_reads > 1. Defaults to 1.
            beta_min (float, optional): Minimum beta (initial inverse temperature). Defaults to None.
            beta_max (float, optional): Maximum beta (final inverse temperature). Defaults to None.
            updater (str, optional): Updater. One can choose "METROPOLIS", "OPT_METROPOLIS", "HEAT_BATH", and "SUWA_TODO". Defaults to "OPT_METROPOLIS".
            random_number_engine (str, optional): Random number engine. One can choose "XORSHIFT", "MT", or "MT_64". Defaults to "XORSHIFT".            
            seed (int, optional): Seed for Monte Carlo algorithm. Defaults to None.
            temperature_schedule (str, optional): Temperature schedule. One can choose "LINEAR", "GEOMETRIC". Defaults to "GEOMETRIC".
            log_history (bool, optional): If True, logs the energy and temperature history. Defaults to False.

        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples::
            To solve f(x) = -x_0 - x_0*x_1 + x_0*x_1*x_2 with bounds:
            x_0 in [-10, 10], x_1 in [3, 10], x_2 in [-2, -1], use the following code:
            >>> import openjij
            >>> sampler = openjij.SASampler()
            >>> J = {(0,): -1, (0, 1): -1, (0, 1, 2): 1}
            >>> bound_list = {0: (-10, 10), 1: (3, 10), 2: (-2, -1)}
            >>> response = sampler.sample_huio(J, bound_list)
        """
        return self._base_integer_sampler(
            J=J,
            bound_list=bound_list,
            include_higher_order=True,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            num_threads=num_threads,
            beta_min=beta_min,
            beta_max=beta_max,
            updater=updater,
            random_number_engine=random_number_engine,
            seed=seed,
            temperature_schedule=temperature_schedule,
            log_history=log_history
        )

def geometric_hubo_beta_schedule(sa_system, beta_max, beta_min, num_sweeps, seed=None):
    max_delta_energy = sa_system.get_max_effective_dE()
    min_delta_energy = sa_system.get_min_effective_dE()

    if beta_min is None:
        beta_min = np.log(2) / max_delta_energy

    if beta_max is None:
        beta_max = np.log(100) / min_delta_energy

    num_sweeps_per_beta = max(1, num_sweeps // 1000)

    schedule = cxxjij.utility.make_classical_schedule_list(
        beta_min=beta_min,
        beta_max=beta_max,
        one_mc_step=num_sweeps_per_beta,
        num_call_updater=num_sweeps // num_sweeps_per_beta,
    )

    return schedule, [beta_max, beta_min]


def geometric_ising_beta_schedule(
    cxxgraph: Union[openjij.cxxjij.graph.Dense, openjij.cxxjij.graph.CSRSparse],
    beta_max=None,
    beta_min=None,
    num_sweeps=1000,
    seed=None,
):
    """Make geometric cooling beta schedule.

    Args:
        cxxgraph (Union[openjij.cxxjij.graph.Dense, openjij.cxxjij.graph.CSRSparse]): Ising graph, must be either `Dense` or `CSRSparse`.
        beta_max (float, optional): [description]. Defaults to None.
        beta_min (float, optional): [description]. Defaults to None.
        num_sweeps (int, optional): [description]. Defaults to 1000.
        seed (int, optional): Seed for random number generation. Defaults to None.
    Returns:
        list of cxxjij.utility.ClassicalSchedule, list of beta range [max, min]
    """

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    THRESHOLD = 1e-8
    dE_max = 1.0
    dE_min = 1.0
    ising_interaction = cxxgraph.get_interactions()
    rate_dE = 0.5

    if beta_min is None or beta_max is None:
        # if `abs_ising_interaction` is empty, set min/max delta_energy to 1 (a trivial case).
        if ising_interaction.shape[0] <= 1:
            dE_max = 1
            dE_min = 1
        else:
            random_spin = np.random.choice([-1, 1], size=(ising_interaction.shape[0], 2))
            random_spin[-1, :] = 1  # last element is bias term

            # Calculate quadratic and linear term energy difference separately
            dE_quad = (ising_interaction[:-1, :-1] @ random_spin[:-1, :] * (-2 * random_spin[:-1, :]))
            if isinstance(ising_interaction, np.ndarray):
                dE_linear = (ising_interaction[:-1, -1][:, np.newaxis] * random_spin[:-1])
            else:
                dE_linear = (ising_interaction[:-1, -1].toarray() * random_spin[:-1])

            dE_quad_abs = np.abs(dE_quad)
            rate_dE = np.max(np.abs(dE_linear[dE_quad_abs > THRESHOLD]) /(dE_quad_abs[dE_quad_abs > THRESHOLD].mean() + THRESHOLD))

            dE = dE_quad
            dE_positive = dE[dE > THRESHOLD]

            if len(dE_positive) == 0:
                dE_min = 1
                dE_max = 1.1
            else:
                dE_max = np.median(dE_positive, axis=0).mean()
                dE_min = np.min(dE_positive, axis=0).mean()


    n = ising_interaction.shape[0]  # n+1

    prob_init = 9/10 - 8/10 * np.tanh(rate_dE/50) + THRESHOLD
    prob_final = 1 / 1000

    if beta_min is None:
        p_init = prob_init
        # 二分探索で 2 beta p_init = exp(-beta dE) - exp(-beta dE_max)を解く
        # Binary search to find beta where f = 0
        beta_low, beta_high = -np.log(prob_init) / dE_max, -np.log(prob_init) / dE_min
        tolerance = 1e-8
        max_iterations = 5
        for iteration in range(max_iterations):
            _beta = (beta_low + beta_high) / 2
            f = 2 * _beta * p_init - np.exp(-_beta * dE_min) + np.exp(-_beta * dE_max)
            if abs(f) < tolerance:
                break
            if f > 0:
                beta_high = _beta
            else:
                beta_low = _beta
        beta_min = beta_low

        beta_min_min = beta_min * 0.8
        beta_min = beta_min_min + (beta_min * 1.5 - beta_min_min) * np.tanh(num_sweeps / n)

    if beta_max is None:
        int_abs = np.abs(ising_interaction)
        int_min = np.min(int_abs[int_abs > THRESHOLD])

        int_min = max(int_min, dE_min / 10.0)

        beta_max_int = -np.log(prob_final) / int_min
        beta_max_dE = -np.log(prob_final) / dE_min

        if rate_dE < 2:
            beta_max_min = min(beta_max_dE, beta_max_int)
            beta_max_max = max(beta_max_dE, beta_max_int)
            beta_max = beta_max_min + (beta_max_max - beta_max_min) * np.tanh(num_sweeps / (2 * n))
        else:
            last_step = min(100, num_sweeps // 10 + 1)
            # last_stepだけ繰り返した時に少なくとも1度フリップする確率を計算
            # 1stepでは exp(-beta dE) でフリップするとする.
            _dE = max(dE_min, int_min)
            beta_max = -np.log(prob_final) / _dE / last_step

        if beta_max < beta_min:
            beta_max = beta_min * 10.0

    if rate_dE > 1:
        beta_min *= rate_dE
        beta_max *= rate_dE

    num_sweeps_per_beta = max(1, num_sweeps // 1000)
    # set schedule to cxxjij
    schedule = cxxjij.utility.make_classical_schedule_list(
        beta_min=beta_min,
        beta_max=beta_max,
        one_mc_step=num_sweeps_per_beta,
        num_call_updater=num_sweeps // num_sweeps_per_beta,
    )

    return schedule, [beta_max, beta_min]
