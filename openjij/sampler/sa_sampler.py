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
from typing import Optional, Union, Any

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

        Automatic temperatures for single-spin updates use coefficient
        quantiles and can include a colder tail. The reported ``beta_max``
        is the final inverse temperature, including the tail when present.

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
            seed (int): seed for reproducible batches with a distinct random seed per read
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
                from_qubo=model.vartype == BINARY,
                use_quantiles=_updater_name == "singlespinflip",
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

            def _generate_init_state(read_seed=seed):
                return (
                    ising_graph.gen_spin(read_seed)
                    if read_seed is not None
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

            def _generate_init_state(read_seed=seed):
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
        read_seeds = None
        if seed is not None and self._params["num_reads"] > 1:
            # Xorshift takes a 32-bit seed. An odd step visits each seed once
            # before wrapping, starting with the original seed for the first read.
            seed_step = 2 * int(np.random.default_rng(seed).integers(2**31)) + 1
            read_seeds = (
                (int(seed) + read * seed_step) % 2**32
                for read in range(self._params["num_reads"])
            )
        response = self._cxxjij_sampling(
            model,
            _generate_init_state,
            algorithm,
            sa_system,
            reinitialize_state,
            seed,
            read_seeds=read_seeds,
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
        log_history: bool = False,
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
            log_history (bool, optional): If True, records the energy and temperature after each sweep in
                ``response.info["log"]``. The arrays have shape ``(num_reads, num_sweeps)`` and do not include
                the initial state. This is supported for dictionary inputs with the "METROPOLIS", "HEAT_BATH",
                or "single spin flip" updater; it is not supported by the legacy ``sample_hubo`` path used for
                polynomial-model inputs and the "k-local" updater. On the supported path, False returns empty
                history arrays with shape ``(num_reads, 0)``. Defaults to False.

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
            if log_history:
                raise NotImplementedError(
                    "log_history is only supported for dictionary inputs with "
                    "the METROPOLIS, HEAT_BATH, or single spin flip updater."
                )
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
                temperature_schedule=temperature_schedule,
                log_history=log_history,
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


def _ising_beta_range_statistics(cxxgraph, *, from_qubo):
    """Per-spin coefficient statistics of an Ising graph.

    Returns:
        Each spin's RMS local field in scaled units, the corresponding
        per-spin scale, and each spin's smallest nonzero coefficient in
        the original units. Spins with all-zero coefficients are omitted
        from the last array.
    """
    n = cxxgraph.size()
    if n == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    # Rows of spins; the last column holds the local fields. The auxiliary
    # last row (with its unit diagonal) is not part of the model.
    coefficients = abs(cxxgraph.get_interactions()[:-1, :])
    # A QUBO field is Q_ii / 2 plus the incident Ising couplings. Near
    # cancellation, use a roundoff bound that grows with the number of
    # terms added and their total magnitude.
    field_roundoff = 2 * np.finfo(float).eps
    if isinstance(coefficients, np.ndarray):
        np.fill_diagonal(coefficients, 0.0)
        if from_qubo:
            fields = coefficients[:, -1]
            couplings = coefficients[:, :-1]
            degree = np.count_nonzero(couplings, axis=1)
            tolerance = (couplings * field_roundoff).sum(axis=1) * (degree + 1)
            fields[fields <= tolerance] = 0.0
        min_coefficients = np.where(coefficients > 0, coefficients, np.inf).min(axis=1)
        scales = coefficients.max(axis=1)
        coefficients /= np.where(scales > 0, scales, 1.0)[:, np.newaxis]
        sigma = np.sqrt(np.sum(coefficients**2, axis=1))
    else:
        coefficients = coefficients.tocsr()
        coefficients.setdiag(0)
        if from_qubo:
            couplings = coefficients[:, :-1]
            couplings.eliminate_zeros()
            degree = np.diff(couplings.indptr)
            tolerance = np.asarray(couplings.multiply(field_roundoff).sum(axis=1)).ravel() * (degree + 1)
            rows = np.repeat(np.arange(n), np.diff(coefficients.indptr))
            is_field = coefficients.indices == n
            coefficients.data[is_field & (coefficients.data <= tolerance[rows])] = 0.0
        coefficients.eliminate_zeros()
        min_coefficients = np.full(n, np.inf)
        scales = np.zeros(n)
        row_sizes = np.diff(coefficients.indptr)
        nonempty = row_sizes > 0
        offsets = coefficients.indptr[:-1][nonempty]
        min_coefficients[nonempty] = np.minimum.reduceat(coefficients.data, offsets)
        scales[nonempty] = np.maximum.reduceat(coefficients.data, offsets)
        coefficients.data /= np.repeat(scales, row_sizes)
        sigma = np.sqrt(np.asarray(coefficients.multiply(coefficients).sum(axis=1)).ravel())

    return sigma, scales, min_coefficients[np.isfinite(min_coefficients)]


def _estimate_initial_beta(scaled_sigma, scales, quantile):
    with np.errstate(over="ignore"):
        sigma = scaled_sigma * scales
    if not np.isfinite(sigma).all():
        # Order fields in log space if their RMS exceeds the float range.
        with np.errstate(divide="ignore"):
            sigma = np.log(scaled_sigma) + np.log(scales)

    rank = quantile * (scales.size - 1)
    lower, upper = math.floor(rank), math.ceil(rank)
    indices = np.argpartition(sigma, (lower, upper))[[lower, upper]]
    scale = float(scales[indices].max())
    if scale == 0.0:
        # Isolated spins supply no temperature scale if both ranked fields are zero.
        active = scales > 0
        return _estimate_initial_beta(scaled_sigma[active], scales[active], quantile)

    low, high = scaled_sigma[indices] * (scales[indices] / scale)
    fraction = rank - lower
    sigma_quantile = (1 - fraction) * low + fraction * high
    return (math.log(2) / float(sigma_quantile)) / scale


def _geometric_schedule_segment(beta_start, beta_end, num_sweeps, sweeps_per_beta, *, skip_start=False):
    num_betas = (num_sweeps + sweeps_per_beta - 1) // sweeps_per_beta
    schedule = cxxjij.utility.make_classical_schedule_list(
        beta_min=beta_start,
        beta_max=beta_end,
        one_mc_step=sweeps_per_beta,
        num_call_updater=num_betas + skip_start,
    )
    if skip_start:
        schedule = schedule[1:]
    if schedule:
        schedule[-1].one_mc_step = num_sweeps - (num_betas - 1) * sweeps_per_beta
        if num_betas > 1 or skip_start:
            schedule[-1].updater_parameter.beta = beta_end
    return schedule


def geometric_ising_beta_schedule(
    cxxgraph: Union[openjij.cxxjij.graph.Dense, openjij.cxxjij.graph.CSRSparse],
    beta_max=None,
    beta_min=None,
    num_sweeps=1000,
    seed=None,
    *,
    from_qubo=False,
    use_quantiles=True,
):
    """Make a geometric cooling schedule with an optional colder tail.

    For single-spin updates, estimate the start as ``ln 2 / Q_0.9(sigma)``,
    where ``sigma_i = sqrt(h_i^2 + sum_j J_ij^2)``. The main interval ends
    at ``ln 100 / (2 Q_0.1(m))``, floored at ten times the start. Here
    ``m_i`` is the smallest nonzero coefficient touching spin ``i``.
    Spins with no such coefficient are excluded from the cold statistics.

    The tail endpoint is ``ln(N_min / 0.01) / (2 m_min)``. The minimum
    group of ``N_min`` spins is counted with relative tolerance 1e-9.
    When the tail endpoint is colder than the main interval, the final
    10 percent of sweeps cool geometrically to it, provided there is
    room for at least two sweeps in each interval. A supplied final
    endpoint sets the end of a single geometric interval and caps an
    automatically estimated starting beta.
    Automatic endpoints that exceed float range use the largest finite beta.

    The sampler omits fields within cancellation roundoff when it converts
    a BINARY model to Ising. For a model supplied in SPIN form, every
    nonzero coefficient contributes, including residue from an earlier
    conversion.

    Args:
        cxxgraph (Union[openjij.cxxjij.graph.Dense, openjij.cxxjij.graph.CSRSparse]): Ising graph, must be either `Dense` or `CSRSparse`.
        beta_max (float, optional): final inverse temperature. Estimated if None.
        beta_min (float, optional): initial inverse temperature. Estimated if None.
        num_sweeps (int, optional): number of sweeps. Defaults to 1000.
        seed (int, optional): unused. Kept for backward compatibility.
        from_qubo (bool, optional): ignore local fields within the roundoff
            of converting a binary model to Ising.
        use_quantiles (bool, optional): use the single-spin schedule.
            Cluster updates use the maximum RMS field and smallest
            nonzero coefficient for a single geometric interval.
    Returns:
        List of cxxjij.utility.ClassicalSchedule and the beta range
        [final beta, initial beta].
    """
    excitation_rate = 0.01
    max_beta = float(np.finfo(float).max)

    estimate_range = beta_min is None or beta_max is None
    beta_tail = None
    if estimate_range:
        scaled_sigma, scales, min_coefficients = _ising_beta_range_statistics(cxxgraph, from_qubo=from_qubo)
        if min_coefficients.size == 0:
            # All coefficients are zero: every state has the same energy.
            scaled_sigma, scales, min_coefficients = np.ones(1), np.ones(1), np.ones(1)

        if beta_min is None:
            limit = max_beta if beta_max is None else min(beta_max, max_beta)
            beta_min = min(_estimate_initial_beta(scaled_sigma, scales, 0.9 if use_quantiles else 1.0), limit)

        if beta_max is None:
            m_min = min_coefficients.min()
            num_min = np.count_nonzero(np.isclose(min_coefficients, m_min, rtol=1e-9, atol=0.0))
            beta_clean = (math.log(num_min / excitation_rate) / 2) / float(m_min)
            if use_quantiles:
                beta_max = (math.log(100) / 2) / float(np.quantile(min_coefficients, 0.1))
                if num_sweeps >= 4:
                    beta_tail = min(beta_clean, max_beta)
            else:
                beta_max = beta_clean
            beta_max = min(max(beta_max, beta_min * 10.0), max_beta)

    if use_quantiles and estimate_range:
        # Evaluate at every sweep through the benchmarked sweep range.
        # Group sweeps at larger budgets to bound the schedule size.
        max_temperature_steps = 10_000
        sweeps_per_beta = max(1, (num_sweeps + max_temperature_steps - 1) // max_temperature_steps)
        if beta_tail is not None and beta_tail > beta_max:
            tail_sweeps = max(2, round(num_sweeps / 10))
            schedule = _geometric_schedule_segment(beta_min, beta_max, num_sweeps - tail_sweeps, sweeps_per_beta)
            schedule.extend(
                _geometric_schedule_segment(beta_max, beta_tail, tail_sweeps, sweeps_per_beta, skip_start=True)
            )
            return schedule, [beta_tail, beta_min]
        schedule = _geometric_schedule_segment(beta_min, beta_max, num_sweeps, sweeps_per_beta)
        return schedule, [beta_max, beta_min]

    num_sweeps_per_beta = max(1, num_sweeps // 1000)
    # set schedule to cxxjij
    schedule = cxxjij.utility.make_classical_schedule_list(
        beta_min=beta_min,
        beta_max=beta_max,
        one_mc_step=num_sweeps_per_beta,
        num_call_updater=num_sweeps // num_sweeps_per_beta,
    )

    return schedule, [beta_max, beta_min]
