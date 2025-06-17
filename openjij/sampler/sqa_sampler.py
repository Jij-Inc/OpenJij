from __future__ import annotations
try:
    from typing import Optional, Union
except ImportError:
    from typing_extensions import Optional, Union

import dimod
import numpy as np

from cimod.utils import get_state_and_energy
from dimod import BINARY

import openjij
import openjij as oj
import openjij.cxxjij as cxxjij

from openjij.sampler.sampler import BaseSampler
from openjij.utils.decorator import deprecated_alias


class SQASampler(BaseSampler):
    """Sampler with Simulated Quantum Annealing (SQA).

    Inherits from :class:`openjij.sampler.sampler.BaseSampler`.
    Hamiltonian

    .. math::

        H(s) = s H_p + \\Gamma (1-s)\\sum_i \\sigma_i^x

    where :math:`H_p` is the problem Hamiltonian we want to solve.

    Args:
        beta (float): Inverse temperature.
        gamma (float): Amplitude of quantum fluctuation.
        trotter (int): Trotter number.
        num_sweeps (int): number of sweeps
        schedule (list): schedule list
        num_reads (int): Number of iterations.
        schedule_info (dict): Information about a annealing schedule.

    Raises:
        ValueError: If the schedule violates as below.
        - not list or numpy.array.
        - schedule range is '0 <= s <= 1'.
    """

    @property
    def parameters(self):
        return {
            "beta": ["parameters"],
            "gamma": ["parameters"],
            "trotter": ["parameters"],
        }

    def __init__(self):

        # Set default parameters
        beta = None
        gamma = None
        num_sweeps = 1000
        num_reads = 1
        schedule = None
        trotter = 4

        self._default_params = {
            "beta": beta,
            "gamma": gamma,
            "num_sweeps": num_sweeps,
            "schedule": schedule,
            "trotter": trotter,
            "num_reads": num_reads,
        }

        self._params = {
            "beta": beta,
            "gamma": gamma,
            "num_sweeps": num_sweeps,
            "schedule": schedule,
            "trotter": trotter,
            "num_reads": num_reads,
        }

        self._make_system = {"singlespinflip": cxxjij.system.make_transverse_ising}
        self._algorithm = {
            "singlespinflip": cxxjij.algorithm.Algorithm_SingleSpinFlip_run
        }

    def _convert_validation_schedule(self, schedule, beta):
        if not isinstance(schedule, (list, np.ndarray)):
            raise ValueError("schedule should be list or numpy.array")

        if isinstance(schedule[0], cxxjij.utility.TransverseFieldSchedule):
            return schedule

        # schedule validation  0 <= s <= 1
        sch = np.array(schedule).T[0]
        if not np.all((0 <= sch) & (sch <= 1)):
            raise ValueError("schedule range is '0 <= s <= 1'.")

        if len(schedule[0]) == 2:
            # schedule element: (s, one_mc_step) with beta fixed
            # convert to list of cxxjij.utility.TransverseFieldSchedule
            cxxjij_schedule = []
            for s, one_mc_step in schedule:
                _schedule = cxxjij.utility.TransverseFieldSchedule()
                _schedule.one_mc_step = one_mc_step
                _schedule.updater_parameter.beta = beta
                _schedule.updater_parameter.s = s
                cxxjij_schedule.append(_schedule)
            return cxxjij_schedule
        elif len(schedule[0]) == 3:
            # schedule element: (s, beta, one_mc_step)
            # convert to list of cxxjij.utility.TransverseFieldSchedule
            cxxjij_schedule = []
            for s, _beta, one_mc_step in schedule:
                _schedule = cxxjij.utility.TransverseFieldSchedule()
                _schedule.one_mc_step = one_mc_step
                _schedule.updater_parameter.beta = _beta
                _schedule.updater_parameter.s = s
                cxxjij_schedule.append(_schedule)
            return cxxjij_schedule
        else:
            raise ValueError(
                """schedule is list of tuple or list
                (annealing parameter s : float, step_length : int) or
                (annealing parameter s : float, beta: float, step_length : int)
                """
            )

    def _get_result(self, system, model):
        state, info = super()._get_result(system, model)

        q_state = system.trotter_spins[:-1].T.astype(int)
        c_energies = [get_state_and_energy(model, state)[1] for state in q_state]
        info["trotter_state"] = q_state
        info["trotter_energies"] = c_energies

        return state, info

    def sample(
        self,
        bqm: Union[
            "openjij.model.model.BinaryQuadraticModel", dimod.BinaryQuadraticModel
        ],
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        num_sweeps: Optional[int] = None,
        schedule: Optional[list] = None,
        trotter: Optional[int] = None,
        num_reads: Optional[int] = None,
        initial_state: Optional[Union[list, dict]] = None,
        updater: Optional[str] = None,
        sparse: Optional[bool] = None,
        reinitialize_state: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> "openjij.sampler.response.Response":
        """Sampling from the Ising model.

        Args:
            bqm (openjij.BinaryQuadraticModel) binary quadratic model
            beta (float, optional): inverse tempareture.
            gamma (float, optional): strangth of transverse field. Defaults to None.
            num_sweeps (int, optional): number of sweeps. Defaults to None.
            schedule (list[list[float, int]], optional): List of annealing parameter. Defaults to None.
            trotter (int): Trotter number.
            num_reads (int, optional): number of sampling. Defaults to 1.
            initial_state (list[int], optional): Initial state. Defaults to None.
            updater (str, optional): update method. Defaults to 'single spin flip'.
            sparse (bool): use sparse matrix or not.
            reinitialize_state (bool, optional): Re-initilization at each sampling. Defaults to True.
            seed (int, optional): Sampling seed. Defaults to None.

        Raises:
            ValueError:

        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples:

            for Ising case::

                >>> h = {0: -1, 1: -1, 2: 1, 3: 1}
                >>> J = {(0, 1): -1, (3, 4): -1}
                >>> sampler = openjij.SQASampler()
                >>> res = sampler.sample_ising(h, J)

            for QUBO case::

                >>> Q = {(0, 0): -1, (1, 1): -1, (2, 2): 1, (3, 3): 1, (4, 4): 1, (0, 1): -1, (3, 4): 1}
                >>> sampler = openjij.SQASampler()
                >>> res = sampler.sample_qubo(Q)
        """

        # Set default parameters
        if sparse is None:
            sparse = True
        if reinitialize_state is None:
            reinitialize_state = True
        if updater is None:
            updater = "single spin flip"

        if isinstance(bqm, dimod.BinaryQuadraticModel):
            bqm = oj.model.model.BinaryQuadraticModel(
                dict(bqm.linear), dict(bqm.quadratic), bqm.offset, bqm.vartype
            )

        ising_graph, offset = bqm.get_cxxjij_ising_graph()

        self._set_params(
            beta=beta,
            gamma=gamma,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            trotter=trotter,
            schedule=schedule,
        )

        # set annealing schedule -------------------------------
        self._annealing_schedule_setting(
            model=bqm,
            ising_graph=ising_graph,
            trotter=self._params["trotter"],
            beta=self._params["beta"],
            gamma=self._params["gamma"],
            num_sweeps=self._params["num_sweeps"],
            schedule=self._params["schedule"],
        )
        # ------------------------------- set annealing schedule

        # make init state generator --------------------------------
        if initial_state is None:

            def init_generator():
                return [
                    ising_graph.gen_spin(seed)
                    if seed is not None
                    else ising_graph.gen_spin()
                    for _ in range(self._params["trotter"])
                ]

        else:
            if isinstance(initial_state, dict):
                initial_state = [initial_state[k] for k in bqm.variables]

            # convert to spin variable
            if bqm.vartype == BINARY:
                temp_initial_state = []
                for v in initial_state:
                    if v != 0 and v != 1:
                        raise RuntimeError(
                            "The initial variables must be 0 or 1 if vartype is BINARY."
                        )
                    temp_initial_state.append(2 * v - 1)

                initial_state = temp_initial_state

            _init_state = np.array(initial_state)

            # validate initial_state size
            if len(initial_state) != ising_graph.size():
                raise ValueError(
                    "the size of the initial state should be {}".format(
                        ising_graph.size()
                    )
                )

            trotter_init_state = [_init_state for _ in range(self._params["trotter"])]

            def init_generator():
                return trotter_init_state

        # -------------------------------- make init state generator

        # choose updater -------------------------------------------
        _updater_name = updater.lower().replace("_", "").replace(" ", "")
        if _updater_name not in self._algorithm:
            raise ValueError('updater is one of "single spin flip"')
        algorithm = self._algorithm[_updater_name]
        sqa_system = self._make_system[_updater_name](
            init_generator(), ising_graph, self._params["gamma"]
        )
        # ------------------------------------------- choose updater

        response = self._cxxjij_sampling(
            bqm, init_generator, algorithm, sqa_system, reinitialize_state, seed
        )

        response.info["schedule"] = self.schedule_info

        return response

    def _annealing_schedule_setting(
        self, model, ising_graph, trotter: int, beta=None, gamma=None, num_sweeps=None, schedule=None
    ):
        beta_min, beta_max, gamma = estimate_beta_schedule(ising_graph, beta, gamma, trotter=trotter)
        self._params["gamma"] = gamma
        self._params["beta"] = beta_max
        if schedule:
            self._params["schedule"] = self._convert_validation_schedule(schedule, beta_max)
            self.schedule_info = {"schedule": "custom schedule"}
        else:
            self._params["schedule"], beta_gamma = quartic_ising_schedule(
                model=model, beta_min=beta_min, beta_max=beta_max, gamma=gamma, num_sweeps=num_sweeps
            )
            self.schedule_info = {
                "beta": beta_gamma[0],
                "gamma": beta_gamma[1],
                "num_sweeps": num_sweeps,
            }


def linear_ising_schedule(model, beta, gamma, num_sweeps):
    """Generate linear ising schedule.

    Args:
        model (:class:`openjij.model.model.BinaryQuadraticModel`): BinaryQuadraticModel
        beta (float): inverse temperature
        gamma (float): transverse field
        num_sweeps (int): number of steps
    Returns:
        generated schedule
    """
    schedule = cxxjij.utility.make_transverse_field_schedule_list(
        beta=beta, one_mc_step=1, num_call_updater=num_sweeps
    )
    return schedule, [beta, gamma]


# TODO: more optimal schedule?


def quartic_ising_schedule(model, beta_min, beta_max, gamma, num_sweeps):
    """Generate quartic ising schedule based on S

    Morita and H. Nishimori,
    Journal of Mathematical Physics 49, 125210 (2008).

    Args:
        model (:class:`openjij.model.model.BinaryQuadraticModel`): BinaryQuadraticModel
        beta (float): inverse temperature
        gamma (float): transverse field
        num_sweeps (int): number of steps
    Returns:
        generated schedule
    """
    beta_geo = np.geomspace(beta_min, beta_max, num_sweeps).tolist()
    s = np.linspace(0, 1, num_sweeps)
    # fs = s**4 * (35 - 84 * s + 70 * s**2 - 20 * s**3)
    n = 10
    fs = 1/2 + (2*s - 1) / (2 * np.sqrt(n - (n-1)*((2*s-1)**2)))
    schedule = [((elem, beta_geo[i]), 1) for i, elem in enumerate(fs)]
    return schedule, [beta_max, gamma]


def estimate_beta_schedule(
    cxxgraph: Union[openjij.cxxjij.graph.Dense, openjij.cxxjij.graph.CSRSparse],
    beta: Optional[float],
    gamma: Optional[float],
    trotter: int,
) -> tuple[float, float, float]:
    linear_term_dE: float = 1.0
    min_delta_energy = 1.0
    max_delta_energy = 1.0

    # generate Ising matrix (with symmetric form)
    ising_interaction = cxxgraph.get_interactions()
    n = ising_interaction.shape[0]  # n+1

    if beta is not None:
        if gamma is None:
            p = min(10.0 / n, 0.1)
            _gamma = np.arctanh(p) / (2*beta)
            return (beta*10.0, beta, _gamma)
        return (beta*10.0, beta, gamma)
    else:
        # if `abs_ising_interaction` is empty, set min/max delta_energy to 1 (a trivial case).
        if ising_interaction.shape[0] <= 1:
            min_delta_energy = 1
            max_delta_energy = 1
            linear_term_dE = 1
        else:
            random_spin = np.random.choice([-1, 1], size=(ising_interaction.shape[0], 2))
            random_spin[-1, :] = 1  # last element is bias term
            # calculate delta energy
            abs_dE = np.abs((2 * ising_interaction @ random_spin * (-2*random_spin)))

            # Check linear term energy difference
            linear_term_dE = abs_dE[-1, :].mean()  # last row corresponds to linear term

            abs_dE = abs_dE[:-1, :]  # remove the last element (bias term)

            # apply threshold to avoid extremely large beta_max
            THRESHOLD = 1e-8
            abs_dE = abs_dE[abs_dE >= THRESHOLD]
            if len(abs_dE) == 0:
                min_delta_energy = 1
                max_delta_energy = 1
            else:
                # apply threshold to avoid extremely large beta_max
                min_delta_energy = np.min(abs_dE, axis=0).mean()
                max_delta_energy = np.mean(abs_dE, axis=0).mean()


    # Same logic in SA
    prob_inv = max(n / 1, 100)
    beta_max = np.log(prob_inv) / min_delta_energy

    # 10 times heat flip accept for 1 sweep in the initial state.
    prob_inv = max(n / 10, 2)
    beta_min = np.log(prob_inv) / max_delta_energy
    if linear_term_dE / max_delta_energy > 100:
        # Fast cooling mode
        beta_min = max(beta_max / 100, beta_min)

    if gamma is None:
        p = min(10.0 / n, 0.1)
        _gamma = np.arctanh(p) / (2*beta_min)
        return (beta_min, beta_max, _gamma)

    beta_min *= trotter
    beta_max *= trotter*10

    return (beta_min, beta_max, gamma)

