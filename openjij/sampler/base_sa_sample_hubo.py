from __future__ import annotations
try:
    from typing import Optional, Union
except ImportError:
    from typing_extensions import Optional, Union

import time
import multiprocessing
import os
from openjij.sampler.response import Response
from openjij.variable_type import BINARY, SPIN
from openjij.cxxjij.graph import (
    BinaryPolynomialModel,
    IsingPolynomialModel
)
from openjij.cxxjij.sampler import make_sa_sampler
from openjij.utils.cxx_cast import (
    cast_to_cxx_update_method,
    cast_to_cxx_random_number_engine,
    cast_to_cxx_temperature_schedule
)

def to_oj_response(
    variables: list[list[Union[int, float]]], 
    index_list: list[Union[int, str, tuple[int, ...]]],
    energies: list[float], 
    vartype: str
) -> Response:
    return Response.from_samples(
        samples_like=[dict(zip(index_list, v_list)) for v_list in variables], 
        vartype=vartype, 
        energy=energies
    )  

def _get_default_num_threads():
    """利用可能なCPUコア数を取得してデフォルト値として返す
    
    環境変数での制御:
    - OPENJIJ_USE_MULTIPROCESSING=1: CPUコア数を使用
    - OPENJIJ_NUM_THREADS: 明示的なスレッド数指定
    - OMP_NUM_THREADS: OpenMP標準のスレッド数指定
    """
    try:
        # 環境変数での制御も可能にする
        if 'OPENJIJ_NUM_THREADS' in os.environ:
            return int(os.environ['OPENJIJ_NUM_THREADS'])
        if 'OMP_NUM_THREADS' in os.environ:
            return int(os.environ['OMP_NUM_THREADS'])
        
        # OPENJIJ_USE_MULTIPROCESSINGが設定されている場合のみCPUコア数を使用
        if os.environ.get('OPENJIJ_USE_MULTIPROCESSING', '').lower() in ('1', 'true', 'yes'):
            cpu_count = multiprocessing.cpu_count()
            return cpu_count
        
        # デフォルトは1（下位互換性のため）
        return 1
    except:
        # 何らかのエラーが発生した場合は1を返す
        return 1

def base_sample_hubo(
    hubo: dict[tuple, float],
    vartype: Optional[str] = None,
    num_sweeps: int = 1000,
    num_reads: Optional[int] = None,
    num_threads: Optional[int] = None,
    beta_min: Optional[float] = None,
    beta_max: Optional[float] = None,
    update_method: str = "METROPOLIS",
    random_number_engine: str = "XORSHIFT",
    seed: Optional[int] = None,
    temperature_schedule: str = "GEOMETRIC",
    local_search: bool = False,
) -> Response:
    
    start_time = time.time()

    # Set default values for num_threads and num_reads if not provided
    if num_threads is None:
        num_threads = _get_default_num_threads()
    
    if num_reads is None:
        num_reads = num_threads
    
    # Ensure num_reads is at least 1
    if num_reads < 1:
        num_reads = 1

    # Define cxx_sampler and set parameters
    start_define_sampler = time.time()
    if vartype in ("BINARY", BINARY):
        sampler = make_sa_sampler(
            BinaryPolynomialModel(
                key_list=list(hubo.keys()), 
                value_list=list(hubo.values())
            )
        )
    elif vartype in ("SPIN", SPIN):
        sampler = make_sa_sampler(
            IsingPolynomialModel(
                key_list=list(hubo.keys()), 
                value_list=list(hubo.values())
            )
        )
    else:
        raise ValueError("vartype must `BINARY` or `SPIN`")


    sampler.set_num_sweeps(num_sweeps=num_sweeps)
    sampler.set_num_reads(num_reads=num_reads)
    sampler.set_num_threads(num_threads=num_threads)
    sampler.set_use_local_search(use_local_search=local_search)
    sampler.set_update_method(
        update_method=cast_to_cxx_update_method(update_method)
    )
    sampler.set_random_number_engine(
        random_number_engine=cast_to_cxx_random_number_engine(random_number_engine)
    )
    sampler.set_temperature_schedule(
        temperature_schedule=cast_to_cxx_temperature_schedule(temperature_schedule)
    )

    if beta_min is not None:
        sampler.set_beta_min(beta_min=beta_min)
    else:
        sampler.set_beta_min_auto()

    if beta_max is not None:
        sampler.set_beta_max(beta_max=beta_max)
    else:
        sampler.set_beta_max_auto()
    define_sampler_time = time.time() - start_define_sampler

    # Start sampling
    start_sample = time.time()
    if seed is not None:
        sampler.sample(seed=seed)
    else:
        sampler.sample()
    sample_time = time.time() - start_sample

    # Make openjij response
    start_make_oj_response = time.time()
    response = to_oj_response(
        sampler.get_samples(), 
        sampler.get_index_list(),
        sampler.calculate_energies(),
        vartype
    )
    make_oj_response_time = time.time() - start_make_oj_response

    response.info["schedule"] = {
        "num_sweeps": num_sweeps,
        "num_reads": num_reads,
        "num_threads": num_threads,
        "beta_min": sampler.get_beta_min(),
        "beta_max": sampler.get_beta_max(),
        "update_method": update_method,
        "random_number_engine": random_number_engine,
        "temperature_schedule": temperature_schedule,
        "seed": sampler.get_seed(),
    }

    # Keep it in for backward compatibility.
    response.info["sampling_time"] = (sample_time + define_sampler_time)*10**6  # micro sec
    response.info["execution_time"] = (sample_time/num_reads)*10**6  # micro sec

    response.info["time"] = {
        "define_cxx_sampler": define_sampler_time,
        "sample": sample_time,
        "make_oj_response": make_oj_response_time,
        "total": time.time() - start_time,
    }

    return response

