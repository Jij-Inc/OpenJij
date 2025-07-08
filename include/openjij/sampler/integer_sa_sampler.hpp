//    Copyright 2023 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#include "openjij/graph/all.hpp"
#include "openjij/system/all.hpp"
#include "openjij/updater/all.hpp"

namespace openjij {
namespace sampler {

struct IntegerSAResult {
  double energy = 0.0;
  std::vector<std::int64_t> solution = {};
  std::vector<double> energy_history = {};
  std::vector<double> temperature_history = {};
};

template <class ModelType, class RandType, class StateUpdater>
IntegerSAResult
BaseSA(const ModelType &model, const utility::TemperatureSchedule schedule,
       const std::int64_t num_sweeps, const typename RandType::result_type seed,
       const double min_T, const double max_T, const bool log_history) {

  // Initialize the system
  system::IntegerSASystem<ModelType, RandType> sa_system(model, seed);

  // Initialize the updater
  auto state_updater = StateUpdater{};

  auto get_T = [&](const std::int64_t sweep) {
    if (schedule == utility::TemperatureSchedule::LINEAR) {
      return max_T +
             (min_T - max_T) * (static_cast<double>(sweep) / (num_sweeps - 1));
    } else if (schedule == utility::TemperatureSchedule::GEOMETRIC) {
      return max_T * std::pow(min_T / max_T,
                              static_cast<double>(sweep) / (num_sweeps - 1));
    } else {
      throw std::runtime_error("Unknown temperature schedule");
    }
  };

  const std::int64_t num_variables = model.GetNumVariables();
  IntegerSAResult result;

  for (std::int64_t sweep = 0; sweep < num_sweeps; ++sweep) {
    const double T = get_T(sweep);
    const double progress = static_cast<double>(sweep) / (num_sweeps - 1);
    for (std::int64_t i = 0; i < num_variables; ++i) {
      const auto new_x =
          state_updater.GenerateNewValue(sa_system, i, T, progress);
      sa_system.SetValue(i, new_x);
    }
    if (log_history) {
      result.energy_history.push_back(sa_system.GetEnergy());
      result.temperature_history.push_back(T);
    }
  }

  result.energy = sa_system.GetEnergy();
  result.solution.resize(num_variables);
  const auto &state = sa_system.GetState();
  for (std::int64_t i = 0; i < num_variables; ++i) {
    result.solution[i] = state[i].value;
  }

  return result;
}

template <class ModelType, class UpdaterType>
IntegerSAResult SolveByIntegerSAImpl(const ModelType &model,
                                     const std::int64_t num_sweeps,
                                     const algorithm::RandomNumberEngine rand_type,
                                     const utility::TemperatureSchedule schedule,
                                     const std::int64_t seed, const double min_T,
                                     const double max_T, const bool log_history) {
  switch (rand_type) {
  case algorithm::RandomNumberEngine::XORSHIFT:
    return BaseSA<ModelType, utility::Xorshift, UpdaterType>(
        model, schedule, num_sweeps,
        static_cast<utility::Xorshift::result_type>(seed), min_T, max_T,
        log_history);
  case algorithm::RandomNumberEngine::MT:
    return BaseSA<ModelType, std::mt19937, UpdaterType>(
        model, schedule, num_sweeps,
        static_cast<std::mt19937::result_type>(seed), min_T, max_T,
        log_history);
  case algorithm::RandomNumberEngine::MT_64:
    return BaseSA<ModelType, std::mt19937_64, UpdaterType>(
        model, schedule, num_sweeps, seed, min_T, max_T, log_history);
  default:
    throw std::runtime_error("Unknown random number engine");
  }
}

template <class ModelType>
IntegerSAResult SolveByIntegerSA(const ModelType &model,
                                 const std::int64_t num_sweeps,
                                 const algorithm::UpdateMethod update_method,
                                 const algorithm::RandomNumberEngine rand_type,
                                 const utility::TemperatureSchedule schedule,
                                 const std::int64_t seed, const double min_T,
                                 const double max_T, const bool log_history) {

  switch (update_method) {
  case algorithm::UpdateMethod::METROPOLIS:
    return SolveByIntegerSAImpl<ModelType, updater::MetropolisUpdater>(
        model, num_sweeps, rand_type, schedule, seed, min_T, max_T, log_history);
  case algorithm::UpdateMethod::HEAT_BATH:
    return SolveByIntegerSAImpl<ModelType, updater::HeatBathUpdater>(
        model, num_sweeps, rand_type, schedule, seed, min_T, max_T, log_history);
  case algorithm::UpdateMethod::SUWA_TODO:
    return SolveByIntegerSAImpl<ModelType, updater::SuwaTodoUpdater>(
        model, num_sweeps, rand_type, schedule, seed, min_T, max_T, log_history);
  case algorithm::UpdateMethod::OPT_METROPOLIS:
    return SolveByIntegerSAImpl<ModelType, updater::OptMetropolisUpdater>(
        model, num_sweeps, rand_type, schedule, seed, min_T, max_T, log_history);
  default:
    throw std::runtime_error("Unknown update method");
  }
}

template <class ModelType>
std::vector<IntegerSAResult>
SampleByIntegerSA(const ModelType &model, const std::int64_t num_sweeps,
                  const algorithm::UpdateMethod update_method,
                  const algorithm::RandomNumberEngine rand_type,
                  const utility::TemperatureSchedule schedule,
                  const std::int64_t num_reads, const std::int64_t seed,
                  const std::int32_t num_threads, const double min_T,
                  const double max_T, const bool log_history) {

  std::vector<IntegerSAResult> results(num_reads);

#pragma omp parallel for schedule(guided) num_threads(num_threads)
  for (std::int64_t i = 0; i < num_reads; ++i) {
    results[i] =
        SolveByIntegerSA(model, num_sweeps, update_method, rand_type, schedule,
                         seed + i, min_T, max_T, log_history);
  }

  return results;
}

} // namespace sampler
} // namespace openjij
