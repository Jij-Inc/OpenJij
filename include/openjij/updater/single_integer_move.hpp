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

#include <random>

namespace openjij {
namespace updater {

struct MetropolisUpdater {
  template <typename SystemType>
  std::int64_t GenerateNewValue(SystemType &sa_system, const std::int64_t index,
                                const double T, const double _progress) {
    const auto candidate_value = sa_system.GenerateCandidateValue(index);
    const double dE = sa_system.GetEnergyDifference(index, candidate_value);
    if (dE <= 0.0 || dist(sa_system.random_number_engine) < std::exp(-dE / T)) {
      return candidate_value;
    } else {
      return sa_system.GetState()[index].value;
    }
  }

  std::uniform_real_distribution<double> dist{0.0, 1.0};
};

struct OptMetropolisUpdater {
  template <typename SystemType>
  std::int64_t GenerateNewValue(SystemType &sa_system, const std::int64_t index,
                                const double T, const double progress) {
    // Metropolis Optimal Transition if possible
    // This is used for systems with only quadratic coefficientsa
    if (sa_system.UnderQuadraticCoeff(index) && dist(sa_system.random_number_engine) < progress) {
      const auto [min_val, min_dE] = sa_system.GetMinEnergyDifference(index);
      if (min_dE <= 0.0 ||
          dist(sa_system.random_number_engine) < std::exp(-min_dE / T)) {
        return min_val;
      } else {
        return sa_system.GetState()[index].value;
      }
    } else {
      const auto candidate_value = sa_system.GenerateCandidateValue(index);
      const double dE = sa_system.GetEnergyDifference(index, candidate_value);
      if (dE <= 0.0 ||
          dist(sa_system.random_number_engine) < std::exp(-dE / T)) {
        return candidate_value;
      } else {
        return sa_system.GetState()[index].value;
      }
    }
  }

  std::uniform_real_distribution<double> dist{0.0, 1.0};
};

struct HeatBathUpdater {
  template <typename SystemType>
  std::int64_t GenerateNewValue(SystemType &sa_system, const std::int64_t index,
                                const double T, const double _progress) {
    if (sa_system.OnlyMultiLinearCoeff(index)) {
      return ForBilinear(sa_system, index, T, _progress);
    } else {
      return ForAll(sa_system, index, T, _progress);
    }
  }

  template <typename SystemType>
  std::int64_t ForAll(SystemType &sa_system, const std::int64_t index,
                      const double T, const double _progress) {

    const auto [max_weight_state_value, min_dE] = sa_system.GetMinEnergyDifference(index);
    const auto &var = sa_system.GetState()[index];
    const double beta = 1.0 / T;
    double z = 0.0;

    // Calculate the partition function
    for (std::int64_t i = 0; i < var.num_states; ++i) {
      const double dE = sa_system.GetEnergyDifference(index, var.GetValueFromState(i)) - min_dE;
      z += std::exp(-beta * dE);
    }

    // Select a state based on the partition function
    std::int64_t selected_state_number = -1;
    double cumulative_prob = 0.0;
    const double rand = dist(sa_system.random_number_engine) * z;

    for (std::int64_t i = 0; i < var.num_states; ++i) {
      const double dE = sa_system.GetEnergyDifference(index, var.GetValueFromState(i)) - min_dE;
      cumulative_prob += std::exp(-beta * dE);
      if (rand <= cumulative_prob) {
        selected_state_number = i;
        break;
      }
    }

    if (selected_state_number == -1) {
      throw std::runtime_error("No state selected.");
    }

    return var.GetValueFromState(selected_state_number);
  }

  template <typename SystemType>
  std::int64_t ForBilinear(SystemType &sa_system,
                          const std::int64_t index, const double T,
                          const double _progress) {
      const auto &state = sa_system.GetState()[index];
      const double linear_coeff = sa_system.GetLinearCoeff(index);

      if (std::abs(linear_coeff) < 1e-10) {
          return state.GenerateRandomValue(sa_system.random_number_engine);
      }

      const double b = -linear_coeff * (1.0 / T);
      const double dxl = static_cast<double>(state.lower_bound - state.value);
      const double dxu = static_cast<double>(state.upper_bound - state.value);

      const double u = this->dist(sa_system.random_number_engine);

      double selected_dz = 0.0;
      if (b > 0) {
          selected_dz = dxu + std::log(u + (1.0 - u) * std::exp(-b * (dxu - dxl + 1))) / b;
      } else {
          selected_dz = dxl - 1.0 + std::log(1.0 - u * (1.0 - std::exp(b * (dxu - dxl + 1)))) / b;
      }

      selected_dz = static_cast<std::int64_t>(std::ceil(std::max(dxl, std::min(selected_dz, dxu))));
      
      return state.value + selected_dz;
  }

  std::uniform_real_distribution<double> dist{0.0, 1.0};
};

struct SuwaTodoUpdater {
  template <typename SystemType>
  std::int64_t GenerateNewValue(SystemType &sa_system, const std::int64_t index,
                                const double T, const double _progress) {
    const auto &var = sa_system.GetState()[index];
    const std::int64_t max_num_state = var.num_states;
    std::vector<double> weight_list(max_num_state, 0.0);
    std::vector<double> sum_weight_list(max_num_state, 0.0);

    const auto [max_weight_state_value, min_dE] = sa_system.GetMinEnergyDifference(index);
    const auto max_weight_state = var.GetStateFromValue(max_weight_state_value);

    for (std::int64_t i = 0; i < max_num_state; ++i) {
      const std::int64_t state = (i == 0) ? max_weight_state : ((i == max_weight_state) ? 0 : i);
      const double dE = sa_system.GetEnergyDifference(index, var.GetValueFromState(state)) - min_dE;
      weight_list[i] = std::exp(-dE / T);
      sum_weight_list[i] = (i == 0) ? weight_list[i] : sum_weight_list[i - 1] + weight_list[i];
    }

    std::int64_t current_state = var.GetStateFromValue(var.value);
    if (current_state == 0) {
      current_state = max_weight_state;
    } else if (current_state == max_weight_state) {
      current_state = 0;
    }

    const double w_0 = weight_list[0];
    const double w_c = weight_list[current_state];
    const double sum_w_c = sum_weight_list[current_state];
    const double rand = dist(sa_system.random_number_engine) * w_c;
    std::int64_t selected_state = -1;
    double prob_sum = 0.0;

    for (std::int64_t j = 0; j < max_num_state; ++j) {
      const double d_ij = sum_w_c - sum_weight_list[(j - 1 + max_num_state) % max_num_state] + w_0;
      prob_sum += std::max(0.0, std::min({d_ij, w_c + (weight_list[j] - d_ij), w_c, weight_list[j]}));
      if (rand <= prob_sum) {
        if (j == 0) {
          selected_state = max_weight_state;
        } else if (j == max_weight_state) {
          selected_state = 0;
        } else {
          selected_state = j;
        }
        break;
      }
    }

    if (selected_state == -1) {
      throw std::runtime_error("No state selected.");
    }

    return var.GetValueFromState(selected_state);
  }

  std::uniform_real_distribution<double> dist{0.0, 1.0};
};

} // namespace updater
} // namespace openjij
