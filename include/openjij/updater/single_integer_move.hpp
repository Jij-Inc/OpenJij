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

struct MetropolisUpdater{
   template<typename SystemType>
   std::int64_t GenerateNewValue(SystemType &sa_system, const std::int64_t index, const double T, const double _progress) {
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

struct OptMetropolisUpdater{
   template<typename SystemType>
   std::int64_t GenerateNewValue(SystemType &sa_system, const std::int64_t index, const double T, const double progress) {
      if (dist(sa_system.random_number_engine) < progress) {
         // Metropolis Optimal Transition
         const auto [min_val, min_dE] = sa_system.GetMinEnergyDifference(index);
         if (min_dE <= 0.0 || dist(sa_system.random_number_engine) < std::exp(-min_dE / T)) {
            return min_val;
         } else {
            return sa_system.GetState()[index].value;
         }
      }
      else {
         const auto candidate_value = sa_system.GenerateCandidateValue(index);
         const double dE = sa_system.GetEnergyDifference(index, candidate_value);
         if (dE <= 0.0 || dist(sa_system.random_number_engine) < std::exp(-dE / T)) {
            return candidate_value;
         } else {
            return sa_system.GetState()[index].value;
         }
      }
   }
   
   std::uniform_real_distribution<double> dist{0.0, 1.0};
};


template<typename T>
struct is_integer_quadratic_model_system : std::false_type {};

template<typename RandType>
struct is_integer_quadratic_model_system<system::IntegerSASystem<graph::IntegerQuadraticModel, RandType>> : std::true_type {};

template<typename T>
inline constexpr bool is_integer_quadratic_model_system_v = is_integer_quadratic_model_system<T>::value;

struct HeatBathUpdater{
   template<typename SystemType>
   std::int64_t GenerateNewValue(SystemType &sa_system, const std::int64_t index, const double T, const double _progress) {
      if constexpr (is_integer_quadratic_model_system_v<SystemType>) {
         if (std::abs(sa_system.GetQuadCoeff()[index]) < 1e-12) {
            return ForBilinear(sa_system, index, T, _progress);
         }
         else {
            return ForAll(sa_system, index, T, _progress);
         }
      } else {
         return ForAll(sa_system, index, T, _progress);
      }
   }
   
   template<typename SystemType>
   std::int64_t ForAll(SystemType &sa_system, const std::int64_t index, const double T, const double _progress) {
      const auto &var = sa_system.GetState()[index];
      const double beta = 1.0 / T;
      std::int64_t selected_state_number = -1;
      double max_z = -std::numeric_limits<double>::infinity();
      
      for (std::int64_t i = 0; i < var.num_states; ++i) {
         const double g = -std::log(-std::log(dist(sa_system.random_number_engine)));
         const double z = -beta * sa_system.GetEnergyDifference(index, var.GetValueFromState(i)) + g;
         if (z > max_z) {
            max_z = z;
            selected_state_number = i;
         }
      }
      if (selected_state_number == -1) {
         throw std::runtime_error("No state selected.");
      }
      return var.GetValueFromState(selected_state_number);
   }
   
   template<typename RandType>
   std::int64_t ForBilinear(system::IntegerSASystem<graph::IntegerQuadraticModel, RandType> &sa_system, const std::int64_t index, const double T, const double _progress) {
      const auto &state = sa_system.GetState()[index];
      const double linear_coeff = sa_system.GetLinearCoeff()[index];
      
      if (std::abs(linear_coeff) < 1e-12) {
         return state.GenerateRandomValue(sa_system.random_number_engine);
      }
      
      const double b = linear_coeff;
      const std::int64_t dxl = state.lower_bound - state.value;
      const std::int64_t dxu = state.upper_bound - state.value;
      const double beta = 1.0 / T;
      
      auto p_sum = [&](std::int64_t u) -> double {
         if (b > 0) {
            const double a1 = std::exp(-beta * b * (u - dxl + 1));
            const double a2 = std::exp(-beta * b);
            return (1 - a1) / (1 - a2);
         } else {
            const double a1 = std::exp(beta * b * (u - dxl + 1));
            const double a2 = std::exp(beta * b);
            const double a3 = std::exp(-beta * b * (u - dxu));
            return a3 * (a1 - 1) / (a2 - 1);
         }
      };
      
      const double rand_value = dist(sa_system.random_number_engine) * p_sum(dxu);
      
      const std::int64_t n = state.num_states;
      std::int64_t low = 0;
      std::int64_t high = n - 1;
      std::int64_t selected_state_number = n;
      const std::int64_t current_value = state.value;
      
      while (low <= high) {
         const std::int64_t mid = low + (high - low) / 2;
         if (p_sum(state.GetValueFromState(mid) - current_value) > rand_value) {
            selected_state_number = mid;
            high = mid - 1;
         } else {
            low = mid + 1;
         }
      }
      
      if (selected_state_number == n) {
         selected_state_number = n - 1;
      }
      
      return state.GetValueFromState(selected_state_number);
   }
   
   std::uniform_real_distribution<double> dist{0.0, 1.0};
};


struct SuwaTodoUpdater {
   template<typename SystemType>
   std::int64_t GenerateNewValue(SystemType &sa_system, const std::int64_t index, const double T, const double _progress) {
      const auto &var = sa_system.GetState()[index];
      const std::int64_t max_num_state = var.num_states;
      
      std::vector<double> weight_list(max_num_state, 0.0);
      std::vector<double> sum_weight_list(max_num_state + 1, 0.0);
      std::vector<double> dE_list(max_num_state, 0.0);
      
      const auto [max_weight_state_value, min_dE] = sa_system.GetMinEnergyDifference(index);
      const std::int64_t max_weight_state = max_weight_state_value - var.lower_bound;
      
      for (std::int64_t state = 0; state < var.num_states; ++state) {
         const std::int64_t value = var.GetValueFromState(state);
         dE_list[state] = sa_system.GetEnergyDifference(index, value) - min_dE;
      }
      
      weight_list[0] = std::exp(-dE_list[max_weight_state] / T);
      sum_weight_list[1] = weight_list[0];
      
      for (std::int64_t state = 1; state < var.num_states; ++state) {
         if (state == max_weight_state) {
            weight_list[state] = std::exp(-dE_list[0] / T);
         } else {
            weight_list[state] = std::exp(-dE_list[state] / T);
         }
         sum_weight_list[state + 1] = sum_weight_list[state] + weight_list[state];
      }
      
      sum_weight_list[0] = sum_weight_list[var.num_states];
      const std::int64_t current_state = var.value - var.lower_bound;
      std::int64_t now_state;
      if (current_state == 0) {
         now_state = max_weight_state;
      } else if (current_state == max_weight_state) {
         now_state = 0;
      } else {
         now_state = current_state;
      }
      
      double prob_sum = 0.0;
      const double rand = dist(sa_system.random_number_engine);
      
      for (std::int64_t j = 0; j < var.num_states; ++j) {
         const double d_ij = sum_weight_list[now_state + 1] - sum_weight_list[j] + sum_weight_list[1];
         prob_sum += std::max(0.0, std::min({d_ij, 1.0 + weight_list[j] - d_ij, 1.0, weight_list[j]}));
         if (rand < prob_sum) {
            std::int64_t new_state;
            if (j == max_weight_state) {
               new_state = 0;
            } else if (j == 0) {
               new_state = max_weight_state;
            } else {
               new_state = j;
            }
            return var.GetValueFromState(new_state);
         }
      }
      
      return var.GetValueFromState(var.num_states - 1);
   }
   
   std::uniform_real_distribution<double> dist{0.0, 1.0};
};

}
}

