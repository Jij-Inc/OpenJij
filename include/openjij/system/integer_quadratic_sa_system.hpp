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

#include "../utility/variable.hpp"
#include "./sa_system.hpp"
#include <cstdint>
#include <random>
#include <vector>

namespace openjij {
namespace system {

template <typename RandType>
class IntegerSASystem<graph::IntegerQuadraticModel, RandType> {

public:
  IntegerSASystem(const graph::IntegerQuadraticModel &model,
                  const typename RandType::result_type seed)
      : model(model), seed(seed), random_number_engine(seed) {

    const std::int64_t num_variables = model.GetNumVariables();

    // Initialize the state with random values
    this->state_.reserve(num_variables);
    const auto &bounds = this->model.GetBounds();
    for (std::int64_t i = 0; i < num_variables; ++i) {
      auto x = utility::IntegerVariable(bounds[i].first, bounds[i].second);
      x.SetRandomValue(this->random_number_engine);
      this->state_.push_back(x);
    }
    this->energy_ = this->CalculateEnergy(this->state_);

    // Precompute the coefficients for energy difference when updating a state
    const auto &squared = this->model.GetSquared();
    this->quad_coeff_ = squared;
    this->linear_coeff_.resize(num_variables, 0.0);

    const auto &linear = this->model.GetLinear();
    const auto &quadratic = this->model.GetQuadratic();

    for (std::int64_t row = 0; row < num_variables; ++row) {
      double dE = linear[row] + 2.0 * squared[row] * this->state_[row].value;
      for (const auto &[col, value] : quadratic[row]) {
        dE += value * this->state_[col].value;
      }
      this->linear_coeff_[row] = dE;
    }
  }

  double
  CalculateEnergy(const std::vector<utility::IntegerVariable> &state) const {
    double energy = this->model.GetConstant();
    const auto &linear = this->model.GetLinear();
    const auto &squared = this->model.GetSquared();
    const auto &quadratic = this->model.GetQuadratic();
    const std::int64_t num_variables = this->model.GetNumVariables();

    for (std::int64_t i = 0; i < num_variables; ++i) {
      auto x = state[i].value;
      energy += linear[i] * x + squared[i] * x * x;
      for (const auto &[j, value] : quadratic[i]) {
        auto y = state[j].value;
        energy += 0.5 * value * x * y;
      }
    }
    return energy;
  }

  std::int64_t GenerateCandidateValue(std::int64_t index) {
    return this->state_[index].GenerateCandidateValue(
        this->random_number_engine);
  }

  double GetEnergyDifference(std::int64_t index, std::int64_t new_value) const {
    std::int64_t d = new_value - this->state_[index].value;
    return this->quad_coeff_[index] * d * d + this->linear_coeff_[index] * d;
  }

  void SetValue(std::int64_t index, std::int64_t new_value) {
    if (this->state_[index].value == new_value) {
      return;
    }

    this->energy_ += this->GetEnergyDifference(index, new_value);
    std::int64_t dx = new_value - this->state_[index].value;
    this->linear_coeff_[index] += 2.0 * this->model.GetSquared()[index] * dx;

    const auto &quadratic = this->model.GetQuadratic();
    for (const auto &[i, q] : quadratic[index]) {
      this->linear_coeff_[i] += q * dx;
    }

    this->state_[index].SetValue(new_value);
  }

  std::pair<int, double> GetMinEnergyDifference(std::int64_t index) {
    const double a = this->quad_coeff_[index];
    const double b = this->linear_coeff_[index];
    const auto &x = this->state_[index];
    const std::int64_t dxl = x.lower_bound - x.value;
    const std::int64_t dxu = x.upper_bound - x.value;

    if (a > 0) {
      const double center = -b / (2 * a);
      if (dxu <= center) {
        return std::make_pair(x.upper_bound,
                              this->GetEnergyDifference(index, x.upper_bound));
      } else if (dxl < center && center < dxu) {
        const std::int64_t dx_left =
            static_cast<std::int64_t>(std::floor(center));
        const std::int64_t dx_right =
            static_cast<std::int64_t>(std::ceil(center));
        if (center - dx_left <= dx_right - center) {
          return std::make_pair(
              x.value + dx_left,
              this->GetEnergyDifference(index, x.value + dx_left));
        } else {
          return std::make_pair(
              x.value + dx_right,
              this->GetEnergyDifference(index, x.value + dx_right));
        }
      } else if (dxl >= center) {
        return std::make_pair(x.lower_bound,
                              this->GetEnergyDifference(index, x.lower_bound));
      } else {
        throw std::runtime_error("Invalid state");
      }
    } else if (a == 0) {
      if (b > 0) {
        return std::make_pair(x.lower_bound,
                              this->GetEnergyDifference(index, x.lower_bound));
      } else if (b < 0) {
        return std::make_pair(x.upper_bound,
                              this->GetEnergyDifference(index, x.upper_bound));
      } else {
        const std::int64_t random_value =
            x.GenerateRandomValue(this->random_number_engine);
        return std::make_pair(random_value, 0.0);
      }
    } else {
      const double dE_lower = this->GetEnergyDifference(index, x.lower_bound);
      const double dE_upper = this->GetEnergyDifference(index, x.upper_bound);
      if (dE_lower <= dE_upper) {
        return std::make_pair(x.lower_bound, dE_lower);
      } else {
        return std::make_pair(x.upper_bound, dE_upper);
      }
    }
  }

  const std::vector<utility::IntegerVariable> &GetState() const {
    return this->state_;
  }

  const std::vector<double> &GetLinearCoeff() const {
    return this->linear_coeff_;
  }

  const std::vector<double> &GetQuadCoeff() const { return this->quad_coeff_; }

  double GetEnergy() const { return this->energy_; }

  bool OnlyMultiLinearCoeff(std::int64_t index) const {
    return this->model.GetOnlyBilinearIndexSet().count(index) > 0;
  }

  bool UnderQuadraticCoeff(std::int64_t index) const { return true; }

  double GetLinearCoeff(std::int64_t index) const {
    return this->linear_coeff_[index];
  }

public:
  const graph::IntegerQuadraticModel model;
  const std::int64_t seed;
  RandType random_number_engine;

private:
  std::vector<utility::IntegerVariable> state_;
  double energy_;
  std::vector<double> quad_coeff_;
  std::vector<double> linear_coeff_;
};

} // namespace system
} // namespace openjij
