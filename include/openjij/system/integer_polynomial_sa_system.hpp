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
#include "../utility/min_polynomial.hpp"
#include "./sa_system.hpp"
#include <cstdint>
#include <random>
#include <vector>

namespace openjij {
namespace system {

template <typename RandType>
class IntegerSASystem<graph::IntegerPolynomialModel, RandType> {

public:
  IntegerSASystem(const graph::IntegerPolynomialModel &model,
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

    // Set zero count and term product
    const auto &key_value_list = this->model.GetKeyValueList();
    const std::size_t num_terms = key_value_list.size();
    this->zero_count_.resize(num_terms, 0);
    this->term_prod_.resize(num_terms, 1.0);

    for (std::size_t i = 0; i < num_terms; ++i) {
      int count = 0;
      double prod = 1.0;
      for (const auto &[index, degree] : key_value_list[i].first) {
        auto x = this->state_[index].value;
        if (x == 0) {
          count++;
        } else {
          prod *= std::pow(x, degree);
        }
      }
      this->zero_count_[i] = count;
      this->term_prod_[i] = prod;
    }

    // Initialize energy difference
    this->base_energy_difference_.resize(num_variables);
    for (std::size_t i = 0; i < num_terms; ++i) {
      const double value = key_value_list[i].second;
      for (const auto &[index, degree] : key_value_list[i].first) {
        const auto x = this->state_[index].value;
        if (x == 0) {
          if (this->zero_count_[i] == 1) {
            this->base_energy_difference_[index][degree] +=
                value * this->term_prod_[i];
          }
        } else {
          if (this->zero_count_[i] == 0) {
            this->base_energy_difference_[index][degree] +=
                value * this->term_prod_[i] / std::pow(x, degree);
          }
        }
      }
    }
  }

  double
  CalculateEnergy(const std::vector<utility::IntegerVariable> &state) const {
    double energy = this->model.GetConstant();
    const auto &key_value_list = this->model.GetKeyValueList();
    for (const auto &term : key_value_list) {
      double prod = 1.0;
      for (const auto &[index, degree] : term.first) {
        prod *= std::pow(state[index].value, degree);
      }
      energy += term.second * prod;
    }
    return energy;
  }

  std::int64_t GenerateCandidateValue(std::int64_t index) {
    return this->state_[index].GenerateCandidateValue(
        this->random_number_engine);
  }

  double GetEnergyDifference(std::int64_t index, std::int64_t new_value) const {
    double dE = 0.0;
    const auto current_value = this->state_[index].value;
    for (const auto &[degree, value] : this->base_energy_difference_[index]) {
      dE += value *
            (std::pow(new_value, degree) - std::pow(current_value, degree));
    }
    return dE;
  }

  void SetValue(std::int64_t index, std::int64_t new_value) {
    const auto current_value = this->state_[index].value;
    if (current_value == new_value) {
      return;
    }

    this->energy_ += this->GetEnergyDifference(index, new_value);
    this->state_[index].SetValue(new_value);

    const auto &interactions = this->model.GetIndexToInteractions().at(index);
    const auto &key_value_list = this->model.GetKeyValueList();

    if (current_value != 0 && new_value != 0) {
      for (const auto &[cons_ind, degree] : interactions) {
        const double a =
            std::pow(static_cast<double>(new_value) / current_value, degree);
        const double ddE = (a - 1) * key_value_list[cons_ind].second *
                           this->term_prod_[cons_ind];
        this->term_prod_[cons_ind] *= a;
        const int total_count = this->zero_count_[cons_ind];
        for (const auto &[i, d] : key_value_list[cons_ind].first) {
          if (i != index) {
            if (this->state_[i].value == 0 && total_count == 1) {
              this->base_energy_difference_[i][d] += ddE;
            } else if (this->state_[i].value != 0 && total_count == 0) {
              this->base_energy_difference_[i][d] +=
                  ddE / std::pow(this->state_[i].value, d);
            }
          }
        }
      }
    } else if (current_value == 0 && new_value != 0) {
      for (const auto &[cons_ind, degree] : interactions) {
        this->zero_count_[cons_ind]--;
        const int total_count = this->zero_count_[cons_ind];
        const double b = std::pow(new_value, degree);
        const double ddE =
            b * key_value_list[cons_ind].second * this->term_prod_[cons_ind];
        this->term_prod_[cons_ind] *= b;
        for (const auto &[i, d] : key_value_list[cons_ind].first) {
          if (i != index) {
            if (this->state_[i].value == 0 && total_count == 1) {
              this->base_energy_difference_[i][d] += ddE;
            } else if (this->state_[i].value != 0 && total_count == 0) {
              this->base_energy_difference_[i][d] +=
                  ddE / std::pow(this->state_[i].value, d);
            }
          }
        }
      }
    } else { // current_value != 0 && new_value == 0
      for (const auto &[cons_ind, degree] : interactions) {
        const int total_count = this->zero_count_[cons_ind];
        this->zero_count_[cons_ind]++;
        const double ddE =
            -key_value_list[cons_ind].second * this->term_prod_[cons_ind];
        this->term_prod_[cons_ind] /= std::pow(current_value, degree);
        for (const auto &[i, d] : key_value_list[cons_ind].first) {
          if (i != index) {
            if (this->state_[i].value == 0 && total_count == 1) {
              this->base_energy_difference_[i][d] += ddE;
            } else if (this->state_[i].value != 0 && total_count == 0) {
              this->base_energy_difference_[i][d] +=
                  ddE / std::pow(this->state_[i].value, d);
            }
          }
        }
      }
    }
  }

  std::pair<int, double> GetMinEnergyDifference(std::int64_t index) {
    const auto &x = this->state_[index];
    const std::int64_t dxl = x.lower_bound - x.value;
    const std::int64_t dxu = x.upper_bound - x.value;

    if (this->UnderQuadraticCoeff(index)) {
      auto it_a = this->base_energy_difference_[index].find(2);
      const double a = (it_a != this->base_energy_difference_[index].end())
                           ? it_a->second
                           : 0.0;
      auto it_b = this->base_energy_difference_[index].find(1);
      const double b = (it_b != this->base_energy_difference_[index].end())
                                ? it_b->second
                                : 0.0;

      const double aa = a;
      const double bb = b + 2 * x.value * a;

      return utility::FindMinimumIntegerQuadratic(aa, bb, dxl, dxu, x.value, this->random_number_engine);
    }
    else if (this->IsCubicCoeff(index)) {
      auto it_a = this->base_energy_difference_[index].find(3);
      const double a = (it_a != this->base_energy_difference_[index].end())
                           ? it_a->second
                           : 0.0;
      auto it_b = this->base_energy_difference_[index].find(2);
      const double b = (it_b != this->base_energy_difference_[index].end())
                                ? it_b->second
                                : 0.0;
      auto it_c = this->base_energy_difference_[index].find(1);
      const double c = (it_c != this->base_energy_difference_[index].end())
                                ? it_c->second
                                : 0.0;

      const double aa = a;
      const double bb = 3 * a * x.value + b;
      const double cc = 3 * a * x.value * x.value + 2 * b * x.value + c;

      return utility::FindMinimumIntegerCubic(aa, bb, cc, dxl, dxu, x.value, this->random_number_engine);
    } else if (this->IsQuarticCoeff(index)) {
      auto it_a = this->base_energy_difference_[index].find(4);
      const double a = (it_a != this->base_energy_difference_[index].end())
                           ? it_a->second
                           : 0.0;
      auto it_b = this->base_energy_difference_[index].find(3);
      const double b = (it_b != this->base_energy_difference_[index].end())
                                ? it_b->second
                                : 0.0;
      auto it_c = this->base_energy_difference_[index].find(2);
      const double c = (it_c != this->base_energy_difference_[index].end())
                                ? it_c->second
                                : 0.0;
      auto it_d = this->base_energy_difference_[index].find(1);
      const double d = (it_d != this->base_energy_difference_[index].end())
                                ? it_d->second
                                : 0.0;

      const double aa = a;
      const double bb = 4 * a * x.value + b;
      const double cc = 6 * a * x.value * x.value + 3 * b * x.value + c;
      const double dd = 4 * a * x.value * x.value * x.value +
                        3 * b * x.value * x.value + 2 * c * x.value + d;
                        
      return utility::FindMinimumIntegerQuartic(aa, bb, cc, dd, dxl, dxu, x.value, this->random_number_engine);
    } else {
      double min_dE = std::numeric_limits<double>::infinity();
      std::int64_t min_value = -1;

      for (std::int64_t val = this->state_[index].lower_bound;
           val <= this->state_[index].upper_bound; ++val) {
        double dE = this->GetEnergyDifference(index, val);
        if (dE < min_dE) {
          min_dE = dE;
          min_value = val;
        }
      }
      if (min_value == -1) {
        throw std::runtime_error("No valid state number found.");
      }
      return {min_value, min_dE};
    }
  }

  double GetLinearCoeff(std::int64_t index) const {
    auto it1 = this->base_energy_difference_[index].find(1);
    double c1 =
        (it1 != this->base_energy_difference_[index].end()) ? it1->second : 0.0;

    auto it2 = this->base_energy_difference_[index].find(2);
    double c2 =
        (it2 != this->base_energy_difference_[index].end()) ? it2->second : 0.0;

    return c1 + 2 * this->state_[index].value * c2;
  }

  const std::vector<utility::IntegerVariable> &GetState() const {
    return this->state_;
  }

  double GetEnergy() const { return this->energy_; }

  bool IsLinearCoeff(std::int64_t index) const {
    return this->model.GetEachVariableDegreeAt(index) == 1;
  }

  bool UnderQuadraticCoeff(std::int64_t index) const {
    return (this->model.GetEachVariableDegreeAt(index) == 2) || (this->model.GetEachVariableDegreeAt(index) == 1);
  }

  bool IsCubicCoeff(std::int64_t index) const {
    return this->model.GetEachVariableDegreeAt(index) == 3;
  }

  bool IsQuarticCoeff(std::int64_t index) const {
    return this->model.GetEachVariableDegreeAt(index) == 4;
  }

  bool CanOptMove(std::int64_t index) const {
    return this->UnderQuadraticCoeff(index) || this->IsCubicCoeff(index) || this->IsQuarticCoeff(index);
  }

public:
  const graph::IntegerPolynomialModel &model;
  const typename RandType::result_type seed;
  RandType random_number_engine;

private:
  std::vector<utility::IntegerVariable> state_;
  double energy_;
  std::vector<int> zero_count_;
  std::vector<double> term_prod_;
  std::vector<std::unordered_map<std::int64_t, double>> base_energy_difference_;
};

} // namespace system
} // namespace openjij
