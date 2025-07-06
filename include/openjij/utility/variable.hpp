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

namespace openjij {
namespace utility {

struct IntegerVariable {

  IntegerVariable(const std::int64_t lower_bound,
                  const std::int64_t upper_bound) {
    this->lower_bound = lower_bound;
    this->upper_bound = upper_bound;
    this->num_states = upper_bound - lower_bound + 1;
    this->value = lower_bound;
  }

  template <typename RandomNumberEngine>
  void SetRandomValue(RandomNumberEngine &random_number_engine) {
    this->value = std::uniform_int_distribution<std::int64_t>(
        this->lower_bound, this->upper_bound)(random_number_engine);
  }

  void SetValue(const std::int64_t value) {
    if (value < this->lower_bound || value > this->upper_bound) {
      throw std::runtime_error("Value out of bounds.");
    }
    this->value = value;
  }

  template <typename RandomNumberEngine>
  std::int64_t
  GenerateCandidateValue(RandomNumberEngine &random_number_engine) const {
    std::int64_t candidate_value = std::uniform_int_distribution<std::int64_t>(
        this->lower_bound, this->upper_bound - 1)(random_number_engine);
    if (candidate_value >= this->value) {
      candidate_value += 1; // Ensure candidate is different from current value
    }
    return candidate_value;
  }

  template <typename RandomNumberEngine>
  std::int64_t
  GenerateRandomValue(RandomNumberEngine &random_number_engine) const {
    return std::uniform_int_distribution<std::int64_t>(
        this->lower_bound, this->upper_bound)(random_number_engine);
  }

  std::int64_t GetValueFromState(std::int64_t state) const {
    return this->lower_bound + state;
  }

  std::int64_t GetStateFromValue(std::int64_t value) const {
    if (value < this->lower_bound || value > this->upper_bound) {
      throw std::runtime_error("Value out of bounds.");
    }
    return value - this->lower_bound;
  }

  std::int64_t lower_bound;
  std::int64_t upper_bound;
  std::int64_t num_states;
  std::int64_t value;
};

} // namespace utility
} // namespace openjij
