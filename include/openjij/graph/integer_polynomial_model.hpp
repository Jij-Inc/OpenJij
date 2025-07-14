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
namespace graph {

class IntegerPolynomialModel {

public:
  IntegerPolynomialModel(
      std::vector<std::vector<std::int64_t>> &key_list,
      std::vector<double> &value_list,
      std::vector<std::pair<std::int64_t, std::int64_t>> &bounds) {
    if (key_list.size() != value_list.size()) {
      throw std::runtime_error("Key and value lists must have the same size.");
    }

    std::unordered_set<std::int64_t> index_set;
    for (const auto &key : key_list) {
      index_set.insert(key.begin(), key.end());
    }

    this->bounds_ = bounds;

    this->index_list_ =
        std::vector<std::int64_t>(index_set.begin(), index_set.end());
    std::sort(this->index_list_.begin(), this->index_list_.end());

    this->num_variables_ = this->index_list_.size();

    this->constant_ = 0.0;
    for (std::size_t i = 0; i < key_list.size(); ++i) {
      if (key_list[i].empty()) {
        this->constant_ += value_list[i];
      } else {
        // Count occurrences of each index
        std::unordered_map<std::int64_t, std::int64_t> index_count;
        for (const auto &k : key_list[i]) {
          index_count[k]++;
        }

        // Convert to vector of pairs
        std::vector<std::pair<std::int64_t, std::int64_t>> int_keys;
        for (const auto &[index, degree] : index_count) {
          int_keys.emplace_back(index, degree);
        }

        std::sort(int_keys.begin(), int_keys.end());

        this->key_value_list_.emplace_back(int_keys, value_list[i]);
      }
    }

    // Sort by number of variables
    std::sort(this->key_value_list_.begin(), this->key_value_list_.end(),
              [](const auto &a, const auto &b) {
                return a.first.size() < b.first.size();
              });

    // Create index_to_interactions
    this->index_to_interactions_.resize(this->num_variables_);
    for (std::size_t i = 0; i < this->key_value_list_.size(); ++i) {
      for (const auto &[index, degree] : this->key_value_list_[i].first) {
        this->index_to_interactions_[index].emplace_back(i, degree);
      }
    }

    // Sort index_to_interactions
    for (auto &interactions : this->index_to_interactions_) {
      std::sort(interactions.begin(), interactions.end());
    }

    // Create only_multilinear_index_set and under_quadratic_index_set
    for (std::int64_t index = 0; index < this->num_variables_; ++index) {
      bool is_multilinear = true;
      for (const auto &[_, degree] : this->index_to_interactions_[index]) {
        if (degree != 1) {
          is_multilinear = false;
          break;
        }
      }
      if (is_multilinear) {
        this->only_multilinear_index_set_.insert(index);
      }

      bool is_under_quadratic = true;
      for (const auto &[_, degree] : this->index_to_interactions_[index]) {
        if (degree > 2) {
          is_under_quadratic = false;
          break;
        }
      }
      if (is_under_quadratic) {
        this->under_quadratic_index_set_.insert(index);
      }
    }
  }

  std::pair<double, double> GetMaxMinTerms() const {
    const double MIN_THRESHOLD = 1e-10;

    auto nonzero_abs_min = [&](double current_min, double value) -> double {
      if (std::abs(value) > MIN_THRESHOLD) {
        return std::min(std::abs(current_min), std::abs(value));
      }
      return current_min;
    };

    auto nonzero_abs_max = [&](double current_max, double value) -> double {
      if (std::abs(value) > MIN_THRESHOLD) {
        return std::max(std::abs(current_max), std::abs(value));
      }
      return current_max;
    };

    double abs_min_dE = std::numeric_limits<double>::infinity();
    double abs_max_dE = 0.0;

    for (std::int64_t i = 0; i < this->key_value_list_.size(); ++i) {
      const double value = this->key_value_list_[i].second;
      double min_term = std::abs(value);
      double max_term = std::abs(value);
      for (const auto &iter: this->key_value_list_[i].first) {
        const std::int64_t index = iter.first;
        const std::int64_t degree = iter.second;
        const std::int64_t lower = this->bounds_[index].first;
        const std::int64_t upper = this->bounds_[index].second;
        if (lower <= 0 && 0 <= upper) {
          max_term *= std::pow(std::max(std::abs(lower), std::abs(upper)), degree);
        }
        else {
          if (lower > 0) {
            min_term *= std::pow(lower, degree);
            max_term *= std::pow(upper, degree);
          } else if (upper < 0) {
            min_term *= std::abs(std::pow(upper, degree));
            max_term *= std::abs(std::pow(lower, degree));
          } else {
            throw std::runtime_error("Bounds are not valid.");
          }
        }
      }
      
      abs_min_dE = nonzero_abs_min(abs_min_dE, min_term);
      abs_max_dE = nonzero_abs_max(abs_max_dE, max_term);
    }

    if (std::isinf(abs_min_dE) || abs_max_dE == 0.0) {
      throw std::runtime_error("No valid energy difference found.");
    }

    return std::make_pair(abs_max_dE, abs_min_dE);
  }

  const std::vector<std::int64_t> &GetIndexList() const {
    return this->index_list_;
  }
  std::int64_t GetNumVariables() const { return this->num_variables_; }
  const std::vector<std::pair<std::int64_t, std::int64_t>> &GetBounds() const {
    return this->bounds_;
  }
  double GetConstant() const { return this->constant_; }
  const std::vector<
      std::pair<std::vector<std::pair<std::int64_t, std::int64_t>>, double>> &
  GetKeyValueList() const {
    return this->key_value_list_;
  }
  const std::vector<std::vector<std::pair<std::size_t, std::int64_t>>> &
  GetIndexToInteractions() const {
    return this->index_to_interactions_;
  }
  const std::unordered_set<std::int64_t> &GetOnlyMultilinearIndexSet() const {
    return this->only_multilinear_index_set_;
  }
  const std::unordered_set<std::int64_t> &GetUnderQuadraticIndexSet() const {
    return this->under_quadratic_index_set_;
  }

private:
  std::vector<std::int64_t> index_list_;
  std::int64_t num_variables_;
  std::vector<std::pair<std::int64_t, std::int64_t>> bounds_;
  double constant_;
  std::vector<
      std::pair<std::vector<std::pair<std::int64_t, std::int64_t>>, double>>
      key_value_list_;
  std::vector<std::vector<std::pair<std::size_t, std::int64_t>>>
      index_to_interactions_;
  std::unordered_set<std::int64_t> only_multilinear_index_set_;
  std::unordered_set<std::int64_t> under_quadratic_index_set_;
};

} // namespace graph
} // namespace openjij
