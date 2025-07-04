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


class IntegerQuadraticModel {
   
public:
   
   IntegerQuadraticModel(std::vector<std::vector<std::int64_t>> &key_list,
                         std::vector<double> &value_list,
                         std::vector<std::pair<std::int64_t, std::int64_t>> &bounds) {
      if (key_list.size() != value_list.size()) {
         throw std::runtime_error("Key and value lists must have the same size.");
      }
      
      std::unordered_set<std::int64_t> index_set;
      for (const auto &key : key_list) {
         if (key.size() > 2) {
            throw std::runtime_error("Key size must be less than or equal to 2.");
         }
         index_set.insert(key.begin(), key.end());
      }
      
      bounds_ = bounds;
      
      index_list_ = std::vector<std::int64_t>(index_set.begin(), index_set.end());
      std::sort(index_list_.begin(), index_list_.end());
      
      num_variables_ = index_list_.size();
      
      quadratic_.resize(num_variables_);
      linear_.resize(num_variables_, 0.0);
      squared_.resize(num_variables_, 0.0);
      constant_ = 0.0;
      
      for (std::size_t i = 0; i < key_list.size(); ++i) {
         const auto &key = key_list[i];
         if (key.size() == 0) {
            constant_ += value_list[i];
         }
         else if (key.size() == 1) {
            if (key[0] < 0 || key[0] >= num_variables_) {
               throw std::runtime_error("Index out of bounds.");
            }
            linear_[key[0]] += value_list[i];
         }
         else if (key.size() == 2) {
            if (key[0] < 0 || key[0] >= num_variables_ || key[1] < 0 || key[1] >= num_variables_) {
               throw std::runtime_error("Index out of bounds.");
            }
            if (key[0] == key[1]) {
               squared_[key[0]] += value_list[i];
            }
            else {
               quadratic_[key[0]].emplace_back(key[1], value_list[i]);
               quadratic_[key[1]].emplace_back(key[0], value_list[i]);
            }
         }
      }
   }

   std::pair<double, double> GetMaxMinCoeffs() const {
      const double MIN_THRESHOLD = 1e-12;

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

      for (std::int64_t i = 0; i < this->num_variables_; ++i) {
         for (const auto &[j, q] : this->quadratic_[i]) {
            abs_min_dE = nonzero_abs_min(abs_min_dE, q);
            abs_max_dE = nonzero_abs_max(abs_max_dE, q);
         }
         abs_min_dE = nonzero_abs_min(abs_min_dE, this->linear_[i]);
         abs_max_dE = nonzero_abs_max(abs_max_dE, this->linear_[i]);
         abs_min_dE = nonzero_abs_min(abs_min_dE, this->squared_[i]);
         abs_max_dE = nonzero_abs_max(abs_max_dE, this->squared_[i]);
      }

      if (std::isinf(abs_min_dE) || abs_max_dE == 0.0) {
         throw std::runtime_error("No valid energy difference found.");
      }

      return std::make_pair(abs_max_dE, abs_min_dE);
   }

   const std::vector<std::int64_t>& GetIndexList() const { return index_list_; }
   std::int64_t GetNumVariables() const { return num_variables_; }
   const std::vector<std::vector<std::pair<std::int64_t, double>>>& GetQuadratic() const { return quadratic_; }
   const std::vector<double>& GetLinear() const { return linear_; }
   const std::vector<double>& GetSquared() const { return squared_; }
   double GetConstant() const { return constant_; }
   const std::vector<std::pair<std::int64_t, std::int64_t>>& GetBounds() const { return bounds_; }
   
   
   
private:
   std::vector<std::int64_t> index_list_;
   std::int64_t num_variables_;
   std::vector<std::vector<std::pair<std::int64_t, double>>> quadratic_;
   std::vector<double> linear_;
   std::vector<double> squared_;
   double constant_;
   std::vector<std::pair<std::int64_t, std::int64_t>> bounds_;
};

}
}
