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
   //! @brief The value type.
   using ValueType = double;
   
   //! @brief The index type.
   using IndexType = std::int64_t;
   
   //! @brief The variable type.
   using VariableType = std::int64_t;
   
   
   IntegerQuadraticModel(std::vector<std::vector<IndexType>> &key_list,
                         std::vector<ValueType> &value_list,
                         std::vector<std::pair<std::int64_t, std::int64_t>> &bounds) {
      if (key_list.size() != value_list.size()) {
         throw std::runtime_error("Key and value lists must have the same size.");
      }
      
      std::unordered_set<IndexType> index_set;
      for (const auto &key : key_list) {
         if (key.size() > 2) {
            throw std::runtime_error("Key size must be less than or equal to 2.");
         }
         index_set.insert(key.begin(), key.end());
      }
      
      index_list_ = std::vector<IndexType>(index_set.begin(), index_set.end());
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
   
   
   
private:
   std::vector<IndexType> index_list_;
   std::int64_t num_variables_;
   std::vector<std::vector<std::pair<IndexType, ValueType>>> quadratic_;
   std::vector<ValueType> linear_;
   std::vector<ValueType> squared_;
   ValueType constant_;
   std::vector<std::pair<std::int64_t, std::int64_t>> bounds_;
};

}
}
