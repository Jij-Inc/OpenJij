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


// include Google Test
#include <gtest/gtest.h>
#include <gmock/gmock.h>

// include STL
#include <iostream>
#include <utility>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

// include OpenJij
#include <openjij/graph/all.hpp>
#include <openjij/system/all.hpp>
#include <openjij/updater/all.hpp>
#include <openjij/algorithm/all.hpp>
#include <openjij/result/all.hpp>
#include <openjij/utility/schedule_list.hpp>
#include <openjij/utility/union_find.hpp>
#include <openjij/utility/random.hpp>
#include <openjij/utility/min_polynomial.hpp>
#include <openjij/utility/gpu/memory.hpp>
#include <openjij/utility/gpu/cublas.hpp>
#include <openjij/sampler/sa_sampler.hpp>
#include <openjij/sampler/integer_sa_sampler.hpp>


// include Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>

// include tests
#define TEST_CASE_INDEX 1

#include "testcase.hpp"
#include "polynomial_test.hpp"

#include "sampler/all.hpp"
#include "system/all.hpp"
#include "graph/all.hpp"
#include "result/all.hpp"
#include "utility/all.hpp"

TEST(ScheduleList, FiniteBetasAcrossExtremeRatios) {
   for (const auto &[first, last] :
        {std::pair{1e-150, 1e160}, std::pair{1e160, 1e-150}}) {
      const auto schedule = openjij::utility::make_classical_schedule_list(first, last, 2, 5);
      ASSERT_EQ(schedule.size(), 5u);
      EXPECT_DOUBLE_EQ(schedule.front().updater_parameter.beta, first);
      EXPECT_DOUBLE_EQ(schedule.back().updater_parameter.beta, last);
      EXPECT_NEAR(schedule[2].updater_parameter.beta / 1e5, 1.0, 1e-12);
      for (const auto &step : schedule) {
         EXPECT_EQ(step.one_mc_step, 2u);
         EXPECT_TRUE(std::isfinite(step.updater_parameter.beta));
         EXPECT_GT(step.updater_parameter.beta, 0.0);
      }
   }
}

TEST(ScheduleList, EqualZeroEndpointsStayAtZero) {
   const auto schedule = openjij::utility::make_classical_schedule_list(0.0, 0.0, 1, 4);
   ASSERT_EQ(schedule.size(), 4u);
   for (const auto &step : schedule) {
      EXPECT_EQ(step.updater_parameter.beta, 0.0);
   }
}

std::int32_t main(std::int32_t argc, char **argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
