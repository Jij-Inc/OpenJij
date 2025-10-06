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
namespace test {

TEST(FindMinimum, FindMinimumIntegerQuadratic) {
  auto random_engine = std::mt19937(0);

  const auto r1 = utility::FindMinimumIntegerQuadratic(1.0, -5.0, -10, 10, 0, random_engine);
  EXPECT_TRUE(r1.first == 2 || r1.first == 3);
  EXPECT_DOUBLE_EQ(r1.second, -6.0);

  const auto r2 = utility::FindMinimumIntegerQuadratic(1.0, 10.0, 0, 10, 0, random_engine);
  EXPECT_EQ(r2.first, 0);
  EXPECT_DOUBLE_EQ(r2.second, 0.0);

  const auto r3 = utility::FindMinimumIntegerQuadratic(-1.0, 2.0, -10, 5, 0, random_engine);
  EXPECT_EQ(r3.first, -10);
  EXPECT_DOUBLE_EQ(r3.second, -120.0);

  const auto r4 = utility::FindMinimumIntegerQuadratic(-1.0, -2.0, -5, 10, 0, random_engine);
  EXPECT_EQ(r4.first, 10);
  EXPECT_DOUBLE_EQ(r4.second, -120.0);
  
  const auto r5 = utility::FindMinimumIntegerQuadratic(0.0, 0.0, 5, 10, 0, random_engine);
  EXPECT_GE(r5.first, 5);
  EXPECT_LE(r5.first, 10);
  EXPECT_DOUBLE_EQ(r5.second, 0.0);
}

TEST(FindMinimum, FindMinimumIntegerCubicTest) {
  auto random_engine = std::mt19937(0);

  const auto r1 = utility::FindMinimumIntegerCubic(0.0, 0.0, 0.0, 10, 20, 0, random_engine);
  EXPECT_GE(r1.first, 10);
  EXPECT_LE(r1.first, 20);
  EXPECT_DOUBLE_EQ(r1.second, 0.0);

  const auto r2 = utility::FindMinimumIntegerCubic(0.0, 0.0, 2.0, -5, 5, 0, random_engine);
  EXPECT_EQ(r2.first, -5);
  EXPECT_DOUBLE_EQ(r2.second, -10.0);

  const auto r3 = utility::FindMinimumIntegerCubic(0.0, 1.0, -5.0, 0, 10, 0, random_engine);
  EXPECT_TRUE(r3.first == 2 || r3.first == 3);
  EXPECT_DOUBLE_EQ(r3.second, -6.0);

  const auto r4 = utility::FindMinimumIntegerCubic(0.0, 1.0, -22.0, -5, 5, 0, random_engine);
  EXPECT_EQ(r4.first, 5);
  EXPECT_DOUBLE_EQ(r4.second, -85.0);
  
  const auto r5 = utility::FindMinimumIntegerCubic(1.0, 1.0, 1.0, -5, 5, 0, random_engine);
  EXPECT_EQ(r5.first, -5);
  EXPECT_DOUBLE_EQ(r5.second, -105.0);

  const auto r6 = utility::FindMinimumIntegerCubic(2.0, -15.0, 24.0, 0, 10, 0, random_engine);
  EXPECT_EQ(r6.first, 4);
  EXPECT_DOUBLE_EQ(r6.second, -16.0);

  const auto r7 = utility::FindMinimumIntegerCubic(-2.0, 15.0, -24.0, 0, 10, 0, random_engine);
  EXPECT_EQ(r7.first, 10);
  EXPECT_DOUBLE_EQ(r7.second, -740.0);
  
  const auto r8 = utility::FindMinimumIntegerCubic(-2.0, 15.0, -24.0, 0, 3, 0, random_engine);
  EXPECT_EQ(r8.first, 1);
  EXPECT_DOUBLE_EQ(r8.second, -11.0);
}

TEST(FindMinimumIntegerQuarticTest, AllCases) {
  auto random_engine = std::mt19937(0);

  const auto r1 = utility::FindMinimumIntegerQuartic(0.0, 0.0, 0.0, 0.0, -5, 5, 0, random_engine);
  EXPECT_GE(r1.first, -5);
  EXPECT_LE(r1.first, 5);
  EXPECT_DOUBLE_EQ(r1.second, 0.0);

  const auto r2 = utility::FindMinimumIntegerQuartic(0.0, 0.0, 0.0, 5.0, -10, 10, 0, random_engine);
  EXPECT_EQ(r2.first, -10);
  EXPECT_DOUBLE_EQ(r2.second, -50.0);

  const auto r3 = utility::FindMinimumIntegerQuartic(0.0, 0.0, 1.0, -6.0, 0, 10, 0, random_engine);
  EXPECT_EQ(r3.first, 3);
  EXPECT_DOUBLE_EQ(r3.second, -9.0);

  const auto r4 = utility::FindMinimumIntegerQuartic(0.0, 1.0, -6.0, 5.0, 0, 10, 0, random_engine);
  EXPECT_EQ(r4.first, 3);
  EXPECT_DOUBLE_EQ(r4.second, -12.0);

  const auto r5 = utility::FindMinimumIntegerQuartic(1.0, 0.0, 0.0, 1.0, -5, 5, 0, random_engine);
  EXPECT_EQ(r5.first, -1);
  EXPECT_DOUBLE_EQ(r5.second, 0.0);
  
  const auto r6 = utility::FindMinimumIntegerQuartic(0.25, 0.0, -2.0, 0.0, -10, 10, 0, random_engine);
  EXPECT_TRUE(r6.first == -2 || r6.first == 2);
  EXPECT_DOUBLE_EQ(r6.second, -4.0);
  
  const auto r7 = utility::FindMinimumIntegerQuartic(0.25, 0.0, -2.0, 0.0, 3, 10, 0, random_engine);
  EXPECT_EQ(r7.first, 3);
  EXPECT_DOUBLE_EQ(r7.second, 2.25);

  const auto r8 = utility::FindMinimumIntegerQuartic(0.25, 0.5, -2.0, -1.0, -10, 10, 0, random_engine);
  EXPECT_EQ(r8.first, -3);
  EXPECT_DOUBLE_EQ(r8.second, -8.25);
}

}
}
