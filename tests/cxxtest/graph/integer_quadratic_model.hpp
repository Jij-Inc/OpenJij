#pragma once

namespace openjij {
namespace test {


TEST(IntegerQuadraticModelTest, BasicFunctionality) {
   std::vector<std::vector<graph::IntegerQuadraticModel::IndexType>> key_list = {
      {0, 0},
      {1, 0},
      {2},
      {}
   };
   
   std::vector<graph::IntegerQuadraticModel::ValueType> value_list = {
      1.0,
      -1.0,
      3.0,
      0.5
   };
   
   std::vector<std::pair<std::int64_t, std::int64_t>> bounds = {
      {0, 1},
      {0, 1},
      {0, 2}
   };
   
   graph::IntegerQuadraticModel model(key_list, value_list, bounds);
   
   const auto& index_list = model.get_index_list();
   EXPECT_EQ(index_list.size(), 3);
   EXPECT_EQ(index_list[0], 0);
   EXPECT_EQ(index_list[1], 1);
   EXPECT_EQ(index_list[2], 2);
   
   EXPECT_EQ(model.get_num_variables(), 3);
   
   const auto& quadratic = model.get_quadratic();
   EXPECT_EQ(quadratic.size(), 3);
   EXPECT_EQ(quadratic[0].size(), 1);
   EXPECT_EQ(quadratic[0][0].first, 1);
   EXPECT_DOUBLE_EQ(quadratic[0][0].second, -1.0);
   
   EXPECT_EQ(quadratic[1].size(), 1);
   EXPECT_EQ(quadratic[1][0].first, 0);
   EXPECT_DOUBLE_EQ(quadratic[1][0].second, -1.0);
   
   EXPECT_EQ(quadratic[2].size(), 0);
   
   const auto& linear = model.get_linear();
   EXPECT_DOUBLE_EQ(linear[0], 0.0);
   EXPECT_DOUBLE_EQ(linear[1], 0.0);
   EXPECT_DOUBLE_EQ(linear[2], 3.0);
   
   const auto& squared = model.get_squared();
   EXPECT_DOUBLE_EQ(squared[0], 1.0);
   EXPECT_DOUBLE_EQ(squared[1], 0.0);
   EXPECT_DOUBLE_EQ(squared[2], 0.0);
   
   EXPECT_DOUBLE_EQ(model.get_constant(), 0.5);
   
   const auto& bounds_result = model.get_bounds();
   EXPECT_EQ(bounds_result.size(), 3);
   EXPECT_EQ(bounds_result[0].first, 0);
   EXPECT_EQ(bounds_result[0].second, 1);
   EXPECT_EQ(bounds_result[1].first, 0);
   EXPECT_EQ(bounds_result[1].second, 1);
   EXPECT_EQ(bounds_result[2].first, 0);
   EXPECT_EQ(bounds_result[2].second, 2);
}

}
}
