#pragma once

namespace openjij {
namespace test {

TEST(IntegerPolynomialModelTest, BasicFunctionality) {
  std::vector<std::vector<std::int64_t>> key_list = {
      {0, 0, 0}, {1, 0, 1}, {1, 2, 3}, {4, 4}, {1, 3}, {2}, {},
  };

  std::vector<double> value_list = {1.0, -1.0, 3.0, -1.5, 0.5, 2.5, 0.5};

  std::vector<std::pair<std::int64_t, std::int64_t>> bounds = {
      {0, 1}, {0, 1}, {0, 2}, {-1, 3}, {-2, 2}};

  graph::IntegerPolynomialModel model(key_list, value_list, bounds);

  const auto &index_list = model.GetIndexList();
  EXPECT_EQ(index_list.size(), 5);
  EXPECT_EQ(index_list[0], 0);
  EXPECT_EQ(index_list[1], 1);
  EXPECT_EQ(index_list[2], 2);
  EXPECT_EQ(index_list[3], 3);
  EXPECT_EQ(index_list[4], 4);

  EXPECT_EQ(model.GetNumVariables(), 5);

  const auto &bounds_result = model.GetBounds();
  EXPECT_EQ(bounds_result.size(), 5);
  EXPECT_EQ(bounds_result[0].first, 0);
  EXPECT_EQ(bounds_result[0].second, 1);
  EXPECT_EQ(bounds_result[1].first, 0);
  EXPECT_EQ(bounds_result[1].second, 1);
  EXPECT_EQ(bounds_result[2].first, 0);
  EXPECT_EQ(bounds_result[2].second, 2);
  EXPECT_EQ(bounds_result[3].first, -1);
  EXPECT_EQ(bounds_result[3].second, 3);
  EXPECT_EQ(bounds_result[4].first, -2);
  EXPECT_EQ(bounds_result[4].second, 2);

  EXPECT_DOUBLE_EQ(model.GetConstant(), 0.5);

  const auto &key_value_list = model.GetKeyValueList();
  EXPECT_EQ(key_value_list.size(), 6);

  EXPECT_EQ(key_value_list[0].first.size(), 1);
  EXPECT_EQ(key_value_list[0].first[0].first, 0);
  EXPECT_EQ(key_value_list[0].first[0].second, 3);
  EXPECT_DOUBLE_EQ(key_value_list[0].second, 1.0);

  EXPECT_EQ(key_value_list[1].first.size(), 1);
  EXPECT_EQ(key_value_list[1].first[0].first, 4);
  EXPECT_EQ(key_value_list[1].first[0].second, 2);
  EXPECT_DOUBLE_EQ(key_value_list[1].second, -1.5);

  EXPECT_EQ(key_value_list[2].first.size(), 1);
  EXPECT_EQ(key_value_list[2].first[0].first, 2);
  EXPECT_EQ(key_value_list[2].first[0].second, 1);
  EXPECT_DOUBLE_EQ(key_value_list[2].second, 2.5);

  EXPECT_EQ(key_value_list[3].first.size(), 2);
  EXPECT_EQ(key_value_list[3].first[0].first, 0);
  EXPECT_EQ(key_value_list[3].first[0].second, 1);
  EXPECT_EQ(key_value_list[3].first[1].first, 1);
  EXPECT_EQ(key_value_list[3].first[1].second, 2);
  EXPECT_DOUBLE_EQ(key_value_list[3].second, -1.0);

  EXPECT_EQ(key_value_list[4].first.size(), 2);
  EXPECT_EQ(key_value_list[4].first[0].first, 1);
  EXPECT_EQ(key_value_list[4].first[0].second, 1);
  EXPECT_EQ(key_value_list[4].first[1].first, 3);
  EXPECT_EQ(key_value_list[4].first[1].second, 1);
  EXPECT_DOUBLE_EQ(key_value_list[4].second, 0.5);

  EXPECT_EQ(key_value_list[5].first.size(), 3);
  EXPECT_EQ(key_value_list[5].first[0].first, 1);
  EXPECT_EQ(key_value_list[5].first[0].second, 1);
  EXPECT_EQ(key_value_list[5].first[1].first, 2);
  EXPECT_EQ(key_value_list[5].first[1].second, 1);
  EXPECT_EQ(key_value_list[5].first[2].first, 3);
  EXPECT_EQ(key_value_list[5].first[2].second, 1);
  EXPECT_DOUBLE_EQ(key_value_list[5].second, 3.0);

  const auto &index_to_interactions = model.GetIndexToInteractions();
  EXPECT_EQ(index_to_interactions.size(), 5);

  EXPECT_EQ(index_to_interactions[0].size(), 2);
  EXPECT_EQ(index_to_interactions[0][0].first, 0);
  EXPECT_EQ(index_to_interactions[0][0].second, 3);
  EXPECT_EQ(index_to_interactions[0][1].first, 3);
  EXPECT_EQ(index_to_interactions[0][1].second, 1);

  EXPECT_EQ(index_to_interactions[1].size(), 3);
  EXPECT_EQ(index_to_interactions[1][0].first, 3);
  EXPECT_EQ(index_to_interactions[1][0].second, 2);
  EXPECT_EQ(index_to_interactions[1][1].first, 4);
  EXPECT_EQ(index_to_interactions[1][1].second, 1);
  EXPECT_EQ(index_to_interactions[1][2].first, 5);
  EXPECT_EQ(index_to_interactions[1][2].second, 1);

  EXPECT_EQ(index_to_interactions[2].size(), 2);
  EXPECT_EQ(index_to_interactions[2][0].first, 2);
  EXPECT_EQ(index_to_interactions[2][0].second, 1);
  EXPECT_EQ(index_to_interactions[2][1].first, 5);
  EXPECT_EQ(index_to_interactions[2][1].second, 1);

  EXPECT_EQ(index_to_interactions[3].size(), 2);
  EXPECT_EQ(index_to_interactions[3][0].first, 4);
  EXPECT_EQ(index_to_interactions[3][0].second, 1);
  EXPECT_EQ(index_to_interactions[3][1].first, 5);
  EXPECT_EQ(index_to_interactions[3][1].second, 1);

  EXPECT_EQ(index_to_interactions[4].size(), 1);
  EXPECT_EQ(index_to_interactions[4][0].first, 1);
  EXPECT_EQ(index_to_interactions[4][0].second, 2);

  const auto &only_multilinear_set = model.GetOnlyMultilinearIndexSet();
  EXPECT_EQ(only_multilinear_set.size(), 2);
  EXPECT_EQ(only_multilinear_set.count(0), 0);
  EXPECT_EQ(only_multilinear_set.count(1), 0);
  EXPECT_EQ(only_multilinear_set.count(2), 1);
  EXPECT_EQ(only_multilinear_set.count(3), 1);
  EXPECT_EQ(only_multilinear_set.count(4), 0);

  const auto &under_quadratic_set = model.GetUnderQuadraticIndexSet();
  EXPECT_EQ(under_quadratic_set.size(), 4);
  EXPECT_EQ(under_quadratic_set.count(1), 1);
  EXPECT_EQ(under_quadratic_set.count(2), 1);
  EXPECT_EQ(under_quadratic_set.count(3), 1);
  EXPECT_EQ(under_quadratic_set.count(4), 1);
  EXPECT_EQ(under_quadratic_set.count(0), 0);

  const auto [max_coeff, min_coeff] = model.GetMaxMinTerms();
  EXPECT_DOUBLE_EQ(max_coeff, 18.0);
  EXPECT_DOUBLE_EQ(min_coeff, 0.5);
}

} // namespace test
} // namespace openjij
