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

TEST(Sampler, SASamplerOperationIsingPolynomial) {
   
   using FloatType = double;
   using Tup = utility::AnyTupleType;
   using IPM = graph::IsingPolynomialModel<FloatType>;
   
   std::vector<std::vector<typename IPM::IndexType>> key_list = {
      {1, 1},
      {"a", "a"},
      {Tup{2, "b"}, Tup{2, "b"}},
      {1, 2},
      {"a", 1},
      {Tup{2, "b"}, Tup{2, "a"}}
   };
      
   std::vector<FloatType> value_list = {
      +4.0,
      +2.0,
      +3.0,
      -1.0,
      -1.5,
      -2.5
   };
   
   const auto ipm = IPM{key_list, value_list};
   
   auto sa_sampler = sampler::SASampler{ipm};
   sa_sampler.SetBetaMaxAuto();
   sa_sampler.SetBetaMinAuto();
   
   std::vector<algorithm::RandomNumberEngine> engine_list = {
      algorithm::RandomNumberEngine::XORSHIFT,
      algorithm::RandomNumberEngine::MT,
      algorithm::RandomNumberEngine::MT_64
   };
   
   std::vector<algorithm::UpdateMethod> updater_list = {
      algorithm::UpdateMethod::METROPOLIS,
      algorithm::UpdateMethod::HEAT_BATH
   };
   
   std::vector<utility::TemperatureSchedule> schedule_list = {
      utility::TemperatureSchedule::LINEAR,
      utility::TemperatureSchedule::GEOMETRIC
   };
   
   for (const auto &engine: engine_list) {
      sa_sampler.SetRandomNumberEngine(engine);
      for (const auto &algorithm: updater_list) {
         sa_sampler.SetUpdateMethod(algorithm);
         for (const auto &schedule: schedule_list) {
            sa_sampler.SetTemperatureSchedule(schedule);
            EXPECT_NO_THROW(sa_sampler.Sample());
            EXPECT_NO_THROW(sa_sampler.CalculateEnergies());
         }
      }
   }

}

TEST(Sampler, SASamplerHistoryIsingPolynomial) {

   using FloatType = double;
   using IPM = graph::IsingPolynomialModel<FloatType>;

   std::vector<std::vector<typename IPM::IndexType>> key_list = {
      {0}, {1}, {2}, {0, 1}, {1, 2}, {0, 1, 2}
   };
   std::vector<FloatType> value_list = {
      1.0, -2.0, 0.5, -1.5, 2.5, -0.75
   };
   const auto model = IPM{key_list, value_list};

   constexpr std::int32_t num_reads = 3;
   constexpr std::int32_t num_sweeps = 7;
   constexpr std::uint64_t seed = 67890;
   constexpr FloatType beta_min = 0.5;
   constexpr FloatType beta_max = 2.0;

   auto sa_sampler = sampler::SASampler{model};
   sa_sampler.SetNumReads(num_reads);
   sa_sampler.SetNumSweeps(num_sweeps);
   sa_sampler.SetNumThreads(2);
   sa_sampler.SetBetaMin(beta_min);
   sa_sampler.SetBetaMax(beta_max);
   sa_sampler.SetRandomNumberEngine(algorithm::RandomNumberEngine::MT);
   sa_sampler.SetUpdateMethod(algorithm::UpdateMethod::HEAT_BATH);
   sa_sampler.SetTemperatureSchedule(utility::TemperatureSchedule::GEOMETRIC);

   EXPECT_FALSE(sa_sampler.GetLogHistory());
   sa_sampler.Sample(seed);
   const auto samples_without_history = sa_sampler.GetSamples();
   const auto energies_without_history = sa_sampler.CalculateEnergies();
   EXPECT_TRUE(sa_sampler.GetEnergyHistory().empty());
   EXPECT_TRUE(sa_sampler.GetTemperatureHistory().empty());

   sa_sampler.SetLogHistory(true);
   EXPECT_TRUE(sa_sampler.GetLogHistory());
   sa_sampler.Sample(seed);

   const auto &energy_history = sa_sampler.GetEnergyHistory();
   const auto &temperature_history = sa_sampler.GetTemperatureHistory();
   const auto energies = sa_sampler.CalculateEnergies();

   EXPECT_EQ(sa_sampler.GetSamples(), samples_without_history);
   EXPECT_EQ(energies, energies_without_history);
   ASSERT_EQ(energy_history.size(), static_cast<std::size_t>(num_reads));
   ASSERT_EQ(temperature_history.size(), static_cast<std::size_t>(num_reads));
   ASSERT_EQ(energies.size(), static_cast<std::size_t>(num_reads));
   for (std::int32_t read = 0; read < num_reads; ++read) {
      ASSERT_EQ(energy_history[read].size(),
                static_cast<std::size_t>(num_sweeps));
      ASSERT_EQ(temperature_history[read].size(),
                static_cast<std::size_t>(num_sweeps));
      EXPECT_NEAR(temperature_history[read].front(), 1.0/beta_min, 1e-12);
      EXPECT_NEAR(temperature_history[read].back(), 1.0/beta_max, 1e-12);
      EXPECT_NEAR(energy_history[read].back(), energies[read], 1e-12);
      EXPECT_EQ(temperature_history[read], temperature_history.front());
   }

   sa_sampler.SetLogHistory(false);
   EXPECT_FALSE(sa_sampler.GetLogHistory());
   sa_sampler.Sample(seed);

   EXPECT_EQ(sa_sampler.GetSamples(), samples_without_history);
   EXPECT_EQ(sa_sampler.CalculateEnergies(), energies_without_history);
   EXPECT_TRUE(sa_sampler.GetEnergyHistory().empty());
   EXPECT_TRUE(sa_sampler.GetTemperatureHistory().empty());
}


}
}
