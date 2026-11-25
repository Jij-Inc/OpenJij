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

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "openjij/system/transverse_ising.hpp"
#include "openjij/utility/schedule_list.hpp"

namespace openjij {
namespace updater {

/**
 * @brief Imaginary-time (Trotter-direction) cluster updater.
 *
 * @tparam System type of system
 */
template <typename System> struct TrotterClusterFlip;

/**
 * @brief Trotter-direction cluster flip for the discrete-time transverse
 * field Ising model.
 *
 * For every classical spin \f$i\f$ the ring of its \f$M\f$ Trotter copies is
 * an Ising chain with ferromagnetic coupling
 * \f$J_\perp = -\frac12\ln\tanh(\beta\Gamma(1-s)/M)\f$. Neighbouring equal
 * copies are bonded with probability \f$1-e^{-2J_\perp}\f$
 * (Swendsen-Wang on the chain), and every resulting segment is flipped with
 * Metropolis probability \f$\min(1, e^{-s\beta\Delta E/M})\f$, where
 * \f$\Delta E\f$ is the classical energy change summed over the segment.
 *
 * - \f$J_\perp\to 0\f$: every segment is one copy, i.e. the move reduces to
 *   single spin flip Metropolis.
 * - \f$J_\perp\to\infty\f$: the whole ring flips at once (a "global move"),
 *   so the replicas keep moving even when they are locked together.
 *
 * The bond probability \f$e^{-2J_\perp}=\tanh(\beta\Gamma(1-s)/M)\f$ is
 * evaluated directly, so \f$s=1\f$ is well defined.
 *
 * @tparam GraphType graph type (Dense, Sparse or CSRSparse)
 */
template <typename GraphType>
struct TrotterClusterFlip<system::TransverseIsing<GraphType>> {
  using QIsing = system::TransverseIsing<GraphType>;
  using FloatType = typename GraphType::value_type;

  template <typename RandomNumberEngine>
  inline static void
  update(QIsing &system, RandomNumberEngine &random_number_engine,
         const utility::TransverseFieldUpdaterParameter &parameter) {
    const std::size_t num_classical_spins = system.num_classical_spins;
    const std::size_t num_trotter_slices = system.trotter_spins.cols();
    const double M = static_cast<double>(num_trotter_slices);
    const double classical_beta = parameter.s * parameter.beta / M;
    // probability that a bond between equal neighbouring copies is cut.
    const double p_cut =
        std::tanh(parameter.beta * system.gamma * (1.0 - parameter.s) / M);

    std::uniform_real_distribution<double> urd(0.0, 1.0);
    auto &spins = system.trotter_spins;
    std::vector<char> bond(num_trotter_slices);

    for (std::size_t i = 0; i < num_classical_spins; ++i) {
      // bond[t]: copies t and t+1 (mod M) belong to the same segment.
      std::size_t first_cut = num_trotter_slices;
      for (std::size_t t = 0; t < num_trotter_slices; ++t) {
        const std::size_t u = t + 1 == num_trotter_slices ? 0 : t + 1;
        bond[t] = spins(i, t) == spins(i, u) && urd(random_number_engine) >= p_cut;
        if (!bond[t] && first_cut == num_trotter_slices) {
          first_cut = t;
        }
      }

      if (first_cut == num_trotter_slices) {
        // The ring is a single segment.
        FloatType dE = 0;
        for (std::size_t t = 0; t < num_trotter_slices; ++t) {
          dE += system.dE(i, t);
        }
        if (accept(dE, classical_beta, urd, random_number_engine)) {
          for (std::size_t t = 0; t < num_trotter_slices; ++t) {
            flip(system, i, t);
          }
        }
        continue;
      }

      // Walk the segments, starting right after the first cut.
      std::size_t start = first_cut + 1 == num_trotter_slices ? 0 : first_cut + 1;
      std::size_t visited = 0;
      while (visited < num_trotter_slices) {
        std::size_t t = start;
        std::size_t length = 1;
        FloatType dE = system.dE(i, t);
        while (bond[t]) {
          t = t + 1 == num_trotter_slices ? 0 : t + 1;
          dE += system.dE(i, t);
          ++length;
        }
        if (accept(dE, classical_beta, urd, random_number_engine)) {
          std::size_t k = start;
          for (std::size_t l = 0; l < length; ++l) {
            flip(system, i, k);
            k = k + 1 == num_trotter_slices ? 0 : k + 1;
          }
        }
        visited += length;
        start = t + 1 == num_trotter_slices ? 0 : t + 1;
      }
    }
  }

private:
  template <typename Dist, typename RandomNumberEngine>
  inline static bool accept(const FloatType dE, const double classical_beta,
                            Dist &urd, RandomNumberEngine &rng) {
    return dE <= 0 || std::exp(-classical_beta * dE) > urd(rng);
  }

  inline static void flip(QIsing &system, const std::size_t i,
                          const std::size_t t) {
    auto &spins = system.trotter_spins;
    system.dE.col(t) +=
        4 * spins(i, t) *
        (system.interaction.row(i).transpose().cwiseProduct(spins.col(t)));
    system.dE(i, t) *= -1;
    spins(i, t) *= -1;
  }
};

} // namespace updater
} // namespace openjij
