#pragma once

#include <cstdint>
#include <utility>
#include <cmath>
#include <algorithm>

namespace openjij {
namespace utility {

const double ZERO_TOL = 1e-13;

template <typename RandomNumberEngine>
std::pair<std::int64_t, double>
FindMinimumIntegerQuadratic(
  const double a,
  const double b,
  const std::int64_t l,
  const std::int64_t u,
  const std::int64_t x,
  RandomNumberEngine &random_number_engine
) {
  if ((std::abs(a) < ZERO_TOL) && (std::abs(b) < ZERO_TOL)) {
    return {x + std::uniform_int_distribution<std::int64_t>(l, u)(random_number_engine), 0.0};
  }

  if (a > 0) {
    const auto clamped_center = std::clamp(-b / (2.0 * a), static_cast<double>(l), static_cast<double>(u));
    const auto dx = static_cast<std::int64_t>(std::round(clamped_center));
    const double energy_diff = a * dx * dx + b * dx;
    return {x + dx, energy_diff};
  } else {
    const double el = a * l * l + b * l;
    const double eu = a * u * u + b * u;
    if (el <= eu) {
      return {x + l, el};
    } else {
      return {x + u, eu};
    }
  }
}

template <typename RandomNumberEngine>
std::pair<std::int64_t, double>
FindMinimumIntegerCubic(
  const double a,
  const double b,
  const double c,
  const std::int64_t l,
  const std::int64_t u,
  const std::int64_t x,
  RandomNumberEngine &random_number_engine
) {

  if (std::abs(a) >= ZERO_TOL) {
    const double val_l = a * l * l * l + b * l * l + c * l;
    const double val_u = a * u * u * u + b * u * u + c * u;

    std::int64_t best_dx;
    double min_val;

    if (val_l <= val_u) {
      best_dx = l;
      min_val = val_l;
    } else {
      best_dx = u;
      min_val = val_u;
    }

    const double delta = b * b - 3 * a * c;
    if (delta >= 0) {
      const double sqrt_delta = std::sqrt(delta);
      const double r1 = (-b + sqrt_delta) / (3 * a);
      const double r2 = (-b - sqrt_delta) / (3 * a);

      const double local_min_root = (a > 0) ? std::max(r1, r2) : std::min(r1, r2);

      if (l < local_min_root && local_min_root < u) {
        const auto cand1 = static_cast<std::int64_t>(std::floor(local_min_root));
        const double val1 = a * cand1 * cand1 * cand1 + b * cand1 * cand1 + c * cand1;
        if (val1 < min_val) {
          min_val = val1;
          best_dx = cand1;
        }

        const auto cand2 = static_cast<std::int64_t>(std::ceil(local_min_root));
        if (cand2 <= u) {
          const double val2 = a * cand2 * cand2 * cand2 + b * cand2 * cand2 + c * cand2;
          if (val2 < min_val) {
            min_val = val2;
            best_dx = cand2;
          }
        }
      }
    }
    return {x + best_dx, min_val};
  } else {
    return FindMinimumIntegerQuadratic(b, c, l, u, x, random_number_engine);
  }
}

template <typename RandomNumberEngine>
std::pair<std::int64_t, double>
FindMinimumIntegerQuartic(
  const double a,
  const double b,
  const double c,
  const double d,
  const std::int64_t l,
  const std::int64_t u,
  const std::int64_t x,
  RandomNumberEngine &random_number_engine
) {

  if (std::abs(a) >= ZERO_TOL) {
    const auto f = [&](double dx) {
      return a * dx * dx * dx * dx + b * dx * dx * dx + c * dx * dx + d * dx;
    };

    const double val_l = f(static_cast<double>(l));
    const double val_u = f(static_cast<double>(u));
    const double m_pi = 3.14159265358979323846;

    std::int64_t best_dx;
    double min_val;

    if (val_l <= val_u) {
      best_dx = l;
      min_val = val_l;
    } else {
      best_dx = u;
      min_val = val_u;
    }
    
    const double A = 4.0 * a;
    const double B = 3.0 * b;
    const double C = 2.0 * c;
    const double D = d;
    const double p = (3.0 * A * C - B * B) / (3.0 * A * A);
    const double q = (2.0 * B * B * B - 9.0 * A * B * C + 27.0 * A * A * D) / (27.0 * A * A * A);

    const double delta = std::pow(q / 2.0, 2) + std::pow(p / 3.0, 3);
    
    const auto check_and_update = 
      [&](double r, std::int64_t current_best_dx, double current_min_val) 
      -> std::pair<std::int64_t, double> {
      const double f_double_prime = 12.0 * a * r * r + 6.0 * b * r + 2.0 * c;
      if (f_double_prime > 0 && l < r && r < u) {
        const double floor_r = std::floor(r);
        const double v1 = f(floor_r);
        if (v1 < current_min_val) {
          current_min_val = v1;
          current_best_dx = static_cast<std::int64_t>(floor_r);
        }
        
        const double ceil_r = std::ceil(r);
        if (static_cast<std::int64_t>(ceil_r) <= u) {
          const double v2 = f(ceil_r);
          if (v2 < current_min_val) {
            current_min_val = v2;
            current_best_dx = static_cast<std::int64_t>(ceil_r);
          }
        }
      }
      return {current_best_dx, current_min_val};
    };

    if (delta >= 0) {
      const double sqrt_delta = std::sqrt(delta);
      const double u_cbrt = std::cbrt(-q / 2.0 + sqrt_delta);
      const double v_cbrt = std::cbrt(-q / 2.0 - sqrt_delta);
      const double r1 = u_cbrt + v_cbrt - B / (3.0 * A);
      const auto result = check_and_update(r1, best_dx, min_val);
      best_dx = result.first;
      min_val = result.second;
    } else {
      const double r_term = std::sqrt(-std::pow(p, 3) / 27.0);
      const double theta = std::acos(-q / (2.0 * r_term));
      
      const double common_term = 2.0 * std::cbrt(r_term);
      const double shift = B / (3.0 * A);
      
      const double r1 = common_term * std::cos(theta / 3.0) - shift;
      auto result1 = check_and_update(r1, best_dx, min_val);
      best_dx = result1.first;
      min_val = result1.second;
      
      const double r2 = common_term * std::cos((theta + 2.0 * m_pi) / 3.0) - shift;
      auto result2 = check_and_update(r2, best_dx, min_val);
      best_dx = result2.first;
      min_val = result2.second;
      
      const double r3 = common_term * std::cos((theta + 4.0 * m_pi) / 3.0) - shift;
      auto result3 = check_and_update(r3, best_dx, min_val);
      best_dx = result3.first;
      min_val = result3.second;
    }
    return {x + best_dx, min_val};
  } else {
    return FindMinimumIntegerCubic(b, c, d, l, u, x, random_number_engine);
  }
}

}
}
