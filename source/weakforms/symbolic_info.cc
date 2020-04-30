// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/weakforms/symbolic_info.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{

  SymbolicNames::SymbolicNames(const std::string dof_value,
                               const std::string test_function,
                               const std::string trial_solution,
                               const std::string shape_function,
                               const std::string JxW,
                               const std::string gradient,
                               const std::string symmetric_gradient,
                               const std::string divergence,
                               const std::string curl,
                               const std::string hessian,
                               const std::string third_derivative)
    : dof_value(dof_value)
    , test_function(test_function)
    , trial_solution(trial_solution)
    , shape_function(shape_function)
    , JxW(JxW)
    , gradient(gradient)
    , symmetric_gradient(symmetric_gradient)
    , divergence(divergence)
    , curl(curl)
    , hessian(hessian)
    , third_derivative(third_derivative)
  {}

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE
