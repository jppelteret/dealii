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

#ifndef dealii_weakforms_symbolic_info_h
#define dealii_weakforms_symbolic_info_h

#include <deal.II/base/config.h>

#include <string>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{

  /**
   * A data structure that defines the labels to be used
   * to contruct symbolic variables identifiers.
   *
   * @note It is critical to ensure that the labels are
   * unique. If not then there is the possibility that one
   * can generate conflicting symbolic expressions that
   * will not be detected during their use.
   */
  struct SymbolicNames
  {
    /**
     * Default constructor
     */
    SymbolicNames(const std::string dof_value          = "U",
                  const std::string test_function      = "d",
                  const std::string trial_solution     = "D",
                  const std::string shape_function     = "Nx",
                  const std::string JxW                = "JxW",
                  const std::string gradient           = "Grad",
                  const std::string symmetric_gradient = "symm_Grad",
                  const std::string divergence         = "Div",
                  const std::string curl               = "Curl",
                  const std::string hessian            = "Hessian",
                  const std::string third_derivative   = "3rd_Derivative");

    /**
     * Default destructor
     */
    virtual ~SymbolicNames() = default;

    /**
     * Symbol for a degree-of-freedom value
     */
    const std::string dof_value;

    /**
     * Symbol for the test function
     */
    const std::string test_function;

    /**
     * Symbol for the trial solution
     */
    const std::string trial_solution;

    /**
     * Symbol for a shape function
     */
    const std::string shape_function;

    /**
     * Symbol for the integration constant
     */
    const std::string JxW;

    /**
     * Symbol for the gradient operator
     */
    const std::string gradient;

    /**
     * Symbol for the symmetric gradient operator
     */
    const std::string symmetric_gradient;

    /**
     * Symbol for the divergence operator
     */
    const std::string divergence;

    /**
     * Symbol for the curl operator
     */
    const std::string curl;

    /**
     * Symbol for the hessian
     */
    const std::string hessian;

    /**
     * Symbol for third derivative
     */
    const std::string third_derivative;
  }; // struct SymbolicNames

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_symbolic_info_h
