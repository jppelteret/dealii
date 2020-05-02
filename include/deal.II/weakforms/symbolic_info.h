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
  namespace internal
  {
    // std::string
    // wrap_test_field_ascii(const std::string &test, const std::string &field)
    // {
    //   return test + "{" + field + "}";
    // }
    // std::string
    // wrap_test_field_latex(const std::string &test, const std::string &field)
    // {
    //   return test + "_{" + field + "}";
    // }

    // std::string
    // wrap_trial_field_ascii(const std::string &trial, const std::string
    // &field)
    // {
    //   return trial + "{" + field + "}";
    // }

    // std::string
    // wrap_trial_field_latex(const std::string &trial, const std::string
    // &field)
    // {
    //   return trial + "_{" + field + "}";
    // }

    // std::string
    // wrap_solution_field_ascii(const std::string &soln, const std::string
    // &field)
    // {
    //   return soln + "{" + field + "}";
    // }

    // std::string
    // wrap_solution_field_latex(const std::string &soln, const std::string
    // &field)
    // {
    //   return soln + "_{" + field + "}";
    // }

    // std::string
    // wrap_operator_ascii(const std::string &op, const std::string &expression)
    // {
    //   return op + "(" + expression + ")";
    // }

    // std::string
    // wrap_term_ascii(const std::string &term)
    // {
    //   return "[" + term + "]";
    // }
  } // namespace internal


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
    explicit SymbolicNames(const std::string solution_field,
                           const std::string test_function,
                           const std::string trial_solution,
                           const std::string shape_function,
                           const std::string dof_value,
                           const std::string JxW,
                           const std::string value,
                           const std::string gradient,
                           const std::string symmetric_gradient,
                           const std::string divergence,
                           const std::string curl,
                           const std::string hessian,
                           const std::string laplacian,
                           const std::string third_derivative);

    /**
     * Default destructor
     */
    virtual ~SymbolicNames() = default;

    /**
     * Symbol for the solution field
     */
    const std::string solution_field;

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
     * Symbol for a degree-of-freedom value
     */
    const std::string dof_value;

    /**
     * Symbol for the integration constant
     */
    const std::string JxW;

    /**
     * Symbol for the value of the operand
     */
    const std::string value;

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
     * Symbol for the Laplacian
     */
    const std::string laplacian;

    /**
     * Symbol for third derivative
     */
    const std::string third_derivative;
  }; // struct SymbolicNames

  struct SymbolicNamesAscii : public SymbolicNames
  {
    /**
     * Default constructor
     */
    explicit SymbolicNamesAscii(
      const std::string solution_field     = "U",
      const std::string test_function      = "d",
      const std::string trial_solution     = "D",
      const std::string shape_function     = "Nx",
      const std::string dof_value          = "v",
      const std::string JxW                = "JxW",
      const std::string value              = "",
      const std::string gradient           = "Grad",
      const std::string symmetric_gradient = "symm_Grad",
      const std::string divergence         = "Div",
      const std::string curl               = "Curl",
      const std::string hessian            = "Hessian",
      const std::string laplacian          = "Laplacian",
      const std::string third_derivative   = "3rd_Derivative");
  }; // struct SymbolicNamesAscii

  struct SymbolicNamesLaTeX : public SymbolicNames
  {
    /**
     * Default constructor
     */
    explicit SymbolicNamesLaTeX(
      const std::string solution_field     = "U",
      const std::string test_function      = "\\delta",
      const std::string trial_solution     = "\\Delta",
      const std::string shape_function     = "N",
      const std::string dof_value          = "\\varphi",
      const std::string JxW                = "\\int",
      const std::string value              = "",
      const std::string gradient           = "\\Nabla",
      const std::string symmetric_gradient = "\\Nabla^{S}",
      const std::string divergence         = "\\Nabla \\cdot",
      const std::string curl               = "\\Nabla \\times",
      const std::string hessian            = "\\Nabla\\Nabla",
      const std::string laplacian          = "\\Nabla^{2}",
      const std::string third_derivative   = "\\Nabla\\Nabla\\Nabla");
  }; // struct SymbolicNamesLaTeX

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_symbolic_info_h
