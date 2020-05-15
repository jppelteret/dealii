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

#ifndef dealii_weakforms_symbolic_decorations_h
#define dealii_weakforms_symbolic_decorations_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

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
    explicit SymbolicNames(
      const std::string solution_field,
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
      const std::string third_derivative,
      const std::string infinitesimal_element_volume,
      const std::string infinitesimal_element_boundary_area,
      const std::string infinitesimal_element_interface_area);

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

    /**
     * Symbol for an infinitesimal volume
     */
    const std::string infinitesimal_element_volume;

    /**
     * Symbol for an infinitesimal boundary surface area
     */
    const std::string infinitesimal_element_boundary_area;

    /**
     * Symbol for an infinitesimal internal interface area
     */
    const std::string infinitesimal_element_interface_area;
  }; // struct SymbolicNames



  struct SymbolicNamesAscii : public SymbolicNames
  {
    /**
     * Default constructor
     */
    explicit SymbolicNamesAscii(
      const std::string solution_field                       = "U",
      const std::string test_function                        = "d",
      const std::string trial_solution                       = "D",
      const std::string shape_function                       = "Nx",
      const std::string dof_value                            = "c",
      const std::string JxW                                  = "JxW",
      const std::string value                                = "",
      const std::string gradient                             = "Grad",
      const std::string symmetric_gradient                   = "symm_Grad",
      const std::string divergence                           = "Div",
      const std::string curl                                 = "Curl",
      const std::string hessian                              = "Hessian",
      const std::string laplacian                            = "Laplacian",
      const std::string third_derivative                     = "3rd_Derivative",
      const std::string infinitesimal_element_volume         = "dV",
      const std::string infinitesimal_element_boundary_area  = "dA",
      const std::string infinitesimal_element_interface_area = "dI");
  }; // struct SymbolicNamesAscii

  struct SymbolicNamesLaTeX : public SymbolicNames
  {
    /**
     * Default constructor
     */
    explicit SymbolicNamesLaTeX(
      const std::string solution_field               = "\\varphi",
      const std::string test_function                = "\\delta",
      const std::string trial_solution               = "\\Delta",
      const std::string shape_function               = "N",
      const std::string dof_value                    = "c",
      const std::string JxW                          = "\\int",
      const std::string value                        = "",
      const std::string gradient                     = "\\nabla",
      const std::string symmetric_gradient           = "\\nabla^{S}",
      const std::string divergence                   = "\\nabla \\cdot",
      const std::string curl                         = "\\nabla \\times",
      const std::string hessian                      = "\\nabla\\nabla",
      const std::string laplacian                    = "\\nabla^{2}",
      const std::string third_derivative             = "\\nabla\\nabla\\nabla",
      const std::string infinitesimal_element_volume = "dV",
      const std::string infinitesimal_element_boundary_area  = "dA",
      const std::string infinitesimal_element_interface_area = "dI");
  }; // struct SymbolicNamesLaTeX


  /**
   * A class to do all decorations
   *
   */
  struct SymbolicDecorations
  {
    SymbolicDecorations(
      const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
      const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX());

    template <typename Operand>
    std::string
    unary_op_operand_as_ascii(const Operand &operand) const
    {
      const std::string field = operand.get_field_ascii();
      if (field == "")
        return operand.get_symbol_ascii();

      return operand.get_symbol_ascii() + "{" + operand.get_field_ascii() + "}";
    }

    template <typename Operand>
    std::string
    unary_op_operand_as_latex(const Operand &operand) const
    {
      const std::string field = operand.get_field_latex();
      if (field == "")
        return operand.get_symbol_latex();

      return operand.get_symbol_latex() + "{" + operand.get_field_latex() + "}";
    }

    template <typename Functor>
    std::string
    unary_op_functor_as_ascii(const Functor &    functor,
                              const unsigned int rank) const
    {
      if (rank == 0)
        return functor.get_symbol_ascii();
      else
        {
          const std::string prefix(rank, '<');
          const std::string suffix(rank, '>');
          return prefix + functor.get_symbol_ascii() + suffix;
        }
    }

    template <typename Functor>
    std::string
    unary_op_functor_as_latex(const Functor &    functor,
                              const unsigned int rank) const
    {
      auto decorate = [&functor](const std::string latex_cmd) {
        return "\\" + latex_cmd + "{" + functor.get_symbol_latex() + "}";
      };

      switch (rank)
        {
          case (0):
            return decorate("mathnormal");
            break;
          case (1):
            return decorate("mathrm");
            break;
          case (2):
            return decorate("mathbf");
            break;
          case (3):
            return decorate("mathfrak");
            break;
          case (4):
            return decorate("mathcal");
            break;
          default:
            break;
        }

      AssertThrow(false, ExcNotImplemented());
      return "";
    }


    /**
     *
     *
     * @param op A string that symbolises the operator that acts on the @p operand.
     * @param operand
     * @return std::string
     */
    std::string
    decorate_with_operator_ascii(const std::string &op,
                                 const std::string &operand) const
    {
      if (op == "")
        return operand;

      return op + "(" + operand + ")";
    }


    /**
     *
     *
     * @param op A string that symbolises the operator that acts on the @p operand.
     * @param operand
     * @return std::string
     */
    std::string
    decorate_with_operator_latex(const std::string &op,
                                 const std::string &operand) const
    {
      if (op == "")
        return operand;

      return op + "\\left\\(" + operand + "\\right\\)";
    }

    std::string
    get_symbol_multiply_latex(const unsigned int n_contracting_indices) const
    {
      switch (n_contracting_indices)
        {
          case (0):
            return " ";
            break;
          case (1):
            return " \\cdot ";
            break;
          case (2):
            return " \\colon ";
            break;
          case (3):
            return " \\vdots ";
            break;
          case (4):
            return " \\colon\\colon ";
            break;
          case (5):
            return " \\vdots\\colon ";
            break;
          case (6):
            return " \\vdots\\vdots ";
            break;
          default:
            return " * ";
            break;
        }

      AssertThrow(false, ExcNotImplemented());
      return " * ";
    }

    const SymbolicNamesAscii naming_ascii;
    const SymbolicNamesLaTeX naming_latex;
  };

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_symbolic_decorations_h
