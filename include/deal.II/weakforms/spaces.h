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

#ifndef dealii_weakforms_spaces_h
#define dealii_weakforms_spaces_h

#include <deal.II/base/config.h>

// TODO: Move FeValuesViews::[Scalar/Vector/...]::Output<> into another header??
#include <deal.II/fe/fe_values.h>

#include <deal.II/weakforms/operators.h>
#include <deal.II/weakforms/symbolic_info.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{

  template <int dim, int spacedim>
  class Space
  {
    using OpType =
      Operators::UnaryOp<Space<dim, spacedim>, Operators::UnaryOpCodes::value>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = dim;

    /**
     * Dimension of the space in which this object operates.
     */
    static const unsigned int space_dimension = spacedim;

    template <typename NumberType>
    using value_type =
      typename FEValuesViews::Scalar<dim, spacedim>::template OutputType<
        NumberType>::value_type;

    template <typename NumberType>
    using gradient_type =
      typename FEValuesViews::Scalar<dim, spacedim>::template OutputType<
        NumberType>::gradient_type;

    template <typename NumberType>
    using hessian_type =
      typename FEValuesViews::Scalar<dim, spacedim>::template OutputType<
        NumberType>::hessian_type;
        
    template <typename NumberType>
    using laplacian_type =
      typename FEValuesViews::Scalar<dim, spacedim>::template OutputType<
        NumberType>::laplacian_type;

    template <typename NumberType>
    using third_derivative_type =
      typename FEValuesViews::Scalar<dim, spacedim>::template OutputType<
        NumberType>::third_derivative_type;

    // Full space
    Space(const std::string &       symbol_ascii,
          const std::string &       symbol_latex,
          const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
          const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
      : field_ascii("")
      , field_latex("")
      , symbol_ascii(symbol_ascii)
      , symbol_latex(symbol_latex != "" ? symbol_latex : symbol_ascii)
      , naming_ascii(naming_ascii)
      , naming_latex(naming_latex)
    {}

    // ----  Ascii ----

    std::string
    as_ascii() const
    {
      return internal::unary_op_operand_as_ascii(*this);
    }

    std::string
    get_field_ascii() const
    {
      return field_ascii;
    }

    std::string
    get_symbol_ascii() const
    {
      return symbol_ascii;
    }

    const SymbolicNamesAscii &
    get_naming_ascii() const
    {
      return naming_ascii;
    }

    // ---- LaTeX ----

    std::string
    as_latex() const
    {
      return internal::unary_op_operand_as_latex(*this);
    }

    std::string
    get_field_latex() const
    {
      return field_latex;
    }

    std::string
    get_symbol_latex() const
    {
      return symbol_latex;
    }

    const SymbolicNamesLaTeX &
    get_naming_latex() const
    {
      return naming_latex;
    }

  protected:
    // Create a subspace
    Space(const std::string &       field_ascii,
          const std::string &       field_latex,
          const std::string &       symbol_ascii,
          const std::string &       symbol_latex,
          const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
          const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
      : field_ascii(field_ascii)
      , field_latex(field_latex != "" ? field_latex : field_ascii)
      , symbol_ascii(symbol_ascii)
      , symbol_latex(symbol_latex != "" ? symbol_latex : symbol_ascii)
      , naming_ascii(naming_ascii)
      , naming_latex(naming_latex)
    {}

    const std::string field_ascii;
    const std::string field_latex;

    const std::string symbol_ascii;
    const std::string symbol_latex;

    const SymbolicNamesAscii naming_ascii;
    const SymbolicNamesLaTeX naming_latex;
  };



  template <int dim, int spacedim = dim>
  class TestFunction : public Space<dim, spacedim>
  {
  public:
    // Full space
    TestFunction(const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
                 const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
      : TestFunction(naming_ascii.solution_field,
                     naming_latex.solution_field,
                     naming_ascii,
                     naming_latex)
    {}

  protected:
    // Subspace
    TestFunction(const std::string         field_ascii,
                 const std::string         field_latex,
                 const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
                 const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
      : Space<dim, spacedim>(field_ascii,
                             field_latex,
                             naming_ascii.test_function,
                             naming_latex.test_function,
                             naming_ascii,
                             naming_latex)
    {}
  };


  template <int dim, int spacedim = dim>
  class TrialSolution : public Space<dim, spacedim>
  {
  public:
    // Full space
    TrialSolution(const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
                  const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
      : TrialSolution(naming_ascii.solution_field,
                      naming_latex.solution_field,
                      naming_ascii,
                      naming_latex)
    {}

  protected:
    // Subspace
    TrialSolution(const std::string         field_ascii,
                  const std::string         field_latex,
                  const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
                  const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
      : Space<dim, spacedim>(field_ascii,
                             field_latex,
                             naming_ascii.trial_solution,
                             naming_latex.trial_solution,
                             naming_ascii,
                             naming_latex)
    {}
  };



  template <int dim, int spacedim = dim>
  class FieldSolution : public Space<dim, spacedim>
  {
  public:
    // Full space
    FieldSolution(const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
                  const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
      : FieldSolution("", "", naming_ascii, naming_latex)
    {}

  protected:
    // Subspace
    FieldSolution(const std::string         field_ascii,
                  const std::string         field_latex,
                  const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
                  const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
      : Space<dim, spacedim>(field_ascii,
                             field_latex,
                             naming_ascii.solution_field,
                             naming_latex.solution_field,
                             naming_ascii,
                             naming_latex)
    {}
  };



  // namespace Linear
  // {
  //   template <int dim, int spacedim = dim>
  //   using TestFunction = WeakForms::TestFunction<dim, spacedim>;

  //   template <int dim, int spacedim = dim>
  //   using TrialSolution = WeakForms::TrialSolution<dim, spacedim>;

  //   template <int dim, int spacedim = dim>
  //   using Solution = WeakForms::Solution<dim, spacedim>;
  // } // namespace Linear



  namespace NonLinear
  {
    template <int dim, int spacedim = dim>
    using Variation = WeakForms::TestFunction<dim, spacedim>;

    template <int dim, int spacedim = dim>
    using Linearization = WeakForms::TrialSolution<dim, spacedim>;

    template <int dim, int spacedim = dim>
    using FieldSolution = WeakForms::FieldSolution<dim, spacedim>;
  } // namespace NonLinear



  namespace SpaceViews
  {
    class Scalar
    {};
    class Vector
    {};
    class Tensor
    {};
    class SymmetricTensor
    {};
  } // namespace SpaceViews

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_spaces_h