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

#ifndef dealii_weakforms_integral_h
#define dealii_weakforms_integral_h

#include <deal.II/base/config.h>

// TODO: Move FeValuesViews::[Scalar/Vector/...]::Output<> into another header??
#include <deal.II/fe/fe_values.h>

#include <deal.II/weak_forms/operators.h>
#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/type_traits.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  class Integral
  {
  public:
    template <typename NumberType>
    using value_type = double;

    Integral(const std::string &        symbol_ascii,
             const std::string &        symbol_latex,
             const SymbolicDecorations &decorator = SymbolicDecorations())
      : symbol_ascii(symbol_ascii)
      , symbol_latex(symbol_latex != "" ? symbol_latex : symbol_ascii)
      , decorator(decorator)
    {}

    const SymbolicDecorations &
    get_decorator() const
    {
      return decorator;
    }

    // ----  Ascii ----

    std::string
    as_ascii() const
    {
      // return get_decorator().unary_op_operand_as_ascii(*this);
      return get_symbol_ascii();
    }

    std::string
    get_symbol_ascii() const
    {
      return symbol_ascii;
    }

    const SymbolicNamesAscii &
    get_naming_ascii() const
    {
      return get_decorator().naming_ascii;
    }

    // ---- LaTeX ----

    std::string
    as_latex() const
    {
      // return get_decorator().unary_op_operand_as_latex(*this);
      return get_symbol_latex();
    }

    std::string
    get_symbol_latex() const
    {
      return symbol_latex;
    }

    const SymbolicNamesLaTeX &
    get_naming_latex() const
    {
      return get_decorator().naming_latex;
    }

  protected:
    const std::string symbol_ascii;
    const std::string symbol_latex;

    const SymbolicDecorations decorator;
  };



  class VolumeIntegral : public Integral
  {
  public:
    VolumeIntegral(const SymbolicDecorations &decorator = SymbolicDecorations())
      : Integral(decorator.naming_ascii.infinitesimal_element_volume,
                 decorator.naming_latex.infinitesimal_element_volume,
                 decorator)
    {}
  };



  class BoundaryIntegral : public Integral
  {
  public:
    BoundaryIntegral(
      const SymbolicDecorations &decorator = SymbolicDecorations())
      : Integral(decorator.naming_ascii.infinitesimal_element_boundary_area,
                 decorator.naming_latex.infinitesimal_element_boundary_area,
                 decorator)
    {}
  };



  class InterfaceIntegral : public Integral
  {
  public:
    InterfaceIntegral(
      const SymbolicDecorations &decorator = SymbolicDecorations())
      : Integral(decorator.naming_ascii.infinitesimal_element_interface_area,
                 decorator.naming_latex.infinitesimal_element_interface_area,
                 decorator)
    {}
  };



  // class CurveIntegral : public Integral
  // {
  // public:
  //   CurveIntegral(const SymbolicDecorations &decorator =
  //   SymbolicDecorations())
  //     : Integral(decorator.naming_ascii.infinitesimal_element_curve_length,
  //                    decorator.naming_latex.infinitesimal_element_curve_length,
  //                    decorator)
  //   {}
  // };

} // namespace WeakForms



// #ifndef DOXYGEN


namespace WeakForms
{
  template<>
  struct is_symbolic_integral<Integral> : std::true_type
  {};

template<>
  struct is_symbolic_integral<VolumeIntegral> : std::true_type
  {};

template<>
  struct is_symbolic_integral<BoundaryIntegral> : std::true_type
  {};

template<>
  struct is_symbolic_integral<InterfaceIntegral> : std::true_type
  {};

} // namespace WeakForms


// #endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_integral_h
