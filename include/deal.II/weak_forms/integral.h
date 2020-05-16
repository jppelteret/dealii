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

#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/type_traits.h>
#include <deal.II/weak_forms/unary_operators.h>


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



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    /**
     * Get the weighted Jacobians for numerical integration
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <typename NumberType, typename Integrand>
    class UnaryOp<Integral, UnaryOpCodes::value, NumberType, Integrand>
    {
      using Op = Integral;

    public:
      template <typename NumberType2>
      using value_type = typename Op::template value_type<NumberType2>;

      template <typename NumberType2>
      using return_type = std::vector<value_type<NumberType2>>;

      static const int rank = 0;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &operand, const Integrand &integrand)
        : operand(operand)
        , integrand(integrand)
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        return operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        const auto &decorator = operand.get_decorator();
        return decorator.unary_op_integral_as_ascii(integrand, operand);
      }

      std::string
      as_latex() const
      {
        const auto &decorator = operand.get_decorator();
        return decorator.unary_op_integral_as_latex(integrand, operand);
      }

      // Return single entry
      template <typename NumberType2, int dim, int spacedim>
      value_type<NumberType2>
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values.JxW(q_point);
      }

      /**
       * Return all JxW values at all quadrature points
       */
      template <typename NumberType2, int dim, int spacedim>
      const return_type<NumberType2> &
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const
      {
        return fe_values.get_JxW_values();
      }

    private:
      const Op &       operand;
      const Integrand &integrand;
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  template <typename NumberType = double, typename Integrand>
  WeakForms::Operators::UnaryOp<WeakForms::Integral,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                Integrand>
  value(const WeakForms::VolumeIntegral &operand, const Integrand &integrand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = Integral;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType, Integrand>;

    return OpType(operand, integrand);
  }


  template <typename NumberType = double, typename Integrand>
  WeakForms::Operators::UnaryOp<WeakForms::Integral,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                Integrand>
  value(const WeakForms::BoundaryIntegral &operand, const Integrand &integrand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = Integral;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType, Integrand>;

    return OpType(operand, integrand);
  }



  template <typename NumberType = double, typename Integrand>
  WeakForms::Operators::UnaryOp<WeakForms::Integral,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                Integrand>
  value(const WeakForms::InterfaceIntegral &operand, const Integrand &integrand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = Integral;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType, Integrand>;

    return OpType(operand, integrand);
  }

  template <typename NumberType = double,
            typename Integrand,
            typename IntegralType,
            typename = typename std::enable_if<
              WeakForms::is_symbolic_integral<IntegralType>::value>::type>
  auto
  integrate(const Integrand &integrand, const IntegralType &integral)
  {
    return value(integral, integrand);
  }


  // WeakForms::Operators::UnaryOp<WeakForms::Integral,
  //                               WeakForms::Operators::UnaryOpCodes::value>
  // value(const WeakForms::CurveIntegral &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = Integral;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::value>;

  //   return OpType(operand);
  // }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Decorator classes

  template <>
  struct is_symbolic_integral<Integral> : std::true_type
  {};

  template <>
  struct is_symbolic_integral<VolumeIntegral> : std::true_type
  {};

  template <>
  struct is_symbolic_integral<BoundaryIntegral> : std::true_type
  {};

  template <>
  struct is_symbolic_integral<InterfaceIntegral> : std::true_type
  {};

  // Unary operators

  template <typename NumberType,
            typename Integrand,
            enum Operators::UnaryOpCodes OpCode>
  struct is_symbolic_integral<
    Operators::UnaryOp<WeakForms::Integral, OpCode, NumberType, Integrand>>
    : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_integral_h
