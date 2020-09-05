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
  template <typename SubDomainType>
  class Integral
  {
  public:
    template <typename NumberType>
    using value_type = double;

    Integral(const std::set<SubDomainType> &subdomains)
      : subdomains(subdomains)
    {}

    bool
    integrate_over_entire_domain() const
    {
      constexpr SubDomainType invalid_index = -1;
      return subdomains.empty() ||
             (subdomains.size() == 1 && *subdomains.begin() == invalid_index);
    }

    const std::set<SubDomainType> &
    get_subdomains() const
    {
      return subdomains;
    }

    // ----  Ascii ----

    std::string
    as_ascii(const SymbolicDecorations &decorator) const
    {
      return get_infinitesimal_symbol_ascii(decorator);
    }

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const = 0;

    virtual std::string
    get_infinitesimal_symbol_ascii(
      const SymbolicDecorations &decorator) const = 0;

    // ---- LaTeX ----

    std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      return get_infinitesimal_symbol_latex(decorator);
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const = 0;

    virtual std::string
    get_infinitesimal_symbol_latex(
      const SymbolicDecorations &decorator) const = 0;

  protected:
    bool
    integrate_on_subdomain(const SubDomainType &idx) const
    {
      if (integrate_over_entire_domain())
        return true;

      return subdomains.find(idx) != subdomains.end();
    }

    // Dictate whether to integrate over the whole
    // volume / boundary / interface, or just a
    // part of it. The invalid index SubDomainType(-1)
    // also indicates that the entire domain is to be
    // integrated over.
    const std::set<SubDomainType> subdomains;
  };



  class VolumeIntegral : public Integral<types::material_id>
  {
  public:
    using subdomain_t = types::material_id;

    VolumeIntegral(const std::set<subdomain_t> &subregions)
      : Integral<subdomain_t>(subregions)
    {}

    VolumeIntegral()
      : VolumeIntegral(std::set<subdomain_t>{})
    {}

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_ascii.volume;
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_latex.volume;
    }

    std::string
    get_infinitesimal_symbol_ascii(
      const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_ascii.infinitesimal_element_volume;
    }

    std::string
    get_infinitesimal_symbol_latex(
      const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_latex.infinitesimal_element_volume;
    }

    template <typename CellIteratorType>
    bool
    integrate_on_cell(const CellIteratorType &cell) const
    {
      return integrate_on_subdomain(cell->material_id());
    }
  };



  class BoundaryIntegral : public Integral<types::boundary_id>
  {
  public:
    using subdomain_t = types::boundary_id;

    BoundaryIntegral(const std::set<subdomain_t> &boundaries)
      : Integral<subdomain_t>(boundaries)
    {}

    BoundaryIntegral()
      : BoundaryIntegral(std::set<subdomain_t>{})
    {}

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_ascii.boundary;
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_latex.boundary;
    }

    std::string
    get_infinitesimal_symbol_ascii(
      const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_ascii.infinitesimal_element_boundary_area;
    }

    std::string
    get_infinitesimal_symbol_latex(
      const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_latex.infinitesimal_element_boundary_area;
    }

    template <typename CellIteratorType>
    bool
    integrate_on_face(const CellIteratorType &cell,
                      const unsigned int      face) const
    {
      if (!cell->face(face)->at_boundary())
        return false;

      return integrate_on_subdomain(cell->face(face)->boundary_id());
    }
  };



  class InterfaceIntegral : public Integral<types::manifold_id>
  {
  public:
    using subdomain_t = types::manifold_id;

    InterfaceIntegral(const std::set<subdomain_t> interfaces)
      : Integral<subdomain_t>(interfaces)
    {}

    InterfaceIntegral()
      : InterfaceIntegral(std::set<subdomain_t>{})
    {}

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_ascii.interface;
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_latex.interface;
    }

    std::string
    get_infinitesimal_symbol_ascii(
      const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_ascii.infinitesimal_element_interface_area;
    }

    std::string
    get_infinitesimal_symbol_latex(
      const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_latex.infinitesimal_element_interface_area;
    }

    template <typename CellIteratorType>
    bool
    integrate_on_face(const CellIteratorType &cell,
                      const unsigned int      face) const
    {
      if (cell->face(face)->at_boundary())
        return false;

      return integrate_on_subdomain(cell->face(face)->manifold_id());
    }
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
    template <typename NumberType,
              typename IntegralType_,
              typename IntegrandType_>
    class UnaryOp<IntegralType_,
                  UnaryOpCodes::value,
                  NumberType,
                  IntegrandType_>
    {
      static_assert(!is_symbolic_integral<IntegrandType_>::value,
                    "Cannot integrate an integral!");

    public:
      using IntegralType  = IntegralType_;
      using IntegrandType = IntegrandType_;

      template <typename NumberType2>
      using value_type =
        typename IntegralType::template value_type<NumberType2>;

      template <typename NumberType2>
      using return_type = std::vector<value_type<NumberType2>>;

      static const int rank = 0;

      // static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const IntegralType & integral_operation,
                       const IntegrandType &integrand)
        : integral_operation(integral_operation)
        , integrand(integrand)
      {}

      bool
      integrate_over_entire_domain() const
      {
        return integral_operation.integrate_over_entire_domain();
      }

      const std::set<typename IntegralType::subdomain_t> &
      get_subdomains() const
      {
        return integral_operation.get_subdomains();
      }

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.unary_op_integral_as_ascii(integrand,
                                                    integral_operation);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return decorator.unary_op_integral_as_latex(integrand,
                                                    integral_operation);
      }

      // ===== Section: Construct assembly operation =====

      const IntegralType &
      get_integral_operation() const
      {
        return integral_operation;
      }

      const IntegrandType &
      get_integrand() const
      {
        return integrand;
      }

      // ===== Section: Perform actions =====

      UpdateFlags
      get_update_flags() const
      {
        return get_integrand().get_update_flags() |
               UpdateFlags::update_JxW_values;
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
      const IntegralType  integral_operation;
      const IntegrandType integrand;
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  template <typename NumberType = double, typename Integrand>
  WeakForms::Operators::UnaryOp<WeakForms::VolumeIntegral,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                Integrand>
  value(const WeakForms::VolumeIntegral &operand, const Integrand &integrand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = VolumeIntegral;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType, Integrand>;

    return OpType(operand, integrand);
  }


  template <typename NumberType = double, typename Integrand>
  WeakForms::Operators::UnaryOp<WeakForms::BoundaryIntegral,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                Integrand>
  value(const WeakForms::BoundaryIntegral &operand, const Integrand &integrand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = BoundaryIntegral;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType, Integrand>;

    return OpType(operand, integrand);
  }



  template <typename NumberType = double, typename Integrand>
  WeakForms::Operators::UnaryOp<WeakForms::InterfaceIntegral,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                Integrand>
  value(const WeakForms::InterfaceIntegral &operand, const Integrand &integrand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = InterfaceIntegral;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType, Integrand>;

    return OpType(operand, integrand);
  }



  template <typename NumberType = double,
            typename Integrand,
            typename IntegralType,
            typename = typename std::enable_if<WeakForms::is_symbolic_integral<
              typename std::decay<IntegralType>::type>::value>::type>
  // auto
  WeakForms::Operators::UnaryOp<IntegralType,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                Integrand>
  integrate(const Integrand &integrand, const IntegralType &integral)
  {
    return value(integral, integrand);
  }

  // template <typename NumberType = double,
  //           typename Integrand,
  //           typename IntegralType,
  //           typename = typename std::enable_if<
  //             WeakForms::is_symbolic_integral<IntegralType>::value>::type>
  // auto
  // integrate(const Integrand &integrand, const IntegralType &integral, const
  // std::set<typename IntegralType::subdomain_t> &subdomains)
  // {
  //   return value(integral, integrand);
  // }


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
  struct is_symbolic_volume_integral<VolumeIntegral> : std::true_type
  {};

  template <>
  struct is_symbolic_boundary_integral<BoundaryIntegral> : std::true_type
  {};

  template <>
  struct is_symbolic_interface_integral<InterfaceIntegral> : std::true_type
  {};

  // Unary operators

  template <typename NumberType,
            typename Integrand,
            enum Operators::UnaryOpCodes OpCode>
  struct is_symbolic_volume_integral<
    Operators::UnaryOp<VolumeIntegral, OpCode, NumberType, Integrand>>
    : std::true_type
  {};

  template <typename NumberType,
            typename Integrand,
            enum Operators::UnaryOpCodes OpCode>
  struct is_symbolic_boundary_integral<
    Operators::UnaryOp<BoundaryIntegral, OpCode, NumberType, Integrand>>
    : std::true_type
  {};

  template <typename NumberType,
            typename Integrand,
            enum Operators::UnaryOpCodes OpCode>
  struct is_symbolic_interface_integral<
    Operators::UnaryOp<InterfaceIntegral, OpCode, NumberType, Integrand>>
    : std::true_type
  {};

  template <typename NumberType,
            typename Integrand,
            enum Operators::UnaryOpCodes OpCode>
  struct is_unary_op<
    Operators::UnaryOp<VolumeIntegral, OpCode, NumberType, Integrand>>
    : std::true_type
  {};

  template <typename NumberType,
            typename Integrand,
            enum Operators::UnaryOpCodes OpCode>
  struct is_unary_op<
    Operators::UnaryOp<BoundaryIntegral, OpCode, NumberType, Integrand>>
    : std::true_type
  {};

  template <typename NumberType,
            typename Integrand,
            enum Operators::UnaryOpCodes OpCode>
  struct is_unary_op<
    Operators::UnaryOp<InterfaceIntegral, OpCode, NumberType, Integrand>>
    : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_integral_h
