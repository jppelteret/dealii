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

#ifndef dealii_weakforms_linear_forms_h
#define dealii_weakforms_linear_forms_h

#include <deal.II/base/config.h>

#include <deal.II/weak_forms/integral.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/type_traits.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  template <typename TestSpaceOp, typename Functor>
  class LinearForm
  {
  public:
    explicit LinearForm(const TestSpaceOp &test_space_op,
                        const Functor &    functor_op)
      : test_space_op(test_space_op)
      , functor_op(functor_op)
    {}

    const SymbolicDecorations &
    get_decorator() const
    {
      // Assert(&lhs_operand.get_decorator() == &rhs_operand.get_decorator(),
      // ExcMessage("LHS and RHS operands do not use the same decorator."));
      return test_space_op.get_decorator();
    }

    std::string
    as_ascii() const
    {
      return "(" + test_space_op.as_ascii() + ", " + functor_op.as_ascii() +
             ")";
    }

    std::string
    as_latex() const
    {
      // const std::string lbrace = "\\left\\[";
      // const std::string rbrace = "\\right\\]";
      return "\\left\\[" + test_space_op.as_latex() + " * " +
             functor_op.as_latex() + "\\right\\]";
    }

    // --- Section: Integration ---

    auto
    dV() const
    {
      return integrate(*this, VolumeIntegral(get_decorator()));
    }

    auto
    dV(const std::set<typename VolumeIntegral::subdomain_t> &subdomains) const
    {
      return integrate(*this, VolumeIntegral(subdomains, get_decorator()));
    }

    auto
    dA() const
    {
      return integrate(*this, BoundaryIntegral(get_decorator()));
    }

    auto
    dA(const std::set<typename BoundaryIntegral::subdomain_t> &boundaries) const
    {
      return integrate(*this, BoundaryIntegral(boundaries, get_decorator()));
    }

    auto
    dI() const
    {
      return integrate(*this, InterfaceIntegral(get_decorator()));
    }

    auto
    dI(
      const std::set<typename InterfaceIntegral::subdomain_t> &interfaces) const
    {
      return integrate(*this, InterfaceIntegral(interfaces, get_decorator()));
    }

  private:
    const TestSpaceOp test_space_op;
    const Functor     functor_op;
  };

} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  template <typename TestSpaceOp, typename Functor>
  LinearForm<TestSpaceOp, Functor>
  linear_form(const TestSpaceOp &test_space_op, const Functor &functor_op)
  {
    return LinearForm<TestSpaceOp, Functor>(test_space_op, functor_op);
  }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  template <typename TestSpaceOp, typename Functor>
  struct is_linear_form<LinearForm<TestSpaceOp, Functor>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_linear_forms_h
