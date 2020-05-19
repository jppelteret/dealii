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

#ifndef dealii_weakforms_bilinear_forms_h
#define dealii_weakforms_bilinear_forms_h

#include <deal.II/base/config.h>

#include <deal.II/base/types.h>

#include <deal.II/weak_forms/integral.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/type_traits.h>
#include <deal.II/weak_forms/utilities.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  template <typename TestSpaceOp_, typename Functor_, typename TrialSpaceOp_>
  class BilinearForm
  {
  public:
    using TestSpaceOp  = TestSpaceOp_;
    using Functor      = Functor_;
    using TrialSpaceOp = TrialSpaceOp_;

    explicit BilinearForm(const TestSpaceOp & test_space_op,
                          const Functor &     functor_op,
                          const TrialSpaceOp &trial_space_op)
      : test_space_op(test_space_op)
      , functor_op(functor_op)
      , trial_space_op(trial_space_op)
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
             ", " + trial_space_op.as_ascii() + ")";
    }

    std::string
    as_latex() const
    {
      // const std::string lbrace = "\\left\\[";
      // const std::string rbrace = "\\right\\]";
      // const auto &decorator = get_decorator();

      // constexpr unsigned int n_contracting_indices_test_functor =
      // WeakForms::Utilities::IndexContraction<TestSpaceOp,Functor>::n_contracting_indices;
      // const std::string symb_mult_test_functor =
      // decorator.get_symbol_multiply_latex(n_contracting_indices_test_functor)

      // constexpr unsigned int n_contracting_indices_functor_trial =
      // WeakForms::Utilities::IndexContraction<Functor,TrialSpaceOp>::n_contracting_indices;
      // const std::string symb_mult_functor_trial =
      // decorator.get_symbol_multiply_latex(n_contracting_indices_test_functor);;

      // return "\\left\\[" + test_space_op.as_latex() + symb_mult_test_functor
      // +
      //        functor_op.as_latex() + symb_mult_functor_trial +
      //        trial_space_op.as_latex() +
      //        "\\right\\]";

      return "\\left\\[" + test_space_op.as_latex() + " * " +
             functor_op.as_latex() + " * " + trial_space_op.as_latex() +
             "\\right\\]";
    }

    // ===== Section: Integration =====

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

    // ===== Section: Construct assembly operation =====

    const TestSpaceOp &
    get_test_space_operation() const
    {
      return test_space_op;
    }

    const Functor &
    get_functor() const
    {
      return functor_op;
    }

    const TrialSpaceOp &
    get_trial_space_operation() const
    {
      return trial_space_op;
    }

  private:
    const TestSpaceOp  test_space_op;
    const Functor      functor_op;
    const TrialSpaceOp trial_space_op;
  };

} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  template <typename TestSpaceOp, typename Functor, typename TrialSpaceOp>
  BilinearForm<TestSpaceOp, Functor, TrialSpaceOp>
  bilinear_form(const TestSpaceOp & test_space_op,
                const Functor &     functor_op,
                const TrialSpaceOp &trial_space_op)
  {
    return BilinearForm<TestSpaceOp, Functor, TrialSpaceOp>(test_space_op,
                                                            functor_op,
                                                            trial_space_op);
  }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  template <typename TestSpaceOp, typename Functor, typename TrialSpaceOp>
  struct is_bilinear_form<BilinearForm<TestSpaceOp, Functor, TrialSpaceOp>>
    : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_bilinear_forms_h
