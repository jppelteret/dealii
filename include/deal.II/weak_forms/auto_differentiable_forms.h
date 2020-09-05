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

#ifndef dealii_weakforms_auto_differentiable_forms_h
#define dealii_weakforms_auto_differentiable_forms_h

#include <deal.II/base/config.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace AutoDifferentiation
  {
    /**
     * OP: (AutoDifferentiableFunctor)
     *
     * First derivatives of this form produce a ResidualForm.
     */
    template <typename ADFunctor, typename... FieldArgs>
    class EnergyFunctional
    {};

    /**
     * OP: (Variation, SymbolicFunctor)
     *
     * This class gets converted into a LinearForm.
     * First derivatives of this form produce a BilinearForm through the
     * LinearizationForm
     */
    class ResidualForm
    {};

    /**
     * OP: (Variation, SymbolicFunctor, Linearization)
     *
     * This class gets converted into a LinearForm.
     * First derivatives of this form produce a BilinearForm through the
     * LinearizationForm
     */
    class LinearizationForm
    {
    private:
      // friend EnergyFunctional;
      // friend ResidualForm;
      LinearizationForm() = default;
    };
  } // namespace AutoDifferentiation

} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  // template <typename TestSpaceOp, typename TrialSpaceOp>
  // BilinearForm<TestSpaceOp, NoOp, TrialSpaceOp>
  // bilinear_form(const TestSpaceOp & test_space_op,
  //               const TrialSpaceOp &trial_space_op)
  // {
  //   return BilinearForm<TestSpaceOp, NoOp, TrialSpaceOp>(test_space_op,
  //                                                        trial_space_op);
  // }

  template <typename ADFunctor, typename... FieldArgs>
  AutoDifferentiation::EnergyFunctional<ADFunctor, FieldArgs...>
  ad_energy_functional_form(const ADFunctor &functor_op,
                            const FieldArgs &... dependent_fields)
  {
    return AutoDifferentiation::EnergyFunctional<ADFunctor, FieldArgs...>(
      functor_op, dependent_fields...);
  }

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_auto_differentiable_forms_h
