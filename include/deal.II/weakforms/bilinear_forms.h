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

#include <deal.II/weakforms/spaces.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  template <typename TestSpaceOp, typename Functor, typename TrialSpaceOp>
  class BilinearForm
  {
  public:
    explicit BilinearForm(const TestSpaceOp & test_space_op,
                          const Functor &     functor_op,
                          const TrialSpaceOp &trial_space_op)
      : test_space_op(test_space_op)
      , functor_op(functor_op)
      , trial_space_op(trial_space_op)
    {}

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
      return "\\left\\[" + test_space_op.as_latex() + "*" +
             functor_op.as_latex() + "*" + trial_space_op.as_latex() +
             "\\right\\]";
    }

  private:
    const TestSpaceOp  test_space_op;
    const Functor      functor_op;
    const TrialSpaceOp trial_space_op;
  };


  /* ========================= CONVENIENCE FUNCTIONS =========================*/


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


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_bilinear_forms_h