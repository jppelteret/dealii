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


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{

  class Space {};

  namespace SpaceViews
  {
    class Scalar {};
    class Vector {};
    class Tensor {};
    class SymmetricTensor {};
  }

  class TestFunction : public Space {};

  class TrialSolution : public Space {};

  namespace Linear
  {
    using TestFunction = WeakForms::TestFunction;
    using Solution = WeakForms::TrialSolution;
  }

  namespace NonLinear
  {
    using Variation = WeakForms::TestFunction;
    using Solution = WeakForms::TrialSolution;
    using Linearization = WeakForms::TrialSolution;
  }

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_spaces_h
