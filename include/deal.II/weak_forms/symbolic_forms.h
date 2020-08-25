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

#ifndef dealii_weakforms_symbolic_forms_h
#define dealii_weakforms_symbolic_forms_h

#include <deal.II/base/config.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Symbolic
  {
    /**
     * OP: (SymbolicFunctor)
     *
     * First derivatives of this form produce a ResidualForm.
     */
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
     *This class gets converted into a LinearForm.
     * First derivatives of this form produce a BilinearForm through the
     * LinearizationForm
     */
    class LinearizationForm
    {
      private:
      friend EnergyFunctional;
      friend ResidualForm;
      LinearizationForm() = default;
    };
  } // namespace Symbolic

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_symbolic_forms_h
