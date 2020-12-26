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

#ifndef dealii_weakforms_weakforms_h
#define dealii_weakforms_weakforms_h

#include <deal.II/base/config.h>

// Grouped by function:


// Utilities
// #include <deal.II/weak_forms/operators.h> // ?
#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/type_traits.h>

// Functors and spaces to be used inside of weak forms
// #include <deal.II/weak_forms/auto_differentiable_functors.h>
#include <deal.II/weak_forms/functors.h>
#include <deal.II/weak_forms/spaces.h>
// #include <deal.II/weak_forms/symbolic_functors.h>

// Subspaces
#include <deal.II/weak_forms/subspace_extractors.h>
#include <deal.II/weak_forms/subspace_views.h>

// Operators that operate and give values to functors and spaces
#include <deal.II/weak_forms/binary_operators.h>
#include <deal.II/weak_forms/cell_face_subface_operators.h>
#include <deal.II/weak_forms/fe_space_operators.h> // ?
#include <deal.II/weak_forms/unary_operators.h>

// The actual forms themselves
// #include <deal.II/weak_forms/auto_differentiable_forms.h>
#include <deal.II/weak_forms/bilinear_forms.h>
#include <deal.II/weak_forms/linear_forms.h>
#include <deal.II/weak_forms/self_linearizing_forms.h>
// #include <deal.II/weak_forms/symbolic_forms.h>

// Common tools for assembly
#include <deal.II/weak_forms/integral.h>
#include <deal.II/weak_forms/integrator.h>

// Assembly
#include <deal.II/weak_forms/assembler.h>


#endif // dealii_weakforms_weakforms_h
