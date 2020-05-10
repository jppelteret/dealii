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

// Grouped by function


// Utilities
#include <deal.II/weakforms/symbolic_decorations.h>
#include <deal.II/weakforms/operators.h>
#include <deal.II/weakforms/type_traits.h>

// Functors and spaces to be used inside of weak forms
#include <deal.II/weakforms/functors.h>
#include <deal.II/weakforms/spaces.h>
#include <deal.II/weakforms/symbolic_functors.h>

// Operators that operate and give values to functors and spaces
#include <deal.II/weakforms/binary_operators.h>
#include <deal.II/weakforms/unary_operators.h>

#include <deal.II/weakforms/fe_space_operators.h> // ?

// The actual forms themselves
#include <deal.II/weakforms/bilinear_forms.h>
#include <deal.II/weakforms/linear_forms.h>
#include <deal.II/weakforms/symbolic_forms.h>

// Common tools for assembly
#include <deal.II/weakforms/integrator.h>

// Matrix-based assembly


#endif // dealii_weakforms_weakforms_h
