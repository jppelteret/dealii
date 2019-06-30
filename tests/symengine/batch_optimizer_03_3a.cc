// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2018 by the deal.II authors
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


// Check that one can perform substitution of symbolic derivatives that
// are the result of explicit or implicit relationships between symbolic
// variables.
// See tests/symengine/basic_06.cc and tests/symengine/basic_07.cc for
// a more simple example of differentiation of symbols with
// explicit/implicit relationships.
// We invoke the batch optimizer before symbolic evaluation takes place.
//
// Here we invoke the LLVM optimizer before symbolic evaluation takes place,
// and we invoke no additional symbolic optimizations as well.

#include "../tests.h"
#include "sd_common_tests/batch_optimizer_03.h"

int
main()
{
  initlog();

  const enum SD::OptimizerType     opt_method = SD::OptimizerType::llvm;
  const enum SD::OptimizationFlags opt_flags =
    SD::OptimizationFlags::optimize_default;

  const int dim = 2;
  run_tests<dim, opt_method, opt_flags>();
}
