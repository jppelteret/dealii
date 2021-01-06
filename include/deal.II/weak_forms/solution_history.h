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

#ifndef dealii_weakforms_solution_history_h
#define dealii_weakforms_solution_history_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <string>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  template <typename VectorType>
  class SolutionHistory
  {
  public:
    using ptr_type = const VectorType *const;

    SolutionHistory(const VectorType &solution_history,
                    const std::string name = "solution")
      : name(name)
      , solution_history(&solution_history)
    {}

    SolutionHistory(const std::vector<ptr_type> &solution_history,
                    const std::string            name = "solution")
      : name(name)
      , solution_history(solution_history)
    {}

    const VectorType &
    get_solution_name(const std::size_t index = 0) const
    {
      Assert(index < solution_history.size(),
             ExcAssertIndexRange(index, 0, solution_history.size()));

      if (index == 0)
        return name;
      else
        return name + "_t" + dealii::Utilities::to_string(index);
    }

    const VectorType &
    get_solution_vector(const std::size_t index = 0) const
    {
      Assert(index < solution_history.size(),
             ExcAssertIndexRange(index, 0, solution_history.size()));
      return *(solution_history[index]);
    }

  private:
    const std::string name;

    // Expected order:
    // - 0: Current solution
    // - 1: Previous solution
    // - 2: Previous-previous solution
    // - ...
    const std::vector<ptr_type> solution_history;
  }; // class SolutionHistory
} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_solution_history_h
