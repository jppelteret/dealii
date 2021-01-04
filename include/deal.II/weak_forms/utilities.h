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

#ifndef dealii_weakforms_utilities_h
#define dealii_weakforms_utilities_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <iterator>
#include <numeric>
#include <string>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Utilities
  {
    // T must be an iterable type
    template <typename IterableObject>
    std::string
    get_comma_separated_string_from(const IterableObject &t)
    {
      // Expand the object of subdomains as a comma separated list
      // https://stackoverflow.com/a/34076796
      return std::accumulate(std::begin(t),
                             std::end(t),
                             std::string{},
                             [](const std::string &a, const auto &b) {
                               return a.empty() ?
                                        dealii::Utilities::to_string(b) :
                                        a + ',' +
                                          dealii::Utilities::to_string(b);
                             });
    }



    /**
     * A small data structure to work out some information that has to do with
     * the contraction of operands during "multiplication" operations
     *
     * @tparam LhsOp
     * @tparam RhsOp
     */
    template <typename LhsOp, typename RhsOp>
    struct IndexContraction
    {
      static const int n_contracting_indices =
        (LhsOp::rank < RhsOp::rank ? LhsOp::rank : RhsOp::rank);
      static const int result_rank =
        (LhsOp::rank < RhsOp::rank ? RhsOp::rank - LhsOp::rank :
                                     LhsOp::rank - RhsOp::rank);

      template <int A>
      struct NonNegative
      {
        static_assert(A >= 0, "Non-negative");
        static const int value = A;
      };

      static_assert(NonNegative<n_contracting_indices>::value >= 0,
                    "Number of contracting indices cannot be negative.");
      static_assert(NonNegative<result_rank>::value >= 0,
                    "Cannot have a result with a negative rank.");
    };



    struct LaTeX
    {
      static constexpr char l_parenthesis[] = "\\left\\(";
      static constexpr char r_parenthesis[] = "\\right\\)";

      static constexpr char l_square_brace[] = "\\left\\[";
      static constexpr char r_square_brace[] = "\\right\\]";

      std::string static get_symbol_multiply(
        const unsigned int n_contracting_indices)
      {
        switch (n_contracting_indices)
          {
            case (0):
              return "\\,";
              break;
            case (1):
              return " \\cdot ";
              break;
            case (2):
              return " \\colon ";
              break;
            case (3):
              return " \\vdots ";
              break;
            case (4):
              return " \\colon\\colon ";
              break;
            case (5):
              return " \\vdots\\colon ";
              break;
            case (6):
              return " \\vdots\\vdots ";
              break;
            default:
              break;
          }

        AssertThrow(false, ExcNotImplemented());
        return " * ";
      }
    };

  } // namespace Utilities

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE


#endif // dealii_weakforms_utilities_h
