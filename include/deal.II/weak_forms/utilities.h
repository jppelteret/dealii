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

#include <deal.II/physics/notation.h>

#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>


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
    // T must be an iterable type
    template <typename Type>
    std::string
    get_separated_string_from(const Type *const  begin,
                              const Type *const  end,
                              const std::string &seperator)
    {
      // Expand the object of subdomains as a comma separated list
      // https://stackoverflow.com/a/34076796
      return std::accumulate(begin,
                             end,
                             std::string{},
                             [&seperator](const std::string &a, const auto &b) {
                               return a.empty() ?
                                        dealii::Utilities::to_string(b) :
                                        a + seperator +
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



    template <typename T, typename U = void>
    struct ConvertNumericToText;


    template <typename ScalarType>
    struct ConvertNumericToText<
      ScalarType,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const ScalarType value)
      {
        return dealii::Utilities::to_string(value);
      }

      static std::string
      to_latex(const ScalarType value)
      {
        return dealii::Utilities::to_string(value);
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      Tensor<0, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const Tensor<0, dim, ScalarType> &value)
      {
        // There's only one element; let's treat it like a scalar.
        return ConvertNumericToText<ScalarType>::to_ascii(*value.begin_raw());
      }

      static std::string
      to_latex(const Tensor<0, dim, ScalarType> &value)
      {
        // There's only one element; let's treat it like a scalar.
        return ConvertNumericToText<ScalarType>::to_latex(*value.begin_raw());
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      Tensor<1, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const Tensor<1, dim, ScalarType> &value)
      {
        std::string out = "[";
        out +=
          get_separated_string_from(value.begin_raw(), value.end_raw(), ",");
        out += "]";
        return out;
      }

      static std::string
      to_latex(const Tensor<1, dim, ScalarType> &value)
      {
        std::string out = "\\begin{pmatrix}";
        out +=
          get_separated_string_from(value.begin_raw(), value.end_raw(), "\\\\");
        out += "\\end{pmatrix}";
        return out;
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      Tensor<2, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const Tensor<2, dim, ScalarType> &value)
      {
        std::string out = "[";
        for (unsigned int i = 0; i < dim; ++i)
          {
            out += get_separated_string_from(value[i].begin_raw(),
                                             value[i].end_raw(),
                                             ",");
            if (i < dim - 1)
              out += ";";
          }
        out += "]";
        return out;
      }

      static std::string
      to_latex(const Tensor<2, dim, ScalarType> &value)
      {
        std::string out = "\\begin{bmatrix}";
        for (unsigned int i = 0; i < dim; ++i)
          {
            out += get_separated_string_from(value[i].begin_raw(),
                                             value[i].end_raw(),
                                             "&");
            if (i < dim - 1)
              out += "\\\\";
          }
        out += "\\end{bmatrix}";
        return out;
      }
    };


    template <typename ScalarType>
    struct ConvertNumericToText<
      FullMatrix<ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const FullMatrix<ScalarType> &value)
      {
        const std::size_t n_rows = value.m();
        const std::size_t n_cols = value.n();

        std::string out = "[";
        for (unsigned int i = 0; i < n_rows; ++i)
          {
            for (unsigned int j = 0; j < n_cols; ++j)
              {
                out += dealii::Utilities::to_string(value[i][j]);
                if (j < n_cols - 1)
                  out += ",";
              }
            if (i < n_rows - 1)
              out += ";";
          }
        out += "]";
        return out;
      }

      static std::string
      to_latex(const FullMatrix<ScalarType> &value)
      {
        const std::size_t n_rows = value.m();
        const std::size_t n_cols = value.n();

        std::string out = "\\begin{bmatrix}";
        for (unsigned int i = 0; i < n_rows; ++i)
          {
            for (unsigned int j = 0; j < n_cols; ++j)
              {
                out += dealii::Utilities::to_string(value[i][j]);
                if (j < n_cols - 1)
                  out += "&";
              }
            if (i < n_rows - 1)
              out += "\\\\";
          }
        out += "\\end{bmatrix}";
        return out;
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      Tensor<3, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const Tensor<3, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_ascii(
          Physics::Notation::Kelvin::to_matrix(value));
      }

      static std::string
      to_latex(const Tensor<3, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_latex(
          Physics::Notation::Kelvin::to_matrix(value));
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      Tensor<4, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const Tensor<4, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_ascii(
          Physics::Notation::Kelvin::to_matrix(value));
      }

      static std::string
      to_latex(const Tensor<4, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_latex(
          Physics::Notation::Kelvin::to_matrix(value));
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      SymmetricTensor<2, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      // TODO: Is it worth copying this to a full matrix, instead of reproducing
      // the implementation here?

      static std::string
      to_ascii(const SymmetricTensor<2, dim, ScalarType> &value)
      {
        std::string out = "[";
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              {
                out += dealii::Utilities::to_string(value[i][j]);
                if (j < dim - 1)
                  out += ",";
              }
            if (i < dim - 1)
              out += ";";
          }
        out += "]";
        return out;
      }

      static std::string
      to_latex(const SymmetricTensor<2, dim, ScalarType> &value)
      {
        std::string out = "\\begin{bmatrix}";
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              {
                out += dealii::Utilities::to_string(value[i][j]);
                if (j < dim - 1)
                  out += "&";
              }
            if (i < dim - 1)
              out += "\\\\";
          }
        out += "\\end{bmatrix}";
        return out;
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      SymmetricTensor<4, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const SymmetricTensor<4, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_ascii(
          Physics::Notation::Kelvin::to_matrix(value));
      }

      static std::string
      to_latex(const SymmetricTensor<4, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_latex(
          Physics::Notation::Kelvin::to_matrix(value));
      }
    };



    struct LaTeX
    {
      static constexpr char l_parenthesis[] = "\\left(";
      static constexpr char r_parenthesis[] = "\\right)";

      static constexpr char l_square_brace[] = "\\left[";
      static constexpr char r_square_brace[] = "\\right]";

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
