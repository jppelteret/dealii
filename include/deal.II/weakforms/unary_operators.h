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

#ifndef dealii_weakforms_unary_operators_h
#define dealii_weakforms_unary_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <deal.II/weakforms/operators.h>
#include <deal.II/weakforms/spaces.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    template <int dim, int spacedim>
    class UnaryOp<Space<dim, spacedim>>
    {
      using Op = Space<dim, spacedim>;

    public:
      UnaryOp(const Op &operand, const enum UnaryOpCodes op_code)
        : operand(operand)
        , op_code(op_code)
      {}

      std::string
      as_ascii() const
      {
        switch (op_code)
          {
            case (UnaryOpCodes::value):
              return operand.get_symbol_ascii() + "{" +
                     operand.get_field_ascii() + "}";
              break;
            default:
              AssertThrow(false, ExcUnaryOperatorNotDefined(op_code));
          }
        return "";
      }

      std::string
      as_latex() const
      {
        switch (op_code)
          {
            case (UnaryOpCodes::value):
              return operand.get_symbol_latex() + "_{" +
                     operand.get_field_latex() + "}";
              break;
            default:
              AssertThrow(false, ExcUnaryOperatorNotDefined(op_code));
          }
        return "";
      }

    private:
      const Op &              operand;
      const enum UnaryOpCodes op_code;
    };

  } // namespace Operators
} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_unary_operators_h
