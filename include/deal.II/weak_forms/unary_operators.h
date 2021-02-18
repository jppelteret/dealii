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

// #include <deal.II/base/exceptions.h>
// #include <deal.II/base/utilities.h>

// #include <deal.II/weak_forms/symbolic_decorations.h>
// #include <deal.II/weak_forms/type_traits.h>



// DEAL_II_NAMESPACE_OPEN


// namespace WeakForms
// {
//   namespace Operators
//   {
//     enum class UnaryOpCodes
//     {
//       /*
//        * Negate the current operand.
//        */
//       // negate,
//       /**
//        * Jump of an operand across an interface
//        */
//       // jump,
//       /**
//        * Average of an operand across an interface
//        */
//       // average,
//       // transpose,
//       // contract,
//       // double contract,
//       // scalar product,
//       // determinant,
//       // invert,
//       // symmetrize
//     };



//     /**
//      * Exception denoting that a class requires some specialization
//      * in order to be used.
//      */
//     DeclExceptionMsg(
//       ExcRequiresUnaryOperatorSpecialization,
//       "This function is called in a class that is expected to be specialized
//       " "for unary operations. All unary operators should be specialized,
//       with " "a structure matching that of the exemplar class.");


//     /**
//      * Exception denoting that a unary operation has not been defined.
//      */
//     DeclException1(ExcUnaryOperatorNotDefined,
//                    enum UnaryOpCodes,
//                    << "The unary operator with code " +
//                           dealii::Utilities::to_string(static_cast<int>(arg1))
//                           + " has not been defined.");



//     /**
//      * @tparam Op
//      * @tparam OpCode
//      * @tparam UnderlyingType Underlying number type (double, std::complex<double>, etc.).
//      * This is necessary because some specializations of the class do not use
//      * the number type in the specialization itself, but they may rely on the
//      * type in their definitions (e.g. class members).
//      * @tparam Args A dumping ground for any other arguments that may be necessary
//      * to form a concrete class instance.
//      */
//     template <typename Op,
//               enum UnaryOpCodes OpCode,
//               typename UnderlyingType = void,
//               typename... Args>
//     class UnaryOp
//     {
//     public:
//       explicit UnaryOp(const Op &operand)
//         : operand(operand)
//       {
//         AssertThrow(false, ExcRequiresUnaryOperatorSpecialization());
//       }

//       std::string
//       as_ascii(const SymbolicDecorations &decorator) const
//       {
//         (void)decorator;
//         AssertThrow(false, ExcRequiresUnaryOperatorSpecialization());
//         return "";
//       }

//       std::string
//       as_latex(const SymbolicDecorations &decorator) const
//       {
//         (void)decorator;
//         AssertThrow(false, ExcRequiresUnaryOperatorSpecialization());
//         return "";
//       }

//     private:
//       const Op operand;
//     }; // class UnaryOp


//   } // namespace Operators


// } // namespace WeakForms



// /* ================== Specialization of unary operators ================== */



// #ifndef DOXYGEN



// namespace WeakForms
// {
//   namespace Operators
//   {} // namespace Operators
// } // namespace WeakForms


// #endif // DOXYGEN



// /* ==================== Specialization of type traits ==================== */



// #ifndef DOXYGEN


// namespace WeakForms
// {
//   // template <typename... Args>
//   // struct is_unary_op<Operators::UnaryOp<Args...>> : std::true_type
//   // {};

// } // namespace WeakForms


// #endif // DOXYGEN



// DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_unary_operators_h
