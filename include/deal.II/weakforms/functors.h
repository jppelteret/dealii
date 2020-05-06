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

#ifndef dealii_weakforms_functors_h
#define dealii_weakforms_functors_h

#include <deal.II/base/config.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/weakforms/operators.h>
#include <deal.II/weakforms/symbolic_info.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  // Forward declarations
  template <typename NumberType>
  class Functor;
  template <typename NumberType>
  class ScalarFunctor;
  namespace Operators
  {
    template <typename NumberType>
    class UnaryOp<Functor<NumberType>, UnaryOpCodes::value>;

    template <typename NumberType>
    class UnaryOp<ScalarFunctor<NumberType>, UnaryOpCodes::value>;
  } // namespace Operators

  // The meat in the middle of the WeakForms
  template <typename NumberType>
  class Functor
  {
    using OpType =
      Operators::UnaryOp<Functor, Operators::UnaryOpCodes::value>;

  public:

    template <typename NumberType2>
    using value_type = NumberType;

    Functor(const std::string &       symbol_ascii,
            const std::string &       symbol_latex,
            const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
            const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
      : symbol_ascii(symbol_ascii)
      , symbol_latex(symbol_latex != "" ? symbol_latex : symbol_ascii)
      , naming_ascii(naming_ascii)
      , naming_latex(naming_latex)
    {}

    // ----  Ascii ----

    std::string
    as_ascii() const
    {
      return OpType(*this).as_ascii();
    }

    std::string
    get_symbol_ascii() const
    {
      return symbol_ascii;
    }

    const SymbolicNamesAscii &
    get_naming_ascii() const
    {
      return naming_ascii;
    }

    // ---- LaTeX ----

    std::string
    as_latex() const
    {
      return OpType(*this).as_latex();
    }

    std::string
    get_symbol_latex() const
    {
      return symbol_latex;
    }

    const SymbolicNamesLaTeX &
    get_naming_latex() const
    {
      return naming_latex;
    }

  protected:
    const std::string symbol_ascii;
    const std::string symbol_latex;

    const SymbolicNamesAscii naming_ascii;
    const SymbolicNamesLaTeX naming_latex;
  };


  template <typename NumberType>
  class ScalarFunctor : public Functor<NumberType>
  {
  public:

    template <typename NumberType2>
    using value_type = typename Functor<NumberType>::template value_type<NumberType2>;

    template <typename NumberType2>
    using function_type = std::function<value_type<NumberType2>(const unsigned int q_point)>;

    ScalarFunctor(const std::string &       symbol_ascii,
                  const std::string &       symbol_latex,
                  const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
                  const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
      : Functor<NumberType>(symbol_ascii, symbol_latex, naming_ascii, naming_latex)
    {}
  };

  template<int rank_, int dim, typename NumberType>
  class TensorFunctor : public Functor<NumberType>
  {
    public:
    /**
     * Rank of this object operates.
     */
    static const unsigned int rank = rank_;

    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = dim;
    
    template <typename NumberType2>
    using value_type = Tensor<rank,dim,typename Functor<NumberType>::template value_type<NumberType2>>;

    template <typename NumberType2>
    using function_type = std::function<value_type<NumberType2>(const unsigned int q_point)>;

    TensorFunctor(const std::string &       symbol_ascii,
                  const std::string &       symbol_latex,
                  const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
                  const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
      : Functor<NumberType>(symbol_ascii, symbol_latex, naming_ascii, naming_latex)
    {}
  };

  // class SymmetricTensorFunctor : public Functor
  // {
  //   template <int rank, int dim, typename NumberType2>
  //   using value_type = SymmetricTensor<rank,dim,typename Functor<NumberType>::template value_type<NumberType2>>;

  //   SymmetricTensorFunctor(const std::string &       symbol_ascii,
  //                          const std::string &       symbol_latex,
  //                          const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
  //                          const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX())
  //     : Functor<NumberType>(symbol_ascii, symbol_latex, naming_ascii, naming_latex)
  //   {}
  // };

  // // Wrap up a dealii::FunctionBase as a functor
  // class FunctionFunctor : public Functor
  // {};

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_functors_h
