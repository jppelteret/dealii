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
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/weakforms/operators.h>
#include <deal.II/weakforms/symbolic_decorations.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{

  // The meat in the middle of the WeakForms
  template <int rank_>
  class Functor
  {
    // using OpType =
    //   Operators::UnaryOp<Functor, Operators::UnaryOpCodes::value>;

  public:

    /**
     * Rank of this object operates.
     */
    static const unsigned int rank = rank_;

    Functor(const std::string &       symbol_ascii,
            const std::string &       symbol_latex,
            const SymbolicDecorations &decorator = SymbolicDecorations())
      : symbol_ascii(symbol_ascii)
      , symbol_latex(symbol_latex != "" ? symbol_latex : symbol_ascii)
      , decorator(decorator)
    {}

    const SymbolicDecorations &
    get_decorator() const
    {
      return decorator;
    }

    // ----  Ascii ----

    std::string
    as_ascii() const
    {
      return get_decorator().unary_op_functor_as_ascii(*this, rank);
    }

    std::string
    get_symbol_ascii() const
    {
      return symbol_ascii;
    }

    const SymbolicNamesAscii &
    get_naming_ascii() const
    {
      return decorator.naming_ascii;
    }

    // ---- LaTeX ----

    std::string
    as_latex() const
    {
      return get_decorator().unary_op_functor_as_latex(*this, rank);
    }

    std::string
    get_symbol_latex() const
    {
      return symbol_latex;
    }

    const SymbolicNamesLaTeX &
    get_naming_latex() const
    {
      return decorator.naming_latex;
    }

  protected:
    const std::string symbol_ascii;
    const std::string symbol_latex;

    const SymbolicDecorations decorator;
  };



  class ScalarFunctor : public Functor<0>
  {
    // using OpType =
    //   Operators::UnaryOp<ScalarFunctor<NumberType>, Operators::UnaryOpCodes::value>;
    using Base = Functor<0>;

  public:

    template <typename NumberType>
    using value_type = NumberType;

    template <typename NumberType>
    using function_type = std::function<value_type<NumberType>(const unsigned int q_point)>;

    ScalarFunctor(const std::string &       symbol_ascii,
                  const std::string &       symbol_latex,
                  const SymbolicDecorations &decorator = SymbolicDecorations())
      : Base(symbol_ascii, symbol_latex, decorator)
    {}

  };



  template<int rank, int dim>
  class TensorFunctor : public Functor<rank>
  {
    // using OpType =
    //   Operators::UnaryOp<TensorFunctor<rank_,dim,NumberType>, Operators::UnaryOpCodes::value>;

    using Base = Functor<rank>;

    public:

    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = dim;
    
    template <typename NumberType>
    using value_type = Tensor<rank,dim,NumberType>;

    template <typename NumberType>
    using function_type = std::function<value_type<NumberType>(const unsigned int q_point)>;

    TensorFunctor(const std::string &       symbol_ascii,
                  const std::string &       symbol_latex,
                  const SymbolicDecorations &decorator = SymbolicDecorations())
      : Base(symbol_ascii, symbol_latex, decorator)
    {}
  };



  template <int dim>
  using VectorFunctor = TensorFunctor<1,dim>;



  template<int rank, int dim>
  class SymmetricTensorFunctor : public Functor<rank>
  {
    // using OpType =
    //   Operators::UnaryOp<SymmetricTensorFunctor<rank_,dim,NumberType>, Operators::UnaryOpCodes::value>;

    using Base = Functor<rank>;

    public:

    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = dim;
    
    template <typename NumberType>
    using value_type = SymmetricTensor<rank,dim,NumberType>;

    template <typename NumberType>
    using function_type = std::function<value_type<NumberType>(const unsigned int q_point)>;

    SymmetricTensorFunctor(const std::string &       symbol_ascii,
                  const std::string &       symbol_latex,
                  const SymbolicDecorations &decorator = SymbolicDecorations())
      : Base(symbol_ascii, symbol_latex, decorator)
    {}
  };



  // Wrap up a scalar dealii::FunctionBase as a functor
  template <int dim>
  class ScalarFunctionFunctor : public Functor<0>
  {
    // using OpType =
    //   Operators::UnaryOp<ScalarFunctionFunctor<dim,NumberType>, Operators::UnaryOpCodes::value>;

    using Base = Functor<0>;

  public:

    template <typename NumberType>
    using function_type = Function<dim, NumberType>;

    // template <typename NumberType>
    // using value_type = typename function_type<NumberType>::value_type;
    
    // template <typename NumberType>
    // using gradient_type = typename function_type<NumberType>::gradient_type;

    template <typename NumberType>
    using value_type = NumberType;
    
    template <typename NumberType>
    using gradient_type = Tensor<1,dim,NumberType>;

    ScalarFunctionFunctor(const std::string &       symbol_ascii,
                  const std::string &       symbol_latex,
                  const SymbolicDecorations &decorator = SymbolicDecorations())
      : Base(symbol_ascii, symbol_latex, decorator)
    {}
  };


  // Wrap up a tensor dealii::TensorFunction as a functor
  template<int rank, int dim>
  class TensorFunctionFunctor : public Functor<rank>
  {
    // using OpType =
    //   Operators::UnaryOp<TensorFunctionFunctor<rank_,dim,NumberType>, Operators::UnaryOpCodes::value>;
    using Base = Functor<rank>;

    public:

    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = dim;

    template <typename NumberType>
    using function_type = TensorFunction<rank,dim,NumberType>;
    
    template <typename NumberType>
    using value_type = typename function_type<NumberType>::value_type;
    
    template <typename NumberType2>
    using gradient_type = typename function_type<NumberType2>::gradient_type;

    TensorFunctionFunctor(const std::string &       symbol_ascii,
                  const std::string &       symbol_latex,
                  const SymbolicDecorations &decorator = SymbolicDecorations())
      : Base(symbol_ascii, symbol_latex, decorator)
    {}
  };

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_functors_h
