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

#ifndef dealii_weakforms_auto_differentiable_functors_h
#define dealii_weakforms_auto_differentiable_functors_h

#include <deal.II/base/config.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/weak_forms/functors.h>



DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace AutoDifferentiation
  {
    // // The meat in the middle of the WeakForms
    // class Functor
    // {};

    using ScalarFunctor = WeakForms::ScalarFunctor;

    template <int dim>
    using VectorFunctor = WeakForms::VectorFunctor<dim>;

    template <int rank, int dim>
    using TensorFunctor = WeakForms::TensorFunctor<rank, dim>;

    template <int rank, int dim>
    using SymmetricTensorFunctor = WeakForms::SymmetricTensorFunctor<rank, dim>;

    template <int dim>
    using ScalarFunctionFunctor = WeakForms::ScalarFunctionFunctor<dim>;

    template <int rank, int dim>
    using TensorFunctionFunctor = WeakForms::TensorFunctionFunctor<rank, dim>;

  } // namespace AutoDifferentiation

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    /* ------------------------ Functors: Custom ------------------------ */

    /**
     * Extract the value from a scalar functor.
     */
    template <typename ADNumberType>
    class UnaryOp<
      AutoDifferentiation::ScalarFunctor,
      UnaryOpCodes::value,
      typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
      ADNumberType>
    {
      using Op = AutoDifferentiation::ScalarFunctor;

    public:
      using ScalarType =
        typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type;
      static constexpr enum Differentiation::AD::NumberTypes ADNumberTypeCode =
        Differentiation::AD::ADNumberTraits<ADNumberType>::type_code;

      template <typename ResultNumberType = ScalarType>
      using value_type = typename Op::template value_type<ResultNumberType>;

      template <typename ResultNumberType = ScalarType>
      using return_type = std::vector<value_type<ResultNumberType>>;

      using ad_type =
        typename Differentiation::AD::NumberTraits<ScalarType,
                                                   ADNumberTypeCode>::ad_type;
      static_assert(std::is_same<ad_type, ADNumberType>::value,
                    "AD types not the same.");

      using ad_function_type =
        typename Op::template function_type<ad_type>; // TODO: Base off of AD
                                                      // number

      static const int rank = 0;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &operand, const ad_function_type &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand, [](const unsigned int) { return ad_type{}; })
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_default;
      }

      template <typename VectorType, int dim, int spacedim>
      void
      update_from_solution(const VectorType &                 solution,
                           const FEValuesBase<dim, spacedim> &fe_values)
      {
        using ADHelperType = Differentiation::AD::
          ScalarFunction<dim, ADNumberTypeCode, ScalarType>;
        static_assert(
          std::is_same<typename ADHelperType::ad_type, ADNumberType>::value,
          "AD types not the same.");

        std::cout << "HERE!" << std::endl;

        // ADHelperType ad_helper (n_independent_variables);

        // ad_helper.register_independent_variable(H, H_dofs);
        // ad_helper.register_independent_variable(C, C_dofs);
        // // NOTE: We have to extract the sensitivities in the order we wish to
        // // introduce them. So this means we have to do it by logical order
        // // of the extractors that we've created.
        // const SymmetricTensor<2,dim,ADNumberType> C_AD =
        //   ad_helper.get_sensitive_variables(C_dofs);
        // const Tensor<1,dim,ADNumberType>          H_AD =
        //   ad_helper.get_sensitive_variables(H_dofs);
        // // Here we define the material stored energy function.
        // // This example is sufficiently complex to warrant the use of AD to,
        // // at the very least, verify an unassisted implementation.
        // const double mu_e = 10;          // Shear modulus
        // const double lambda_e = 15;      // Lam&eacute; parameter
        // const double mu_0 = 4*M_PI*1e-7; // Magnetic permeability constant
        // const double mu_r = 5;           // Relative magnetic permeability
        // const ADNumberType J = std::sqrt(determinant(C_AD));
        // const SymmetricTensor<2,dim,ADNumberType> C_inv_AD = invert(C_AD);
        // const ADNumberType psi =
        //   0.5*mu_e*(1.0+std::tanh((H_AD*H_AD)/100.0))*
        //     (trace(C_AD) - dim - 2*std::log(J)) +
        //   lambda_e*std::log(J)*std::log(J) -
        //   0.5*mu_0*mu_r*J*H_AD*C_inv_AD*H_AD;
        // // Register the definition of the total stored energy
        // ad_helper.register_dependent_variable(psi_CH);

        // Vector<double> Dpsi (ad_helper.n_dependent_variables());
        // FullMatrix<double> D2psi (ad_helper.n_dependent_variables(),
        //                           ad_helper.n_independent_variables());
        // const double psi = ad_helper.compute_value();
        // ad_helper.compute_gradient(Dpsi);
        // ad_helper.compute_hessian(D2psi);
      }

      // Return single entry
      template <typename ResultNumberType = ScalarType>
      value_type<ResultNumberType>
      operator()(const unsigned int q_point) const
      {
        Assert(function, ExcNotInitialized());
        return function(q_point);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = ScalarType, int dim, int spacedim>
      return_type<ResultNumberType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const
      {
        return_type<ScalarType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.quadrature_point_indices())
          out.emplace_back(this->operator()<ResultNumberType>(q_point));

        return out;
      }

    private:
      const Op               operand;
      const ad_function_type function;
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  template <enum Differentiation::AD::NumberTypes ADNumberTypeCode,
            typename ScalarType = double>
  WeakForms::Operators::UnaryOp<WeakForms::AutoDifferentiation::ScalarFunctor,
                                WeakForms::Operators::UnaryOpCodes::value,
                                ScalarType>
  value(const WeakForms::AutoDifferentiation::ScalarFunctor &operand,
        const typename WeakForms::AutoDifferentiation::ScalarFunctor::
          template function_type<ScalarType> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = AutoDifferentiation::ScalarFunctor;
    using ADNumberType =
      typename Differentiation::AD::NumberTraits<ScalarType,
                                                 ADNumberTypeCode>::ad_type;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, ADNumberType>;

    return OpType(operand, function);
  }



  // template <typename NumberType = double, int rank, int dim>
  // WeakForms::Operators::UnaryOp<WeakForms::TensorFunctor<rank, dim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::TensorFunctor<rank, dim> &operand,
  //       const typename WeakForms::TensorFunctor<rank, dim>::
  //         template function_type<NumberType> &function)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = TensorFunctor<rank, dim>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

  //   return OpType(operand, function /*,NumberType()*/);
  // }



  // template <typename NumberType = double, int rank, int dim>
  // WeakForms::Operators::UnaryOp<WeakForms::SymmetricTensorFunctor<rank, dim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::SymmetricTensorFunctor<rank, dim> &operand,
  //       const typename WeakForms::SymmetricTensorFunctor<rank, dim>::
  //         template function_type<NumberType> &function)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SymmetricTensorFunctor<rank, dim>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

  //   return OpType(operand, function);
  // }



  // template <typename NumberType = double, int dim>
  // WeakForms::Operators::UnaryOp<WeakForms::ScalarFunctionFunctor<dim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::ScalarFunctionFunctor<dim> &operand,
  //       const typename WeakForms::ScalarFunctionFunctor<
  //         dim>::template function_type<NumberType> &function)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = ScalarFunctionFunctor<dim>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

  //   return OpType(operand, function);
  // }



  // template <typename NumberType = double, int rank, int dim>
  // WeakForms::Operators::UnaryOp<WeakForms::TensorFunctionFunctor<rank, dim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::TensorFunctionFunctor<rank, dim> &operand,
  //       const typename WeakForms::TensorFunctionFunctor<rank, dim>::
  //         template function_type<NumberType> &function)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = TensorFunctionFunctor<rank, dim>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

  //   return OpType(operand, function);
  // }
} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Decorator classes

  // template <int dim, int spacedim>
  // struct is_ad_functor<FieldSolution<dim, spacedim>> : std::true_type
  // {};

  // Unary operations

  template <typename ADNumberType>
  struct is_ad_functor<Operators::UnaryOp<
    AutoDifferentiation::ScalarFunctor,
    Operators::UnaryOpCodes::value,
    typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
    ADNumberType>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_auto_differentiable_functors_h
