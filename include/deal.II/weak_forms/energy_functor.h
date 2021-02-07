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

#ifndef dealii_weakforms_energy_functor_h
#define dealii_weakforms_energy_functor_h

#include <deal.II/base/config.h>

#include <deal.II/algorithms/general_data_storage.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/tensor.h>

#include <deal.II/differentiation/ad.h>
#include <deal.II/differentiation/sd.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/weak_forms/functors.h>
#include <deal.II/weak_forms/solution_storage.h>
#include <deal.II/weak_forms/type_traits.h>

#include <tuple>
#include <utility>


DEAL_II_NAMESPACE_OPEN

// TODO: Move this to some central location
namespace WeakForms
{
  namespace Operators
  {
    namespace internal
    {
      // template <typename T>
      // struct UnaryOpUnpacker;

      // template <typename... UnaryOps>
      // struct UnaryOpUnpacker
      // {
      //   // TEMP
      //   static constexpr int n_components = 0 ;
      //   ///UnaryOpUnpacker<UnaryOps...>
      // };

      // template<typename T>
      // T adder(T first) {
      //   return first;
      // }

      // template<typename T, typename... Args>
      // T adder(T first, Args... args) {
      //   return first + adder(args...);
      // }


      // TODO: This is replicated in self_linearizing_forms.h
      template <typename T>
      class is_scalar_type
      {
        // See has_begin_and_end() in template_constraints.h
        // and https://stackoverflow.com/a/10722840

        template <typename A>
        static constexpr auto
        test(int) -> decltype(std::declval<typename EnableIfScalar<A>::type>(),
                              std::true_type())
        {
          return true;
        }

        template <typename A>
        static std::false_type
        test(...);

      public:
        using type = decltype(test<T>(0));

        static const bool value = type::value;
      };


      // template <typename T, typename U = void>
      // struct FieldType;

      // template <typename T>
      // struct FieldType<T,
      //                  typename
      //                  std::enable_if<is_scalar_type<T>::value>::type>
      // {
      //   static constexpr unsigned int n_components = 1;
      // };

      // template <int rank, int dim, typename T>
      // struct FieldType<Tensor<rank, dim, T>,
      //                  typename
      //                  std::enable_if<is_scalar_type<T>::value>::type>
      // {
      //   static constexpr unsigned int n_components =
      //     Tensor<rank, dim, T>::n_independent_components;
      // };

      // template <int rank, int dim, typename T>
      // struct FieldType<SymmetricTensor<rank, dim, T>,
      //                  typename
      //                  std::enable_if<is_scalar_type<T>::value>::type>
      // {
      //   static constexpr unsigned int n_components =
      //     SymmetricTensor<rank, dim, T>::n_independent_components;
      // };

      // template <typename T>
      // struct UnaryOpSubSpaceViewsHelper;

      // // For SubSpaceViews::Scalar and SubSpaceViews::Vector
      // template <template <class> typename SubSpaceViewsType,
      //           typename SpaceType,
      //           enum WeakForms::Operators::UnaryOpCodes OpCode,
      //           std::size_t                             solution_index>
      // struct UnaryOpSubSpaceViewsHelper<WeakForms::Operators::UnaryOp<
      //   SubSpaceViewsType<SpaceType>,
      //   OpCode,
      //   void,
      //   WeakForms::internal::SolutionIndex<solution_index>>>
      // {
      //   //  static const unsigned int space_dimension =
      //   //  SubSpaceViewsType<SpaceType>::space_dimension; static constexpr
      //   int
      //   //  n_components = FieldType<typename
      //   //
      //   SubSpaceViewsType<rank,SpaceType>::value_type<double>>::n_components;

      //   using FEValuesExtractorType =
      //     typename SubSpaceViewsType<SpaceType>::FEValuesExtractorType;
      // };

      // // For SubSpaceViews::Tensor and SubSpaceViews::SymmetricTensor
      // template <template <int, class> typename SubSpaceViewsType,
      //           typename SpaceType,
      //           int                                     rank,
      //           enum WeakForms::Operators::UnaryOpCodes OpCode,
      //           std::size_t                             solution_index>
      // struct UnaryOpSubSpaceViewsHelper<WeakForms::Operators::UnaryOp<
      //   SubSpaceViewsType<rank, SpaceType>,
      //   OpCode,
      //   void,
      //   WeakForms::internal::SolutionIndex<solution_index>>>
      // {
      //   //  static const unsigned int space_dimension =
      //   //  SubSpaceViewsType<rank,SpaceType>::space_dimension; static
      //   constexpr
      //   //  int n_components = FieldType<typename
      //   //
      //   SubSpaceViewsType<rank,SpaceType>::value_type<double>>::n_components;

      //   using FEValuesExtractorType =
      //     typename SubSpaceViewsType<rank, SpaceType>::FEValuesExtractorType;
      // };


      template <typename... UnaryOpsSubSpaceFieldSolution>
      struct UnaryOpsSubSpaceFieldSolutionHelper
      {
        using field_args_t = std::tuple<UnaryOpsSubSpaceFieldSolution...>;
        using field_extractors_t =
          std::tuple<typename UnaryOpsSubSpaceFieldSolution::extractor_type...>;

        static constexpr int
        n_operators()
        {
          return sizeof...(UnaryOpsSubSpaceFieldSolution);
        }

        static constexpr unsigned int
        get_n_components()
        {
          return unpack_n_components<UnaryOpsSubSpaceFieldSolution...>();
        }

        static field_extractors_t &&
        get_initialized_extractors()
        {
          field_extractors_t field_extractors;
          unsigned int       n_previous_field_components = 0;

          unpack_initialize_extractors<0, UnaryOpsSubSpaceFieldSolution...>(
            field_extractors, n_previous_field_components);

          return std::move(field_extractors);
        }

        template <typename UnaryOpField>
        static typename UnaryOpField::extractor_type
        get_initialized_extractor(const UnaryOpField &field,
                                  const field_args_t &field_args)
        {
          using Extractor_t = typename UnaryOpField::extractor_type;
          unsigned int n_previous_field_components = 0;

          unpack_n_previous_field_components<0,
                                             UnaryOpField,
                                             UnaryOpsSubSpaceFieldSolution...>(
            field, field_args, n_previous_field_components);

          return Extractor_t(n_previous_field_components);
        }

        // =============
        // AD operations
        // =============

        template <typename ADHelperType, int dim, int spacedim>
        static void
        ad_register_independent_variables(
          ADHelperType &                          ad_helper,
          MeshWorker::ScratchData<dim, spacedim> &scratch_data,
          const std::vector<std::string> &        solution_names,
          const unsigned int                      q_point,
          const field_args_t &                    field_args,
          const field_extractors_t &              field_extractors)
        {
          unpack_ad_register_independent_variables<
            0,
            ADHelperType,
            dim,
            spacedim,
            UnaryOpsSubSpaceFieldSolution...>(ad_helper,
                                              scratch_data,
                                              solution_names,
                                              q_point,
                                              field_args,
                                              field_extractors);
        }

        template <typename ADHelperType,
                  typename ADFunctionType,
                  int dim,
                  int spacedim>
        static auto
        ad_call_function(const ADHelperType &                    ad_helper,
                         const ADFunctionType &                  ad_function,
                         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &        solution_names,
                         const unsigned int                      q_point,
                         const field_extractors_t &field_extractors)
        {
          // https://riptutorial.com/cplusplus/example/26687/turn-a-std--tuple-t-----into-function-parameters
          return unpack_ad_call_function(
            ad_helper,
            ad_function,
            scratch_data,
            solution_names,
            q_point,
            field_extractors,
            std::make_index_sequence<
              std::tuple_size<field_extractors_t>::value>());
        }

      private:
        template <typename UnaryOpType>
        static constexpr unsigned int
        get_unary_op_field_n_components()
        {
          return UnaryOpType::n_components;

          // using ArbitraryType = double;
          // return FieldType<typename UnaryOpType::template value_type<
          //   ArbitraryType>>::n_components;
        }

        // End point
        template <typename UnaryOpType>
        static constexpr unsigned int
        unpack_n_components()
        {
          return get_unary_op_field_n_components<UnaryOpType>();
        }

        template <typename UnaryOpType, typename... OtherUnaryOpTypes>
        static constexpr
          typename std::enable_if<(sizeof...(OtherUnaryOpTypes) > 0),
                                  unsigned int>::type
          unpack_n_components()
        {
          return unpack_n_components<UnaryOpType>() +
                 unpack_n_components<OtherUnaryOpTypes...>();
        }

        template <std::size_t I = 0, typename... UnaryOpType>
          static
          typename std::enable_if < I<sizeof...(UnaryOpType), void>::type
                                    unpack_initialize_extractors(
                                      field_extractors_t &field_extractors,
                                      unsigned int &n_previous_field_components)
        {
          using FEValuesExtractorType = decltype(std::get<I>(field_extractors));
          std::get<I>(field_extractors) =
            FEValuesExtractorType(n_previous_field_components);

          // Move on to the next field, noting that we've allocated a certain
          // number of components to this scalar/vector/tensor field.
          using UnaryOp_t = typename std::decay<decltype(
            std::get<I>(std::declval<field_args_t>()))>::type;
          n_previous_field_components +=
            get_unary_op_field_n_components<UnaryOp_t>();
          unpack_initialize_extractors<I + 1, UnaryOpType...>(
            field_extractors, n_previous_field_components);
        }

        // End point
        template <std::size_t I = 0, typename... UnaryOpType>
        static typename std::enable_if<I == sizeof...(UnaryOpType), void>::type
        unpack_initialize_extractors(field_extractors_t &field_extractors,
                                     unsigned int &n_previous_field_components)
        {
          (void)field_extractors;
          (void)n_previous_field_components;
        }

        template <std::size_t I = 0,
                  typename UnaryOpField,
                  typename... UnaryOpType>
          static
          typename std::enable_if < I<sizeof...(UnaryOpType), void>::type
                                    unpack_n_previous_field_components(
                                      const UnaryOpField &field,
                                      const field_args_t &field_args,
                                      unsigned int &n_previous_field_components)
        {
          // Exit if we've found the entry in the tuple that matches the input
          // field. We can only do this through string matching, since multiple
          // fields might be using an op with the same signature.
          const SymbolicDecorations decorator;
          const auto &              listed_field = std::get<I>(field_args);
          if (listed_field.as_ascii(decorator) == field.as_ascii(decorator))
            return;

          // Move on to the next field, noting that we've allocated a certain
          // number of components to this scalar/vector/tensor field.
          using UnaryOp_t = typename std::decay<decltype(listed_field)>::type;
          n_previous_field_components +=
            get_unary_op_field_n_components<UnaryOp_t>();
          unpack_n_previous_field_components<I + 1,
                                             UnaryOpField,
                                             UnaryOpType...>(
            field, field_args, n_previous_field_components);
        }

        // End point
        template <std::size_t I = 0,
                  typename UnaryOpField,
                  typename... UnaryOpType>
        static typename std::enable_if<I == sizeof...(UnaryOpType), void>::type
        unpack_n_previous_field_components(
          const UnaryOpField &field,
          const field_args_t &field_args,
          unsigned int &      n_previous_field_components)
        {
          (void)field;
          (void)field_args;
          (void)n_previous_field_components;
          AssertThrow(false,
                      ExcMessage(
                        "Could not find UnaryOp for the field solution."));
        }

        // =============
        // AD operations
        // =============

        template <std::size_t I = 0,
                  typename ADHelperType,
                  int dim,
                  int spacedim,
                  typename... UnaryOpType>
          static typename std::enable_if <
          I<sizeof...(UnaryOpType), void>::type
          unpack_ad_register_independent_variables(
            ADHelperType &                          ad_helper,
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<std::string> &        solution_names,
            const unsigned int                      q_point,
            const std::tuple<UnaryOpType...> &      unary_op_field_solutions,
            const field_extractors_t &              field_extractors)
        {
          using scalar_type = typename ADHelperType::scalar_type;

          const auto &unary_op_field_solution =
            std::get<I>(unary_op_field_solutions);
          const auto &                       field_solutions =
            unary_op_field_solution.template operator()<scalar_type>(
              scratch_data, solution_names); // Cached solution at all QPs
          Assert(q_point < field_solutions.size(),
                 ExcIndexRange(q_point, 0, field_solutions.size()));
          const auto &field_solution  = field_solutions[q_point];
          const auto &field_extractor = get<I>(field_extractors);

          // std::cout << "Extr: " << field_extractor << " <-> Soln: " <<
          // field_solution << std::endl;

          ad_helper.register_independent_variable(field_solution,
                                                  field_extractor);

          unpack_ad_register_independent_variables<I + 1,
                                                   ADHelperType,
                                                   dim,
                                                   spacedim,
                                                   UnaryOpType...>(
            ad_helper,
            scratch_data,
            solution_names,
            q_point,
            unary_op_field_solutions,
            field_extractors);
        }

        // Get update flags from a unary op: End point
        template <std::size_t I = 0,
                  typename ADHelperType,
                  int dim,
                  int spacedim,
                  typename... UnaryOpType>
        static typename std::enable_if<I == sizeof...(UnaryOpType), void>::type
        unpack_ad_register_independent_variables(
          ADHelperType &                          ad_helper,
          MeshWorker::ScratchData<dim, spacedim> &scratch_data,
          const std::vector<std::string> &        solution_names,
          const unsigned int                      q_point,
          const std::tuple<UnaryOpType...> &      unary_op_field_solution,
          const field_extractors_t &              field_extractors)
        {
          // Do nothing
          (void)ad_helper;
          (void)scratch_data;
          (void)solution_names;
          (void)q_point;
          (void)unary_op_field_solution;
          (void)field_extractors;
        }

        template <typename ADHelperType,
                  typename ADFunctionType,
                  int dim,
                  int spacedim,
                  std::size_t... I>
        static auto
        unpack_ad_call_function(
          const ADHelperType &                    ad_helper,
          const ADFunctionType &                  ad_function,
          MeshWorker::ScratchData<dim, spacedim> &scratch_data,
          const std::vector<std::string> &        solution_names,
          const unsigned int &                    q_point,
          const field_extractors_t &              field_extractors,
          const std::index_sequence<I...>)
        {
          // https://riptutorial.com/cplusplus/example/26687/turn-a-std--tuple-t-----into-function-parameters
          return ad_function(scratch_data,
                             solution_names,
                             q_point,
                             ad_helper.get_sensitive_variables(
                               get<I>(field_extractors))...);
        }
      };
    } // namespace internal
  }   // namespace Operators
} // namespace WeakForms


namespace WeakForms
{
  template <typename... UnaryOpsSubSpaceFieldSolution>
  class EnergyFunctor : public WeakForms::Functor<0>
  {
    using Base = WeakForms::Functor<0>;

  public:
    template <typename ADSDNumberType>
    using value_type = ADSDNumberType;

    template <typename ADSDNumberType, int dim, int spacedim = dim>
    using function_type = std::function<value_type<ADSDNumberType>(
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &              solution_names,
      const unsigned int                            q_point,
      const typename UnaryOpsSubSpaceFieldSolution::template value_type<
        ADSDNumberType> &... field_solutions)>;

    template <typename ScalarType,
              enum Differentiation::AD::NumberTypes ADNumberTypeCode>
    using ad_type =
      typename Differentiation::AD::NumberTraits<ScalarType,
                                                 ADNumberTypeCode>::ad_type;
    template <typename ScalarType>
    using sd_type = Differentiation::SD::Expression;

    EnergyFunctor(
      const std::string &symbol_ascii,
      const std::string &symbol_latex,
      const UnaryOpsSubSpaceFieldSolution &... unary_op_field_solutions)
      : Base(symbol_ascii, symbol_latex)
      , unary_op_field_solutions(unary_op_field_solutions...)
    {}

    // ----  Ascii ----

    virtual std::string
    as_ascii(const SymbolicDecorations &decorator) const override
    {
      return Base::as_ascii(decorator) +
             decorator.unary_field_ops_as_ascii(get_field_args());
    }

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      return symbol_ascii;
    }

    // ---- LaTeX ----

    virtual std::string
    as_latex(const SymbolicDecorations &decorator) const override
    {
      return Base::as_latex(decorator) +
             decorator.unary_field_ops_as_latex(get_field_args());
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      return symbol_latex;
    }

    // Call operator to promote this class to a UnaryOp
    template <typename ADNumberType, int dim, int spacedim = dim>
    auto
    operator()(const function_type<ADNumberType, dim, spacedim> &function,
               const UpdateFlags update_flags) const;

    template <typename ADNumberType, int dim, int spacedim = dim>
    auto
    operator()(const function_type<ADNumberType, dim, spacedim> &function) const
    {
      return this->operator()<ADNumberType, dim, spacedim>(
        function, UpdateFlags::update_default);
    }

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    operator()(
      const function_type<SDNumberType, dim, spacedim> &function,
      const enum Differentiation::SD::OptimizerType     optimization_method,
      const enum Differentiation::SD::OptimizationFlags optimization_flags,
      const UpdateFlags                                 update_flags) const;

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    operator()(
      const function_type<SDNumberType, dim, spacedim> &function,
      const enum Differentiation::SD::OptimizerType     optimization_method,
      const enum Differentiation::SD::OptimizationFlags optimization_flags)
      const
    {
      return this->operator()<SDNumberType, dim, spacedim>(
        function,
        optimization_method,
        optimization_flags,
        UpdateFlags::update_default);
    }

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename ADNumberType, int dim, int spacedim = dim>
    auto
    value(const function_type<ADNumberType, dim, spacedim> &function,
          const UpdateFlags                                 update_flags) const
    {
      return this->operator()<ADNumberType, dim, spacedim>(function,
                                                           update_flags);
    }

    template <typename ADNumberType, int dim, int spacedim = dim>
    auto
    value(const function_type<ADNumberType, dim, spacedim> &function) const
    {
      return this->operator()<ADNumberType, dim, spacedim>(function);
    }

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    value(const function_type<SDNumberType, dim, spacedim> &function,
          const enum Differentiation::SD::OptimizerType     optimization_method,
          const enum Differentiation::SD::OptimizationFlags optimization_flags,
          const UpdateFlags                                 update_flags) const
    {
      return this->operator()<SDNumberType, dim, spacedim>(function,
                                                           optimization_method,
                                                           optimization_flags,
                                                           update_flags);
    }

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    value(const function_type<SDNumberType, dim, spacedim> &function,
          const enum Differentiation::SD::OptimizerType     optimization_method,
          const enum Differentiation::SD::OptimizationFlags optimization_flags)
      const
    {
      return this->operator()<SDNumberType, dim, spacedim>(function,
                                                           optimization_method,
                                                           optimization_flags);
    }

    const std::tuple<UnaryOpsSubSpaceFieldSolution...> &
    get_field_args() const
    {
      return unary_op_field_solutions;
    }

  private:
    const std::tuple<UnaryOpsSubSpaceFieldSolution...> unary_op_field_solutions;
  };

  // namespace AutoDifferentiation
  // {

  //   template <int dim>
  //   using VectorFunctor = WeakForms::VectorFunctor<dim>;

  //   template <int rank, int dim>
  //   using TensorFunctor = WeakForms::TensorFunctor<rank, dim>;

  //   template <int rank, int dim>
  //   using SymmetricTensorFunctor = WeakForms::SymmetricTensorFunctor<rank,
  //   dim>;

  //   template <int dim>
  //   using ScalarFunctionFunctor = WeakForms::ScalarFunctionFunctor<dim>;

  //   template <int rank, int dim>
  //   using TensorFunctionFunctor = WeakForms::TensorFunctionFunctor<rank,
  //   dim>;

  // } // namespace AutoDifferentiation

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    /* ------------------------ Functors: Custom ------------------------ */

    /**
     * Extract the value from a scalar functor.
     *
     * Variant for auto-differentiable number.
     */
    template <typename ADNumberType,
              int dim,
              int spacedim,
              typename... UnaryOpsSubSpaceFieldSolution>
    class UnaryOp<
      EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
      UnaryOpCodes::value,
      typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
      ADNumberType,
      WeakForms::internal::DimPack<dim, spacedim>>
    {
      static_assert(Differentiation::AD::is_ad_number<ADNumberType>::value,
                    "Expected an AD number.");

      using Op = EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>;

      using OpHelper_t = internal::UnaryOpsSubSpaceFieldSolutionHelper<
        UnaryOpsSubSpaceFieldSolution...>;

    public:
      using scalar_type =
        typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type;

      static constexpr enum Differentiation::AD::NumberTypes ADNumberTypeCode =
        Differentiation::AD::ADNumberTraits<ADNumberType>::type_code;

      using ad_helper_type = Differentiation::AD::
        ScalarFunction<spacedim, ADNumberTypeCode, scalar_type>;
      using ad_type = typename ad_helper_type::ad_type;

      static_assert(
        std::is_same<typename Differentiation::AD::
                       NumberTraits<scalar_type, ADNumberTypeCode>::ad_type,
                     ADNumberType>::value,
        "AD types not the same.");
      static_assert(std::is_same<ad_type, ADNumberType>::value,
                    "AD types not the same.");

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType, dim, spacedim>;

      template <typename ResultScalarType>
      using return_type = void;
      // using return_type = std::vector<value_type<ResultScalarType>>;

      using ad_function_type = function_type<ad_type>;

      static const int rank = 0;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &              operand,
                       const ad_function_type &function,
                       const UpdateFlags       update_flags)
        : operand(operand)
        , function(function)
        , update_flags(update_flags)
        , extractors(OpHelper_t::get_initialized_extractors())
      {}

      // explicit UnaryOp(const Op &operand)
      //   : UnaryOp(operand, [](const unsigned int) { return ad_type{}; })
      // {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(
          naming.value, operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(
          naming.value, operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return update_flags;
      }

      const ad_helper_type &
      get_derivative_helper(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache.get_object_with_name<ad_helper_type>(get_name_ad_helper());
      }

      template <typename UnaryOpField>
      typename UnaryOpField::extractor_type
      get_derivative_extractor(const UnaryOpField &field) const
      {
        return OpHelper_t::get_initialized_extractor(field, get_field_args());
      }

      const std::vector<Vector<scalar_type>> &
      get_gradients(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache.get_object_with_name<std::vector<Vector<scalar_type>>>(
          get_name_gradient());
      }

      const std::vector<FullMatrix<scalar_type>> &
      get_hessians(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache.get_object_with_name<std::vector<FullMatrix<scalar_type>>>(
          get_name_hessian());
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType, int dim2>
      return_type<ResultScalarType>
      operator()(MeshWorker::ScratchData<dim2, spacedim> &scratch_data,
                 const std::vector<std::string> &         solution_names) const
      {
        // Follow the recipe described in the documentation:
        // - Initialize helper.
        // - Register independent variables and set the values for all fields.
        // - Extract the sensitivities.
        // - Use sensitivities in AD functor.
        // - Register the definition of the total stored energy.
        // - Compute gradient, linearization, etc.
        // - Later, extract the desired components of the gradient,
        //   linearization etc.

        // Note: All user functions have the same parameterisation, so we can
        // use the same ADHelper for each of them. This does not restrict the
        // user to use the same definition for the energy itself at each QP!
        ad_helper_type &ad_helper = get_mutable_ad_helper(scratch_data);
        std::vector<Vector<scalar_type>> &Dpsi =
          get_mutable_gradients(scratch_data, ad_helper);
        std::vector<FullMatrix<scalar_type>> &D2psi =
          get_mutable_hessians(scratch_data, ad_helper);

        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        // In the HP case, we might traverse between cells with a different
        // number of quadrature points. So we need to resize the output data
        // accordingly.
        if (Dpsi.size() != fe_values.n_quadrature_points ||
            D2psi.size() != fe_values.n_quadrature_points)
          {
            Dpsi.resize(fe_values.n_quadrature_points);
            D2psi.resize(fe_values.n_quadrature_points);
          }

        for (const auto &q_point : fe_values.quadrature_point_indices())
          {
            ad_helper.reset();

            // Register the independent variables. The actual field solution at
            // the quadrature point is fetched from the scratch_data cache. It
            // is paired with its counterpart extractor, which should not have
            // any indiced overlapping with the extractors for the other fields
            // in the field_args.
            OpHelper_t::ad_register_independent_variables(
              ad_helper,
              scratch_data,
              solution_names,
              q_point,
              get_field_args(),
              get_field_extractors());

            // Evaluate the functor to compute the total stored energy.
            // To do this, we extract all sensitivities and pass them directly
            // in the user-provided function.
            const ad_type psi =
              OpHelper_t::ad_call_function(ad_helper,
                                           function,
                                           scratch_data,
                                           solution_names,
                                           q_point,
                                           get_field_extractors());

            // Register the definition of the total stored energy
            ad_helper.register_dependent_variable(psi);

            // Store the output function value, its gradient and linearization.
            ad_helper.compute_gradient(Dpsi[q_point]);
            ad_helper.compute_hessian(D2psi[q_point]);
          }
      }

      const Op &
      get_op() const
      {
        return operand;
      }

      const typename OpHelper_t::field_args_t &
      get_field_args() const
      {
        // Get the unary op field solutions from the EnergyFunctor
        return get_op().get_field_args();
      }

      const typename OpHelper_t::field_extractors_t &
      get_field_extractors() const
      {
        return extractors;
      }

    private:
      const Op               operand;
      const ad_function_type function;
      // Some additional update flags that the user might require in order to
      // evaluate their AD function (e.g. UpdateFlags::update_quadrature_points)
      const UpdateFlags update_flags;

      const typename OpHelper_t::field_extractors_t
        extractors; // FEValuesExtractors to work with multi-component fields

      std::string
      get_name_ad_helper() const
      {
        const SymbolicDecorations decorator;
        return "_deal_II__EnergyFunctor_ADHelper_" +
               operand.as_ascii(decorator);
      }

      std::string
      get_name_gradient() const
      {
        const SymbolicDecorations decorator;
        return "_deal_II__EnergyFunctor_ADHelper_Gradients_" +
               operand.as_ascii(decorator);
      }

      std::string
      get_name_hessian() const
      {
        const SymbolicDecorations decorator;
        return "_deal_II__EnergyFunctor_ADHelper_Hessians_" +
               operand.as_ascii(decorator);
      }

      ad_helper_type &
      get_mutable_ad_helper(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        GeneralDataStorage &cache = scratch_data.get_general_data_storage();
        const std::string   name_ad_helper = get_name_ad_helper();

        // Unfortunately we cannot perform a check like this because the
        // ScratchData is reused by many cells during the mesh loop. So
        // there's no real way to verify that the user is not accidentally
        // re-using an object because they forget to uniquely name the
        // EnergyFunctor upon which this op is based.
        //
        // Assert(!(cache.stores_object_with_name(name_ad_helper)),
        //        ExcMessage("ADHelper is already present in the cache."));

        return cache.get_or_add_object_with_name<ad_helper_type>(
          name_ad_helper, OpHelper_t::get_n_components());
      }

      std::vector<Vector<scalar_type>> &
      get_mutable_gradients(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data,
        const ad_helper_type &                  ad_helper) const
      {
        GeneralDataStorage &cache = scratch_data.get_general_data_storage();
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        return cache
          .get_or_add_object_with_name<std::vector<Vector<scalar_type>>>(
            get_name_gradient(),
            fe_values.n_quadrature_points,
            Vector<scalar_type>(ad_helper.n_dependent_variables()));
      }

      std::vector<FullMatrix<scalar_type>> &
      get_mutable_hessians(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                           const ad_helper_type &ad_helper) const
      {
        GeneralDataStorage &cache = scratch_data.get_general_data_storage();
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        return cache
          .get_or_add_object_with_name<std::vector<FullMatrix<scalar_type>>>(
            get_name_hessian(),
            fe_values.n_quadrature_points,
            FullMatrix<scalar_type>(ad_helper.n_dependent_variables(),
                                    ad_helper.n_independent_variables()));
      }
    };



    /**
     * Extract the value from a scalar functor.
     *
     * Variant for symbolic expressions.
     */
    template <int dim, int spacedim, typename... UnaryOpsSubSpaceFieldSolution>
    class UnaryOp<EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
                  UnaryOpCodes::value,
                  void,
                  Differentiation::SD::Expression,
                  WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>;

      using OpHelper_t = internal::UnaryOpsSubSpaceFieldSolutionHelper<
        UnaryOpsSubSpaceFieldSolution...>;

    public:
      using scalar_type =
        std::nullptr_t; // SD expressions can represent anything
      template <typename ReturnType>
      using sd_helper_type = Differentiation::SD::BatchOptimizer<ReturnType>;
      using sd_type        = Differentiation::SD::Expression;

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType, dim, spacedim>;

      template <typename ResultScalarType>
      using return_type = void;

      using sd_function_type = function_type<sd_type>;

      static const int rank = 0;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(
        const Op &                                        operand,
        const sd_function_type &                          function,
        const enum Differentiation::SD::OptimizerType     optimization_method,
        const enum Differentiation::SD::OptimizationFlags optimization_flags,
        const UpdateFlags                                 update_flags)
        : operand(operand)
        , function(function)
        , optimization_method(optimization_method)
        , optimization_flags(optimization_flags)
        , update_flags(update_flags)
        , extractors(OpHelper_t::get_initialized_extractors())
      {}

      // explicit UnaryOp(const Op &operand)
      //   : UnaryOp(operand, [](const unsigned int) { return ad_type{}; })
      // {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(
          naming.value, operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(
          naming.value, operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return update_flags;
      }

      template <typename ResultScalarType>
      const sd_helper_type<ResultScalarType> &
      get_derivative_helper(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache.get_object_with_name<sd_helper_type<ResultScalarType>>(
          get_name_sd_batch_optimizer());
      }

      template <typename UnaryOpField>
      typename UnaryOpField::template value_type<sd_type>
      get_derivative_extractor(const UnaryOpField &field) const
      {
        AssertThrow(false, ExcNotImplemented());
        return typename UnaryOpField::template value_type<sd_type>();
        // return OpHelper_t::get_initialized_extractor(field,
        // get_field_args());
      }

      template <typename ResultScalarType>
      const std::vector<std::vector<ResultScalarType>> &
      get_evaluated_dependent_functions(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache
          .get_object_with_name<std::vector<std::vector<ResultScalarType>>>(
            get_name_evaluated_dependent_functions());
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType, int dim2>
      return_type<ResultScalarType>
      operator()(MeshWorker::ScratchData<dim2, spacedim> &scratch_data,
                 const std::vector<std::string> &         solution_names) const
      {
        AssertThrow(false, ExcNotImplemented());

        // // Follow the recipe described in the documentation:
        // // - Initialize helper.
        // // - Register independent variables and set the values for all
        // fields.
        // // - Extract the sensitivities.
        // // - Use sensitivities in AD functor.
        // // - Register the definition of the total stored energy.
        // // - Compute gradient, linearization, etc.
        // // - Later, extract the desired components of the gradient,
        // //   linearization etc.

        // // Note: All user functions have the same parameterisation, so we can
        // // use the same ADHelper for each of them. This does not restrict the
        // // user to use the same definition for the energy itself at each QP!
        // sd_helper_type &ad_helper = get_mutable_ad_helper(scratch_data);
        // std::vector<Vector<scalar_type>> &Dpsi =
        //   get_mutable_gradients(scratch_data, ad_helper);
        // std::vector<FullMatrix<scalar_type>> &D2psi =
        //   get_mutable_hessians(scratch_data, ad_helper);

        // const FEValuesBase<dim, spacedim> &fe_values =
        //   scratch_data.get_current_fe_values();

        // // In the HP case, we might traverse between cells with a different
        // // number of quadrature points. So we need to resize the output data
        // // accordingly.
        // if (Dpsi.size() != fe_values.n_quadrature_points ||
        //     D2psi.size() != fe_values.n_quadrature_points)
        //   {
        //     Dpsi.resize(fe_values.n_quadrature_points);
        //     D2psi.resize(fe_values.n_quadrature_points);
        //   }

        // for (const auto &q_point : fe_values.quadrature_point_indices())
        //   {
        //     ad_helper.reset();

        //     // Register the independent variables. The actual field solution
        //     at
        //     // the quadrature point is fetched from the scratch_data cache.
        //     It
        //     // is paired with its counterpart extractor, which should not
        //     have
        //     // any indiced overlapping with the extractors for the other
        //     fields
        //     // in the field_args.
        //     OpHelper_t::ad_register_independent_variables(
        //       ad_helper,
        //       scratch_data,
        //       solution_names,
        //       q_point,
        //       get_field_args(),
        //       get_field_extractors());

        //     // Evaluate the functor to compute the total stored energy.
        //     // To do this, we extract all sensitivities and pass them
        //     directly
        //     // in the user-provided function.
        //     const sd_type psi =
        //       OpHelper_t::ad_call_function(ad_helper,
        //                                    function,
        //                                    scratch_data,
        //                                    solution_names,
        //                                    q_point,
        //                                    get_field_extractors());

        //     // Register the definition of the total stored energy
        //     ad_helper.register_dependent_variable(psi);

        //     // Store the output function value, its gradient and
        //     linearization. ad_helper.compute_gradient(Dpsi[q_point]);
        //     ad_helper.compute_hessian(D2psi[q_point]);
        //   }
      }

      const Op &
      get_op() const
      {
        return operand;
      }

      const typename OpHelper_t::field_args_t &
      get_field_args() const
      {
        // Get the unary op field solutions from the EnergyFunctor
        return get_op().get_field_args();
      }

      const typename OpHelper_t::field_extractors_t &
      get_field_extractors() const
      {
        return extractors;
      }

    private:
      const Op                                          operand;
      const sd_function_type                            function;
      const enum Differentiation::SD::OptimizerType     optimization_method;
      const enum Differentiation::SD::OptimizationFlags optimization_flags;
      // Some additional update flags that the user might require in order to
      // evaluate their AD function (e.g. UpdateFlags::update_quadrature_points)
      const UpdateFlags update_flags;

      const typename OpHelper_t::field_extractors_t
        extractors; // FEValuesExtractors to work with multi-component fields

      std::string
      get_name_sd_batch_optimizer() const
      {
        const SymbolicDecorations decorator;
        return "_deal_II__EnergyFunctor_SDBatchOptimizer_" +
               operand.as_ascii(decorator);
      }

      std::string
      get_name_evaluated_dependent_functions() const
      {
        const SymbolicDecorations decorator;
        return "_deal_II__EnergyFunctor_ADHelper_Evaluated_Dependent_Functions" +
               operand.as_ascii(decorator);
      }

      // std::string
      // get_name_gradient() const
      // {
      //   const SymbolicDecorations decorator;
      //   return "_deal_II__EnergyFunctor_ADHelper_Gradients_" +
      //          operand.as_ascii(decorator);
      // }

      // std::string
      // get_name_hessian() const
      // {
      //   const SymbolicDecorations decorator;
      //   return "_deal_II__EnergyFunctor_ADHelper_Hessians_" +
      //          operand.as_ascii(decorator);
      // }

      // sd_helper_type &
      // get_mutable_ad_helper(
      //   MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      // {
      //   GeneralDataStorage &cache = scratch_data.get_general_data_storage();
      //   const std::string   name_ad_helper = get_name_sd_batch_optimizer();

      //   // Unfortunately we cannot perform a check like this because the
      //   // ScratchData is reused by many cells during the mesh loop. So
      //   // there's no real way to verify that the user is not accidentally
      //   // re-using an object because they forget to uniquely name the
      //   // EnergyFunctor upon which this op is based.
      //   //
      //   // Assert(!(cache.stores_object_with_name(name_ad_helper)),
      //   //        ExcMessage("ADHelper is already present in the cache."));

      //   return cache.get_or_add_object_with_name<sd_helper_type>(
      //     name_ad_helper, OpHelper_t::get_n_components());
      // }

      template <typename ResultScalarType>
      std::vector<std::vector<ResultScalarType>> &
      get_mutable_evaluated_dependent_functions(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data,
        const sd_helper_type<ResultScalarType> &sd_helper) const
      {
        GeneralDataStorage &cache = scratch_data.get_general_data_storage();
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        return cache.get_or_add_object_with_name<
          std::vector<std::vector<ResultScalarType>>>(
          get_name_evaluated_dependent_functions(),
          fe_values.n_quadrature_points,
          std::vector<ResultScalarType>(sd_helper.n_dependent_variables()));
      }

      // std::vector<Vector<scalar_type>> &
      // get_mutable_gradients(
      //   MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      //   const sd_helper_type &                  ad_helper) const
      // {
      //   GeneralDataStorage &cache = scratch_data.get_general_data_storage();
      //   const FEValuesBase<dim, spacedim> &fe_values =
      //     scratch_data.get_current_fe_values();

      //   return cache
      //     .get_or_add_object_with_name<std::vector<Vector<scalar_type>>>(
      //       get_name_gradient(),
      //       fe_values.n_quadrature_points,
      //       Vector<scalar_type>(ad_helper.n_dependent_variables()));
      // }

      // std::vector<FullMatrix<scalar_type>> &
      // get_mutable_hessians(MeshWorker::ScratchData<dim, spacedim>
      // &scratch_data,
      //                      const sd_helper_type &ad_helper) const
      // {
      //   GeneralDataStorage &cache = scratch_data.get_general_data_storage();
      //   const FEValuesBase<dim, spacedim> &fe_values =
      //     scratch_data.get_current_fe_values();

      //   return cache
      //     .get_or_add_object_with_name<std::vector<FullMatrix<scalar_type>>>(
      //       get_name_hessian(),
      //       fe_values.n_quadrature_points,
      //       FullMatrix<scalar_type>(ad_helper.n_dependent_variables(),
      //                               ad_helper.n_independent_variables()));
      // }
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /**
   * Shortcut so that we don't need to do something like this:
   *
   * <code>
   * const FieldSolution<dim> solution;
   * const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s",
   * "s");
   *
   * const auto soln_ss   = solution[subspace_extractor];
   * const auto soln_val  = soln_ss.value();    // Solution value
   * const auto soln_grad = soln_ss.gradient(); // Solution gradient
   * ...
   *
   * const EnergyFunctor<decltype(soln_val), decltype(soln_grad), ...>
   * energy("e", "\\Psi", soln_val, soln_grad, ...);
   * </code>
   */
  template <typename... UnaryOpsSubSpaceFieldSolution>
  EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>
  energy_functor(
    const std::string &symbol_ascii,
    const std::string &symbol_latex,
    const UnaryOpsSubSpaceFieldSolution &... unary_op_field_solutions)
  {
    return EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>(
      symbol_ascii, symbol_latex, unary_op_field_solutions...);
  }


  template <typename ADNumberType,
            int dim,
            int spacedim = dim,
            typename... UnaryOpsSubSpaceFieldSolution,
            typename = typename std::enable_if<
              Differentiation::AD::is_ad_number<ADNumberType>::value>::type>
  WeakForms::Operators::UnaryOp<
    WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
    WeakForms::Operators::UnaryOpCodes::value,
    typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
    ADNumberType,
    internal::DimPack<dim, spacedim>>
  value(
    const WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...> &operand,
    const typename WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::
      template function_type<ADNumberType, dim, spacedim> &function,
    const UpdateFlags                                      update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>;
    using ScalarType =
      typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           ScalarType,
                           ADNumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function, update_flags);
  }


  template <typename ADNumberType,
            int dim,
            int spacedim = dim,
            typename... UnaryOpsSubSpaceFieldSolution,
            typename = typename std::enable_if<
              Differentiation::AD::is_ad_number<ADNumberType>::value>::type>
  WeakForms::Operators::UnaryOp<
    WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
    WeakForms::Operators::UnaryOpCodes::value,
    typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
    ADNumberType,
    internal::DimPack<dim, spacedim>>
  value(
    const WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...> &operand,
    const typename WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::
      template function_type<ADNumberType, dim, spacedim> &function,
    const enum Differentiation::SD::OptimizerType          optimization_method,
    const enum Differentiation::SD::OptimizationFlags      optimization_flags)
  {
    return WeakForms::value<ADNumberType, dim, spacedim>(
      operand, function, UpdateFlags::update_default);
  }



  template <typename SDNumberType,
            int dim,
            int spacedim = dim,
            typename... UnaryOpsSubSpaceFieldSolution,
            typename = typename std::enable_if<
              Differentiation::SD::is_sd_number<SDNumberType>::value>::type>
  WeakForms::Operators::UnaryOp<
    WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
    WeakForms::Operators::UnaryOpCodes::value,
    void,
    SDNumberType,
    internal::DimPack<dim, spacedim>>
  value(
    const WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...> &operand,
    const typename WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::
      template function_type<SDNumberType, dim, spacedim> &function,
    const enum Differentiation::SD::OptimizerType          optimization_method,
    const enum Differentiation::SD::OptimizationFlags      optimization_flags,
    const UpdateFlags                                      update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           void,
                           SDNumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(
      operand, function, optimization_method, optimization_flags, update_flags);
  }



  template <typename SDNumberType,
            int dim,
            int spacedim = dim,
            typename... UnaryOpsSubSpaceFieldSolution,
            typename = typename std::enable_if<
              Differentiation::SD::is_sd_number<SDNumberType>::value>::type>
  WeakForms::Operators::UnaryOp<
    WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
    WeakForms::Operators::UnaryOpCodes::value,
    void,
    SDNumberType,
    internal::DimPack<dim, spacedim>>
  value(
    const WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...> &operand,
    const typename WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::
      template function_type<SDNumberType, dim, spacedim> &function)
  {
    return WeakForms::value<SDNumberType, dim, spacedim>(
      operand, function, UpdateFlags::update_default);
  }


  // ======


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



/* ==================== Class method definitions ==================== */

namespace WeakForms
{
  template <typename... UnaryOpsSubSpaceFieldSolution>
  template <typename ADNumberType, int dim, int spacedim>
  auto
  EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::operator()(
    const typename WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::
      template function_type<ADNumberType, dim, spacedim> &function,
    const UpdateFlags                                      update_flags) const
  {
    return WeakForms::value<ADNumberType, dim, spacedim>(*this,
                                                         function,
                                                         update_flags);
  }


  template <typename... UnaryOpsSubSpaceFieldSolution>
  template <typename SDNumberType, int dim, int spacedim>
  auto
  EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::operator()(
    const typename WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::
      template function_type<SDNumberType, dim, spacedim> &function,
    const enum Differentiation::SD::OptimizerType          optimization_method,
    const enum Differentiation::SD::OptimizationFlags      optimization_flags,
    const UpdateFlags                                      update_flags) const
  {
    return WeakForms::value<SDNumberType, dim, spacedim>(
      *this, function, optimization_method, optimization_flags, update_flags);
  }

} // namespace WeakForms



#ifndef DOXYGEN


namespace WeakForms
{
  // Decorator classes

  // template <int dim, int spacedim>
  // struct is_ad_functor<FieldSolution<dim, spacedim>> : std::true_type
  // {};

  // Unary operations

  template <typename ADNumberType,
            int dim,
            int spacedim,
            typename... UnaryOpsSubSpaceFieldSolution>
  struct is_ad_functor<Operators::UnaryOp<
    EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
    Operators::UnaryOpCodes::value,
    typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
    ADNumberType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};


  template <int dim, int spacedim, typename... UnaryOpsSubSpaceFieldSolution>
  struct is_sd_functor<
    Operators::UnaryOp<EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
                       Operators::UnaryOpCodes::value,
                       void,
                       Differentiation::SD::Expression,
                       WeakForms::internal::DimPack<dim, spacedim>>>
    : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_energy_functor_h
