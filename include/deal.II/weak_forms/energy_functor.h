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

#include <deal.II/differentiation/ad.h>
#include <deal.II/differentiation/sd.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/weak_forms/differentiation.h>
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


      // // TODO: This is replicated in self_linearizing_forms.h
      // template <typename T>
      // class is_scalar_type
      // {
      //   // See has_begin_and_end() in template_constraints.h
      //   // and https://stackoverflow.com/a/10722840

      //   template <typename A>
      //   static constexpr auto
      //   test(int) -> decltype(std::declval<typename
      //   EnableIfScalar<A>::type>(),
      //                         std::true_type())
      //   {
      //     return true;
      //   }

      //   template <typename A>
      //   static std::false_type
      //   test(...);

      // public:
      //   using type = decltype(test<T>(0));

      //   static const bool value = type::value;
      // };


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

      // // https://stackoverflow.com/a/11251408
      // // https://stackoverflow.com/a/13101086
      // template<typename T, typename U>
      // struct is_specialization_of;

      // template < typename NonTemplate, typename T >
      // struct is_specialization_of<NonTemplate,T> : std::false_type {};

      // template < template <typename...> class Template, typename T >
      // struct is_specialization_of<Template, T> : std::false_type {};

      // // template < template <typename...> class Template, typename T >
      // // struct is_specialization_of : std::false_type {};

      // template < template <typename...> class Template, typename... Args >
      // struct is_specialization_of< Template, Template<Args...> > :
      // std::true_type {};

      template <typename>
      struct is_tuple : std::false_type
      {};
      template <typename... T>
      struct is_tuple<std::tuple<T...>> : std::true_type
      {};


      // ===================
      // SD helper functions
      // ===================

      template <typename ReturnType>
      typename std::enable_if<
        std::is_same<ReturnType, Differentiation::SD::Expression>::value,
        ReturnType>::type
      make_symbolic(const std::string &name)
      {
        return Differentiation::SD::make_symbol(name);
      }

      template <typename ReturnType>
      typename std::enable_if<
        std::is_same<ReturnType,
                     Tensor<ReturnType::rank,
                            ReturnType::dimension,
                            Differentiation::SD::Expression>>::value,
        ReturnType>::type
      make_symbolic(const std::string &name)
      {
        constexpr int rank = ReturnType::rank;
        constexpr int dim  = ReturnType::dimension;
        return Differentiation::SD::make_tensor_of_symbols<rank, dim>(name);
      }

      template <typename ReturnType>
      typename std::enable_if<
        std::is_same<ReturnType,
                     SymmetricTensor<ReturnType::rank,
                                     ReturnType::dimension,
                                     Differentiation::SD::Expression>>::value,
        ReturnType>::type
      make_symbolic(const std::string &name)
      {
        constexpr int rank = ReturnType::rank;
        constexpr int dim  = ReturnType::dimension;
        return Differentiation::SD::make_symmetric_tensor_of_symbols<rank, dim>(
          name);
      }

      template <typename ExpressionType, typename UnaryOpField>
      typename UnaryOpField::template value_type<ExpressionType>
      make_symbolic(const UnaryOpField &       field,
                    const SymbolicDecorations &decorator)
      {
        using ReturnType =
          typename UnaryOpField::template value_type<ExpressionType>;

        const std::string name = "_deal_II__Field_" + field.as_ascii(decorator);
        return make_symbolic<ReturnType>(name);
      }


      template <typename... UnaryOpsSubSpaceFieldSolution>
      struct UnaryOpsSubSpaceFieldSolutionHelper
      {
        // ===================
        // AD type definitions
        // ===================

        using field_args_t = std::tuple<UnaryOpsSubSpaceFieldSolution...>;
        using field_extractors_t =
          std::tuple<typename UnaryOpsSubSpaceFieldSolution::extractor_type...>;


        // ===================
        // SD type definitions
        // ===================
        template <typename ScalarType>
        using field_values_t =
          std::tuple<typename UnaryOpsSubSpaceFieldSolution::
                       template value_type<ScalarType>...>;

        // Typical use case expects FunctionType to be an SD:Expression,
        // or a tensor of SD:Expressions. ScalarType should be a scalar
        // expression type.

        template <typename ScalarType, typename FunctionType>
        using first_derivatives_value_t =
          typename WeakForms::internal::Differentiation::
            DiffOpResult<FunctionType, field_values_t<ScalarType>>::type;

        template <typename ScalarType, typename FunctionType>
        using second_derivatives_value_t =
          typename WeakForms::internal::Differentiation::DiffOpResult<
            first_derivatives_value_t<ScalarType, FunctionType>,
            field_values_t<ScalarType>>::type;

        //--------
        // template<typename ScalarType, typename FunctionType, typename
        // UnaryOpSubSpaceFieldSolution> using first_derivative_t =
        // WeakForms::internal::Differentiation::
        //   DiffOpResult<FunctionType, typename
        //   UnaryOpSubSpaceFieldSolution::template
        //   value_type<ScalarType>>::type;

        // template<typename ScalarType, typename FunctionType>
        // using first_derivatives_t = std::tuple<first_derivative_t<ScalarType,
        // FunctionType, UnaryOpsSubSpaceFieldSolution>...>;

        // template<typename ScalarType, typename FunctionType, typename
        // UnaryOpSubSpaceFieldSolution_1, typename
        // UnaryOpSubSpaceFieldSolution_2> using second_first_derivative_t =
        // first_derivative_t<ScalarType, first_derivative_t<ScalarType,
        // FunctionType, UnaryOpSubSpaceFieldSolution_1>,
        // UnaryOpSubSpaceFieldSolution_2>;
        //--------

        // template<typename ScalarType, typename FunctionType>
        // using second_derivatives_t =
        // std::tuple<WeakForms::internal::Differentiation::
        //   DiffOpResult<first_derivatives_t<ScalarType,FunctionType>, typename
        //   UnaryOpsSubSpaceFieldSolution::value_type<ScalarType>...>>;

        // ========================
        // Generic helper functions
        // ========================

        static constexpr int
        n_operators()
        {
          return sizeof...(UnaryOpsSubSpaceFieldSolution);
        }

        // ===================
        // AD helper functions
        // ===================

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

        // ===================
        // SD helper functions
        // ===================

        template <typename SDNumberType>
        static field_values_t<SDNumberType>
        get_symbolic_fields(const field_args_t &      field_args,
                            const SymbolicDecorations decorator)
        {
          return unpack_get_symbolic_fields<SDNumberType>(
            field_args,
            decorator,
            std::make_index_sequence<std::tuple_size<field_args_t>::value>());
        }

        template <typename SDNumberType>
        static Differentiation::SD::types::substitution_map
        sd_get_symbol_map(
          const field_values_t<SDNumberType> &symbolic_field_values)
        {
          return unpack_sd_get_symbol_map<SDNumberType>(
            symbolic_field_values,
            std::make_index_sequence<
              std::tuple_size<field_values_t<SDNumberType>>::value>());
        }

        template <typename SDNumberType, typename SDFunctionType>
        static auto
        sd_call_function(
          const SDFunctionType &              sd_function,
          const field_values_t<SDNumberType> &symbolic_field_values)
        {
          return unpack_sd_call_function<SDNumberType>(
            sd_function,
            symbolic_field_values,
            std::make_index_sequence<
              std::tuple_size<field_values_t<SDNumberType>>::value>());
        }

        // Expect SDSubstitutionFunctionType to be a std::function
        template <typename SDNumberType, typename SDSubstitutionFunctionType>
        static Differentiation::SD::types::substitution_map
        sd_call_substitution_function(
          const SDSubstitutionFunctionType &  substitution_function,
          const field_values_t<SDNumberType> &symbolic_field_values)
        {
          if (substitution_function)
            return unpack_sd_call_substitution_function<SDNumberType>(
              substitution_function,
              symbolic_field_values,
              std::make_index_sequence<
                std::tuple_size<field_values_t<SDNumberType>>::value>());
          else
            return Differentiation::SD::types::substitution_map{};
        }

        // SDExpressionType can be an SD::Expression or a tensor of expressions.
        // Tuples of the former types are dealt with by the other variant.
        // Unfortunately it looks like we need to use the SFINAE idiom to help
        // the compiler, as it might try to implicitly convert these types
        // to tuples and get confused between the two functions.
        template <typename SDNumberType,
                  typename SDExpressionType,
                  typename = typename std::enable_if<
                    !is_tuple<SDExpressionType>::value>::type>
        static first_derivatives_value_t<SDNumberType, SDExpressionType>
        sd_differentiate(
          const SDExpressionType &            sd_expression,
          const field_values_t<SDNumberType> &symbolic_field_values)
        {
          return unpack_sd_differentiate<SDNumberType>(
            sd_expression,
            symbolic_field_values,
            std::make_index_sequence<
              std::tuple_size<field_values_t<SDNumberType>>::value>());
        }

        template <typename SDNumberType, typename... SDExpressionTypes>
        static std::tuple<
          first_derivatives_value_t<SDNumberType, SDExpressionTypes>...>
        sd_differentiate(
          const std::tuple<SDExpressionTypes...> &sd_expressions,
          const field_values_t<SDNumberType> &    symbolic_field_values)
        {
          return unpack_sd_differentiate<SDNumberType>(
            sd_expressions,
            symbolic_field_values,
            std::make_index_sequence<
              std::tuple_size<std::tuple<SDExpressionTypes...>>::value>(),
            std::make_index_sequence<
              std::tuple_size<field_values_t<SDNumberType>>::value>());
        }

        template <typename SDExpressionType>
        static void
        sd_substitute(
          const SDExpressionType &                            sd_expression,
          const Differentiation::SD::types::substitution_map &substitution_map)
        {
          return Differentiation::SD::substitute(sd_expression,
                                                 substitution_map);
        }

        template <typename... SDExpressionTypes>
        static void
        sd_substitute(
          std::tuple<SDExpressionTypes...> &                  sd_expressions,
          const Differentiation::SD::types::substitution_map &substitution_map)
        {
          unpack_sd_substitute<0, SDExpressionTypes...>(sd_expressions,
                                                        substitution_map);
        }

        template <typename SDNumberType, typename SDExpressionType>
        static first_derivatives_value_t<SDNumberType, SDExpressionType>
        sd_substitute_and_differentiate(
          const SDExpressionType &                            sd_expression,
          const Differentiation::SD::types::substitution_map &substitution_map,
          const field_values_t<SDNumberType> &symbolic_field_values)
        {
          if (substitution_map.size() > 0)
            {
              SDExpressionType sd_expression_subs{sd_expression};
              sd_substitute(sd_expression_subs, substitution_map);
              return sd_differentiate<SDNumberType>(sd_expression_subs,
                                                    symbolic_field_values);
            }
          else
            return sd_differentiate<SDNumberType>(sd_expression,
                                                  symbolic_field_values);
        }

        template <typename SDNumberType,
                  typename SDExpressionType,
                  typename BatchOptimizerType>
        static void
        sd_register_functions(
          BatchOptimizerType &batch_optimizer,
          const first_derivatives_value_t<SDNumberType, SDExpressionType>
            &derivatives)
        {
          return unpack_sd_register_functions<SDNumberType, SDExpressionType>(
            batch_optimizer,
            derivatives,
            std::make_index_sequence<std::tuple_size<
              first_derivatives_value_t<SDNumberType,
                                        SDExpressionType>>::value>());
        }

        template <typename SDNumberType,
                  typename SDExpressionType,
                  typename BatchOptimizerType>
        static void
        sd_register_functions(
          BatchOptimizerType &batch_optimizer,
          const second_derivatives_value_t<SDNumberType, SDExpressionType>
            &derivatives)
        {
          return unpack_sd_register_functions<SDNumberType, SDExpressionType>(
            batch_optimizer, derivatives);
        }

        template <typename SDNumberType,
                  typename ScalarType,
                  int dim,
                  int spacedim>
        static Differentiation::SD::types::substitution_map
        sd_get_substitution_map(
          MeshWorker::ScratchData<dim, spacedim> &scratch_data,
          const std::vector<std::string> &        solution_names,
          const unsigned int                      q_point,
          const field_values_t<SDNumberType> &    symbolic_field_values,
          const field_args_t &                    field_args)
        {
          static_assert(std::tuple_size<field_values_t<SDNumberType>>::value ==
                          std::tuple_size<field_args_t>::value,
                        "Size mismatch");

          Differentiation::SD::types::substitution_map substitution_map;

          unpack_sd_add_to_substitution_map<SDNumberType, ScalarType>(
            substitution_map,
            scratch_data,
            solution_names,
            q_point,
            symbolic_field_values,
            field_args);

          return substitution_map;
        }

      private:
        // ===================
        // AD helper functions
        // ===================

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

        // ===================
        // SD helper functions
        // ===================

        template <typename SDNumberType, std::size_t... I>
        static field_values_t<SDNumberType>
        unpack_get_symbolic_fields(const field_args_t &      field_args,
                                   const SymbolicDecorations decorator,
                                   const std::index_sequence<I...>)
        {
          return {internal::make_symbolic<SDNumberType>(std::get<I>(field_args),
                                                        decorator)...};
        }


        template <typename SDNumberType, std::size_t... I>
        static Differentiation::SD::types::substitution_map
        unpack_sd_get_symbol_map(
          const field_values_t<SDNumberType> &symbolic_field_values,
          const std::index_sequence<I...>)
        {
          return Differentiation::SD::make_symbol_map(
            std::get<I>(symbolic_field_values)...);
        }

        template <typename SDNumberType,
                  typename SDFunctionType,
                  std::size_t... I>
        static auto
        unpack_sd_call_function(
          const SDFunctionType &              sd_function,
          const field_values_t<SDNumberType> &symbolic_field_values,
          const std::index_sequence<I...>)
        {
          return sd_function(std::get<I>(symbolic_field_values)...);
        }

        // Expect SDSubstitutionFunctionType to be a std::function
        template <typename SDNumberType,
                  typename SDSubstitutionFunctionType,
                  std::size_t... I>
        static Differentiation::SD::types::substitution_map
        unpack_sd_call_substitution_function(
          const SDSubstitutionFunctionType &  substitution_function,
          const field_values_t<SDNumberType> &symbolic_field_values,
          const std::index_sequence<I...>)
        {
          Assert(substitution_function, ExcNotInitialized());
          return substitution_function(std::get<I>(symbolic_field_values)...);
        }

        template <typename SDNumberType,
                  typename SDExpressionType,
                  std::size_t... I>
        static first_derivatives_value_t<SDNumberType, SDExpressionType>
        unpack_sd_differentiate(
          const SDExpressionType &            sd_expression,
          const field_values_t<SDNumberType> &symbolic_field_values,
          const std::index_sequence<I...>)
        {
          return {Differentiation::SD::differentiate(
            sd_expression, std::get<I>(symbolic_field_values))...};
        }

        template <typename SDNumberType,
                  typename... SDExpressionTypes,
                  std::size_t... I,
                  std::size_t... J>
        static std::tuple<
          first_derivatives_value_t<SDNumberType, SDExpressionTypes>...>
        unpack_sd_differentiate(
          const std::tuple<SDExpressionTypes...> &sd_expressions,
          const field_values_t<SDNumberType> &    symbolic_field_values,
          const std::index_sequence<I...>,
          const std::index_sequence<J...> &seq_j)
        {
          // For a fixed row "I", expand all the derivatives of expression "I"
          // with respect to fields "J"
          return {unpack_sd_differentiate<SDNumberType>(
            std::get<I>(sd_expressions), symbolic_field_values, seq_j)...};
        }

        template <std::size_t I = 0, typename... SDExpressionTypes>
          static typename std::enable_if <
          I<sizeof...(SDExpressionTypes), void>::type
          unpack_sd_substitute(
            std::tuple<SDExpressionTypes...> &sd_expressions,
            const Differentiation::SD::types::substitution_map
              &substitution_map)
        {
          sd_substitute(std::get<I>(sd_expressions), substitution_map);
          unpack_sd_substitute<I + 1, SDExpressionTypes...>(sd_expressions,
                                                            substitution_map);
        }

        template <std::size_t I = 0, typename... SDExpressionTypes>
        static
          typename std::enable_if<I == sizeof...(SDExpressionTypes), void>::type
          unpack_sd_substitute(
            std::tuple<SDExpressionTypes...> &sd_expressions,
            const Differentiation::SD::types::substitution_map
              &substitution_map)
        {
          // Do nothing
          (void)sd_expressions;
          (void)substitution_map;
        }

        // Registration for first derivatives (stored in a single tuple)
        // Register a single expression
        template <typename /*SDNumberType*/,
                  typename /*SDExpressionType*/,
                  typename... SDExpressions,
                  typename BatchOptimizerType,
                  std::size_t... I>
        static typename std::enable_if<!is_tuple<SDExpressions...>::value &&
                                       (sizeof...(I) == 1)>::type
        unpack_sd_register_functions(
          BatchOptimizerType &                batch_optimizer,
          const std::tuple<SDExpressions...> &derivatives,
          const std::index_sequence<I...>)
        {
          batch_optimizer.register_function(std::get<I>(derivatives)...);
        }

        // Registration for first derivatives (stored in a single tuple)
        // Register multiple expressions simultaneously
        template <typename /*SDNumberType*/,
                  typename /*SDExpressionType*/,
                  typename... SDExpressions,
                  typename BatchOptimizerType,
                  std::size_t... I>
        static typename std::enable_if<!is_tuple<SDExpressions...>::value &&
                                       (sizeof...(I) > 1)>::type
        unpack_sd_register_functions(
          BatchOptimizerType &                batch_optimizer,
          const std::tuple<SDExpressions...> &derivatives,
          const std::index_sequence<I...>)
        {
          batch_optimizer.register_functions(std::get<I>(derivatives)...);
        }

        // Registration for higher-order derivatives
        template <typename SDNumberType,
                  typename SDExpressionType,
                  std::size_t I = 0,
                  typename BatchOptimizerType,
                  typename... Ts>
        static
          typename std::enable_if<is_tuple<Ts...>::value && (I < sizeof...(Ts)),
                                  void>::type
          unpack_sd_register_functions(
            BatchOptimizerType &     batch_optimizer,
            const std::tuple<Ts...> &higher_order_derivatives)
        {
          // Filter through the outer tuple and dispatch the work to the
          // other function (specialised for some first derivative types)
          sd_register_functions<SDNumberType, SDExpressionType>(
            batch_optimizer, std::get<I>(higher_order_derivatives));
          unpack_sd_register_functions<SDNumberType, SDExpressionType, I + 1>(
            batch_optimizer, higher_order_derivatives);
        }

        template <typename /*SDNumberType*/,
                  typename /*SDExpressionType*/,
                  std::size_t I = 0,
                  typename BatchOptimizerType,
                  typename... Ts>
        static typename std::
          enable_if<is_tuple<Ts...>::value && (I == sizeof...(Ts)), void>::type
          unpack_sd_register_functions(
            BatchOptimizerType &     batch_optimizer,
            const std::tuple<Ts...> &higher_order_derivatives)
        {
          // Do nothing
          (void)batch_optimizer;
          (void)higher_order_derivatives;
        }

        template <typename SDNumberType,
                  typename ScalarType,
                  std::size_t I = 0,
                  int         dim,
                  int         spacedim,
                  typename... UnaryOpType>
          static typename std::enable_if <
          I<sizeof...(UnaryOpType), void>::type
          unpack_sd_add_to_substitution_map(
            Differentiation::SD::types::substitution_map &substitution_map,
            MeshWorker::ScratchData<dim, spacedim> &      scratch_data,
            const std::vector<std::string> &              solution_names,
            const unsigned int                            q_point,
            const field_values_t<SDNumberType> &          symbolic_field_values,
            const std::tuple<UnaryOpType...> &unary_op_field_solutions)
        {
          static_assert(std::tuple_size<field_values_t<SDNumberType>>::value ==
                          std::tuple_size<std::tuple<UnaryOpType...>>::value,
                        "Size mismatch");

          // Get the field value
          const auto &unary_op_field_solution =
            std::get<I>(unary_op_field_solutions);
          const auto &                       field_solutions =
            unary_op_field_solution.template operator()<ScalarType>(
              scratch_data, solution_names); // Cached solution at all QPs
          Assert(q_point < field_solutions.size(),
                 ExcIndexRange(q_point, 0, field_solutions.size()));
          const auto &field_solution = field_solutions[q_point];

          // Get the symbol for the field
          const auto &symbolic_field_solution =
            std::get<I>(symbolic_field_values);

          // Append these to the substitution map, and recurse.
          Differentiation::SD::add_to_substitution_map(substitution_map,
                                                       symbolic_field_solution,
                                                       field_solution);
          unpack_sd_add_to_substitution_map<SDNumberType, ScalarType, I + 1>(
            substitution_map,
            scratch_data,
            solution_names,
            q_point,
            symbolic_field_values,
            unary_op_field_solutions);
        }

        template <typename SDNumberType,
                  typename ScalarType,
                  std::size_t I = 0,
                  int         dim,
                  int         spacedim,
                  typename... UnaryOpType>
        static typename std::enable_if<I == sizeof...(UnaryOpType), void>::type
        unpack_sd_add_to_substitution_map(
          Differentiation::SD::types::substitution_map &substitution_map,
          MeshWorker::ScratchData<dim, spacedim> &      scratch_data,
          const std::vector<std::string> &              solution_names,
          const unsigned int                            q_point,
          const field_values_t<SDNumberType> &          symbolic_field_values,
          const std::tuple<UnaryOpType...> &unary_op_field_solutions)
        {
          // Do nothing
          (void)substitution_map;
          (void)scratch_data;
          (void)solution_names;
          (void)q_point;
          (void)symbolic_field_values;
          (void)unary_op_field_solutions;
        }

      }; // struct UnaryOpsSubSpaceFieldSolutionHelper

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
    template <typename ADorSDNumberType>
    using value_type = ADorSDNumberType;

    template <typename ScalarType,
              enum Differentiation::AD::NumberTypes ADNumberTypeCode>
    using ad_type =
      typename Differentiation::AD::NumberTraits<ScalarType,
                                                 ADNumberTypeCode>::ad_type;

    template <typename ADNumberType, int dim, int spacedim = dim>
    using ad_function_type = std::function<value_type<ADNumberType>(
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &              solution_names,
      const unsigned int                            q_point,
      const typename UnaryOpsSubSpaceFieldSolution::template value_type<
        ADNumberType> &... field_solutions)>;

    template <typename ScalarType>
    using sd_type               = Differentiation::SD::Expression;
    using substitution_map_type = Differentiation::SD::types::substitution_map;

    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_function_type = std::function<value_type<SDNumberType>(
      const typename UnaryOpsSubSpaceFieldSolution::template value_type<
        SDNumberType> &... field_solutions)>;

    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_intermediate_substitution_function_type =
      std::function<substitution_map_type(
        const typename UnaryOpsSubSpaceFieldSolution::template value_type<
          SDNumberType> &... field_solutions)>;

    // This also allows the user to encode symbols/parameters in terms of
    // the (symbolic) field variables, for which we'll supply the values.
    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_register_symbols_function_type =
      std::function<substitution_map_type(
        const typename UnaryOpsSubSpaceFieldSolution::template value_type<
          SDNumberType> &... field_solutions)>;

    template <typename SDNumberType, int dim, int spacedim = dim>
    using sd_substitution_function_type = std::function<substitution_map_type(
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &              solution_names,
      const unsigned int                            q_point)>;


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
    operator()(const ad_function_type<ADNumberType, dim, spacedim> &function,
               const UpdateFlags update_flags) const;

    template <typename ADNumberType, int dim, int spacedim = dim>
    auto
    operator()(
      const ad_function_type<ADNumberType, dim, spacedim> &function) const
    {
      return this->operator()<ADNumberType, dim, spacedim>(
        function, UpdateFlags::update_default);
    }

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    operator()(
      const sd_function_type<SDNumberType, dim, spacedim> &function,
      const enum Differentiation::SD::OptimizerType        optimization_method,
      const enum Differentiation::SD::OptimizationFlags    optimization_flags,
      const UpdateFlags                                    update_flags) const;

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    operator()(
      const sd_function_type<SDNumberType, dim, spacedim> &function,
      const enum Differentiation::SD::OptimizerType        optimization_method,
      const enum Differentiation::SD::OptimizationFlags    optimization_flags)
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
    value(const ad_function_type<ADNumberType, dim, spacedim> &function,
          const UpdateFlags update_flags) const
    {
      return this->operator()<ADNumberType, dim, spacedim>(function,
                                                           update_flags);
    }

    template <typename ADNumberType, int dim, int spacedim = dim>
    auto
    value(const ad_function_type<ADNumberType, dim, spacedim> &function) const
    {
      return this->operator()<ADNumberType, dim, spacedim>(function);
    }

    template <typename SDNumberType, int dim, int spacedim = dim>
    auto
    value(const sd_function_type<SDNumberType, dim, spacedim> &function,
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
    value(const sd_function_type<SDNumberType, dim, spacedim> &function,
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
        typename Op::template ad_function_type<ResultScalarType, dim, spacedim>;

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
      get_ad_helper(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache.get_object_with_name<ad_helper_type>(get_name_ad_helper());
      }

      template <std::size_t FieldIndex, typename UnaryOpField>
      typename UnaryOpField::extractor_type
      get_derivative_extractor(const UnaryOpField &) const
      {
        static_assert(FieldIndex < OpHelper_t::n_operators(),
                      "Index out of bounds.");
        return std::get<FieldIndex>(get_field_extractors());

        // TODO: Remove obsolete implementation in OpHelper_t
        // return OpHelper_t::get_initialized_extractor(field,
        // get_field_args());
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

        // Note: All user functions have the same parameterization, so we can
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
            Dpsi.resize(fe_values.n_quadrature_points,
                        Vector<scalar_type>(ad_helper.n_dependent_variables()));
            D2psi.resize(
              fe_values.n_quadrature_points,
              FullMatrix<scalar_type>(ad_helper.n_dependent_variables(),
                                      ad_helper.n_independent_variables()));
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
      using substitution_map_type =
        Differentiation::SD::types::substitution_map;

      using energy_type = sd_type;

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template sd_function_type<ResultScalarType, dim, spacedim>;

      template <typename ResultScalarType>
      using return_type = void;

      using sd_function_type = function_type<sd_type>;
      using sd_intermediate_substitution_function_type =
        typename Op::template sd_intermediate_substitution_function_type<
          sd_type,
          dim,
          spacedim>;
      // TODO: If this needs a template <typename ResultScalarType> then the
      // entire unary op must get one.
      using sd_register_symbols_function_type = typename Op::
        template sd_register_symbols_function_type<sd_type, dim, spacedim>;
      using sd_substitution_function_type = typename Op::
        template sd_substitution_function_type<sd_type, dim, spacedim>;
      ;

      static const int rank = 0;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(
        const Op &                               operand,
        const sd_function_type &                 function,
        const sd_register_symbols_function_type &user_symbol_registration_map,
        const sd_substitution_function_type &    user_substitution_map,
        const sd_intermediate_substitution_function_type
          &user_intermediate_substitution_map,
        const enum Differentiation::SD::OptimizerType     optimization_method,
        const enum Differentiation::SD::OptimizationFlags optimization_flags,
        const UpdateFlags                                 update_flags)
        : operand(operand)
        , function(function)
        , user_symbol_registration_map(user_symbol_registration_map)
        , user_substitution_map(user_substitution_map)
        , user_intermediate_substitution_map(user_intermediate_substitution_map)
        , optimization_method(optimization_method)
        , optimization_flags(optimization_flags)
        , update_flags(update_flags)
        , symbolic_fields(OpHelper_t::template get_symbolic_fields<sd_type>(
            get_field_args(),
            SymbolicDecorations()))
        , psi(OpHelper_t::template sd_call_function<sd_type>(function,
                                                             symbolic_fields))
        , first_derivatives(
            OpHelper_t::template sd_differentiate<sd_type>(psi,
                                                           symbolic_fields))
        , second_derivatives(
            OpHelper_t::template sd_substitute_and_differentiate<sd_type>(
              first_derivatives,
              OpHelper_t::template sd_call_substitution_function<sd_type>(
                user_intermediate_substitution_map,
                symbolic_fields),
              symbolic_fields))
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
      get_batch_optimizer(
        const MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        const GeneralDataStorage &cache =
          scratch_data.get_general_data_storage();

        return cache.get_object_with_name<sd_helper_type<ResultScalarType>>(
          get_name_sd_batch_optimizer());
      }

      template <std::size_t FieldIndex>
      const auto &
      get_symbolic_first_derivative() const
      {
        static_assert(FieldIndex < OpHelper_t::n_operators(),
                      "Index out of bounds.");
        return std::get<FieldIndex>(first_derivatives);
      }

      template <std::size_t FieldIndex_1, std::size_t FieldIndex_2>
      const auto &
      get_symbolic_second_derivative() const
      {
        static_assert(FieldIndex_1 < OpHelper_t::n_operators(),
                      "Row index out of bounds.");
        static_assert(FieldIndex_2 < OpHelper_t::n_operators(),
                      "Column index out of bounds.");
        // Get the row tuple, then the column entry in that row tuple.
        return std::get<FieldIndex_2>(
          std::get<FieldIndex_1>(second_derivatives));
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
        // Follow the recipe described in the documentation:
        // - Define some independent variables.
        // - Compute symbolic expressions that are dependent on the independent
        //   variables.
        // - Create a optimizer to evaluate the dependent functions.
        // - Register symbols that represent independent variables.
        // - Register symbolic expressions that represent dependent functions.
        // - Optimize: Determine computationally efficient code path for
        //   evaluation.
        // - Substitute: Pass the optimizer the numeric values that thee
        //   independent variables to represent.
        // - Extract the numeric equivalent of the dependent functions from the
        //   optimizer.

        // Note: All user functions have the same parameterization, so on the
        // face of it we can use the same BatchOptimizer for each of them. In
        // theory the user can encode the QPoint into the energy function: this
        // current implementation restricts the user to use the same definition
        // for the energy itself at each QP.
        sd_helper_type<ResultScalarType> &batch_optimizer =
          get_mutable_sd_batch_optimizer<ResultScalarType>(scratch_data);
        if (batch_optimizer.optimized() == false)
          {
            Assert(batch_optimizer.n_independent_variables() == 0,
                   ExcMessage(
                     "Expected the batch optimizer to be uninitialized."));
            Assert(batch_optimizer.n_dependent_variables() == 0,
                   ExcMessage(
                     "Expected the batch optimizer to be uninitialized."));
            Assert(batch_optimizer.values_substituted() == false,
                   ExcMessage(
                     "Expected the batch optimizer to be uninitialized."));

            // Create and register field variables (the independent variables).
            // We deal with the fields before the user data just in case
            // the users try to overwrite these field symbols. It shouldn't
            // happen, but this way its not possible to do overwrite what's
            // already in the map.
            Differentiation::SD::types::substitution_map symbol_map =
              OpHelper_t::template sd_get_symbol_map<sd_type>(
                get_symbolic_fields());
            if (user_symbol_registration_map)
              {
                Differentiation::SD::add_to_symbol_map(
                  symbol_map,
                  OpHelper_t::template sd_call_function<sd_type>(
                    user_symbol_registration_map, get_symbolic_fields()));
              }
            batch_optimizer.register_symbols(symbol_map);

            // The next typical few steps that precede function resistration
            // have already been performed in the class constructor:
            // - Evaluate the functor to compute the total stored energy.
            // - Compute the first derivatives of the energy function.
            // - If there's some intermediate substitution to be done (modifying
            // the first derivatives), then do it before computing the second
            // derivatives.
            // (Why the intermediate substitution? If the first derivatives
            // represent the partial derivatives, then this substitution may be
            // done to ensure that the consistent linearization is given by the
            // second derivatives.)
            // - Differentiate the first derivatives (perhaps a modified form)
            // to get the second derivatives.

            // Register the dependent variables.
            OpHelper_t::template sd_register_functions<sd_type, energy_type>(
              batch_optimizer, first_derivatives);
            OpHelper_t::template sd_register_functions<sd_type, energy_type>(
              batch_optimizer, second_derivatives);

            // Finalize the optimizer.
            batch_optimizer.optimize();
          }

        // Check that we've actually got a state that we can do some work with.
        Assert(batch_optimizer.n_independent_variables() > 0,
               ExcMessage("Expected the batch optimizer to be initialized."));
        Assert(batch_optimizer.n_dependent_variables() > 0,
               ExcMessage("Expected the batch optimizer to be initialized."));

        std::vector<std::vector<ResultScalarType>>
          &evaluated_dependent_functions =
            get_mutable_evaluated_dependent_functions<ResultScalarType>(
              scratch_data, batch_optimizer);

        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        // In the HP case, we might traverse between cells with a different
        // number of quadrature points. So we need to resize the output data
        // accordingly.
        if (evaluated_dependent_functions.size() !=
            fe_values.n_quadrature_points)
          {
            evaluated_dependent_functions.resize(
              fe_values.n_quadrature_points,
              std::vector<ResultScalarType>(
                batch_optimizer.n_dependent_variables()));
          }

        for (const auto &q_point : fe_values.quadrature_point_indices())
          {
            // Substitute the field variables and whatever user symbols
            // are defined.
            // First we do the values from finite element fields,
            // followed by the values for user parameters, etc.
            Differentiation::SD::types::substitution_map substitution_map =
              OpHelper_t::template sd_get_substitution_map<sd_type,
                                                           ResultScalarType>(
                scratch_data,
                solution_names,
                q_point,
                get_symbolic_fields(),
                get_field_args());
            if (user_substitution_map)
              {
                Differentiation::SD::add_to_substitution_map(
                  substitution_map,
                  user_substitution_map(scratch_data, solution_names, q_point));
              }

            // Perform the value substitution at this quadrature point
            batch_optimizer.substitute(substitution_map);

            // Extract evaluated data to be retrieved later.
            evaluated_dependent_functions[q_point] = batch_optimizer.evaluate();
          }
      }

    private:
      const Op               operand;
      const sd_function_type function;

      const sd_register_symbols_function_type user_symbol_registration_map;
      const sd_substitution_function_type     user_substitution_map;
      const sd_intermediate_substitution_function_type
        user_intermediate_substitution_map;

      const enum Differentiation::SD::OptimizerType     optimization_method;
      const enum Differentiation::SD::OptimizationFlags optimization_flags;

      // Some additional update flags that the user might require in order to
      // evaluate their SD function (e.g. UpdateFlags::update_quadrature_points)
      const UpdateFlags update_flags;

      // Independent variables
      const typename OpHelper_t::template field_values_t<sd_type>
        symbolic_fields;

      // Dependent variables
      const energy_type psi; // The energy
      const typename OpHelper_t::template first_derivatives_value_t<sd_type,
                                                                    energy_type>
        first_derivatives;
      const typename OpHelper_t::
        template second_derivatives_value_t<sd_type, energy_type>
          second_derivatives;

      // const typename OpHelper_t::field_extractors_t
      //   extractors; // FEValuesExtractors to work with multi-component fields

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

      // template <typename UnaryOpField>
      // typename UnaryOpField::template value_type<sd_type>
      // get_symbolic_field(const UnaryOpField &field) const
      // {
      //   const SymbolicDecorations decorator;
      //   return internal::make_symbolic<sd_type>(field, decorator);
      // }

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

      template <typename ResultScalarType>
      sd_helper_type<ResultScalarType> &
      get_mutable_sd_batch_optimizer(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data) const
      {
        GeneralDataStorage &cache = scratch_data.get_general_data_storage();
        const std::string   name_sd_batch_optimizer =
          get_name_sd_batch_optimizer();

        // Unfortunately we cannot perform a check like this because the
        // ScratchData is reused by many cells during the mesh loop. So
        // there's no real way to verify that the user is not accidentally
        // re-using an object because they forget to uniquely name the
        // EnergyFunctor upon which this op is based.
        //
        // Assert(!(cache.stores_object_with_name(name_ad_helper)),
        //        ExcMessage("ADHelper is already present in the cache."));

        return cache
          .get_or_add_object_with_name<sd_helper_type<ResultScalarType>>(
            name_sd_batch_optimizer, optimization_method, optimization_flags);
      }

      template <typename ResultScalarType>
      std::vector<std::vector<ResultScalarType>> &
      get_mutable_evaluated_dependent_functions(
        MeshWorker::ScratchData<dim, spacedim> &scratch_data,
        const sd_helper_type<ResultScalarType> &batch_optimizer) const
      {
        GeneralDataStorage &cache = scratch_data.get_general_data_storage();
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();

        return cache.get_or_add_object_with_name<
          std::vector<std::vector<ResultScalarType>>>(
          get_name_evaluated_dependent_functions(),
          fe_values.n_quadrature_points,
          std::vector<ResultScalarType>(
            batch_optimizer.n_dependent_variables()));
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

      const typename OpHelper_t::template field_values_t<sd_type> &
      get_symbolic_fields() const
      {
        return symbolic_fields;
      }
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
      template ad_function_type<ADNumberType, dim, spacedim> &function,
    const UpdateFlags                                         update_flags)
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
      template ad_function_type<ADNumberType, dim, spacedim> &function,
    const enum Differentiation::SD::OptimizerType     optimization_method,
    const enum Differentiation::SD::OptimizationFlags optimization_flags)
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
      template sd_function_type<SDNumberType, dim, spacedim> &function,
    const enum Differentiation::SD::OptimizerType     optimization_method,
    const enum Differentiation::SD::OptimizationFlags optimization_flags,
    const UpdateFlags                                 update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           void,
                           SDNumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;


    const typename WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::
      template sd_register_symbols_function_type<SDNumberType, dim, spacedim>
        dummy_symb_reg_map;

    const typename WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::
      template sd_substitution_function_type<SDNumberType, dim, spacedim>
        dummy_subs_map;

    const typename WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::
      template sd_intermediate_substitution_function_type<SDNumberType,
                                                          dim,
                                                          spacedim>
        dummy_intermediate_subs_map;

    return OpType(operand,
                  function,
                  dummy_symb_reg_map,
                  dummy_subs_map,
                  dummy_intermediate_subs_map,
                  optimization_method,
                  optimization_flags,
                  update_flags);
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
      template sd_function_type<SDNumberType, dim, spacedim> &function)
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
      template ad_function_type<ADNumberType, dim, spacedim> &function,
    const UpdateFlags update_flags) const
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
      template sd_function_type<SDNumberType, dim, spacedim> &function,
    const enum Differentiation::SD::OptimizerType     optimization_method,
    const enum Differentiation::SD::OptimizationFlags optimization_flags,
    const UpdateFlags                                 update_flags) const
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
