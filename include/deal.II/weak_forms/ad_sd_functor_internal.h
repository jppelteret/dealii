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

#ifndef dealii_weakforms_ad_sd_functor_internal_h
#define dealii_weakforms_ad_sd_functor_internal_h

#include <deal.II/base/config.h>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/differentiation/ad.h>
#include <deal.II/differentiation/sd.h>

#include <deal.II/weak_forms/differentiation.h>

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


DEAL_II_NAMESPACE_OPEN


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


      // ===================
      // SD helper functions
      // ===================
      std::string
      get_deal_II_prefix()
      {
        return "__DEAL_II__";
      }

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

        const std::string name =
          get_deal_II_prefix() + "Field_" + field.as_ascii(decorator);
        return make_symbolic<ReturnType>(name);
      }

      // Check that all types in a parameter pack are not tuples
      // without using C++17 fold expressions...
      // https://stackoverflow.com/a/29671981
      // https://stackoverflow.com/a/29603896
      // https://stackoverflow.com/a/32234520

      template <typename>
      struct is_tuple : std::false_type
      {};

      template <typename... T>
      struct is_tuple<std::tuple<T...>> : std::true_type
      {};

      template <bool...>
      struct bool_pack;
      template <bool... bs>
      using all_true =
        std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;
      template <bool... bs>
      using all_false =
        std::is_same<bool_pack<bs..., false>, bool_pack<false, bs...>>;
      template <typename... Ts>
      using are_tuples = all_true<is_tuple<Ts>::value...>;
      template <typename... Ts>
      using are_not_tuples = all_false<is_tuple<Ts>::value...>;

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
          return unpack_sd_register_1st_order_functions<SDNumberType,
                                                        SDExpressionType>(
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
          return unpack_sd_register_2nd_order_functions<SDNumberType,
                                                        SDExpressionType>(
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
        static
          typename std::enable_if<(are_not_tuples<SDExpressions...>::value) &&
                                  (sizeof...(I) == 1)>::type
          unpack_sd_register_1st_order_functions(
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
        static
          typename std::enable_if<(are_not_tuples<SDExpressions...>::value) &&
                                  (sizeof...(I) > 1)>::type
          unpack_sd_register_1st_order_functions(
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
        static typename std::enable_if<(I < sizeof...(Ts)), void>::type
        unpack_sd_register_2nd_order_functions(
          BatchOptimizerType &     batch_optimizer,
          const std::tuple<Ts...> &higher_order_derivatives)
        {
          static_assert(are_tuples<Ts...>::value,
                        "Expected all inner objects to be tuples");

          // Filter through the outer tuple and dispatch the work to the
          // other function (specialized for some first derivative types).
          // Note: A recursive call to sd_register_functions(), in the hopes
          // that it would detect std::get<I>(higher_order_derivatives) as a
          // lower-order derivative, does not seem to work. It, for some reason,
          // calls the higher-order variant again and sends us into an infinite
          // loop. So don't do that!
          using InnerTupleType = typename std::decay<decltype(
            std::get<I>(higher_order_derivatives))>::type;
          unpack_sd_register_1st_order_functions<SDNumberType,
                                                 SDExpressionType>(
            batch_optimizer,
            std::get<I>(higher_order_derivatives),
            std::make_index_sequence<std::tuple_size<InnerTupleType>::value>());
          unpack_sd_register_2nd_order_functions<SDNumberType,
                                                 SDExpressionType,
                                                 I + 1>(
            batch_optimizer, higher_order_derivatives);
        }


        template <typename /*SDNumberType*/,
                  typename /*SDExpressionType*/,
                  std::size_t I = 0,
                  typename BatchOptimizerType,
                  typename... Ts>
        static typename std::enable_if<(I == sizeof...(Ts)), void>::type
        unpack_sd_register_2nd_order_functions(
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


DEAL_II_NAMESPACE_CLOSE


#endif // dealii_weakforms_ad_sd_functor_internal_h
