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

#ifndef dealii_weakforms_self_linearizing_forms_h
#define dealii_weakforms_self_linearizing_forms_h

#include <deal.II/base/config.h>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/tensor.h>

#include <boost/core/demangle.hpp>

#include <deal.II/weak_forms/subspace_extractors.h>
#include <deal.II/weak_forms/subspace_views.h>
#include <deal.II/weak_forms/type_traits.h>
#include <deal.II/weak_forms/unary_operators.h>

#include <string>
#include <typeinfo>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace SelfLinearization
  {
    namespace internal
    {
      // Make the link between FEValuesExtractors and the weak form
      // SubSpaceExtractors
      template <typename FEValuesExtractors_t>
      struct SubSpaceExtractor;


      template <>
      struct SubSpaceExtractor<FEValuesExtractors::Scalar>
      {
        using type = WeakForms::SubSpaceExtractors::Scalar;
      };


      template <>
      struct SubSpaceExtractor<FEValuesExtractors::Vector>
      {
        using type = WeakForms::SubSpaceExtractors::Vector;
      };


      template <int rank>
      struct SubSpaceExtractor<FEValuesExtractors::Tensor<rank>>
      {
        using type = WeakForms::SubSpaceExtractors::Tensor<rank>;
      };


      template <int rank>
      struct SubSpaceExtractor<FEValuesExtractors::SymmetricTensor<rank>>
      {
        using type = WeakForms::SubSpaceExtractors::SymmetricTensor<rank>;
      };


      // Convert field solutions to a test function or trial solution.
      // This is required because we'll probe the arguments for the
      // functor supplied to the self-linearizing form, and from these
      // we'll have to construct the relevant residual (linear) and
      // linearization (bilinear) forms.
      struct ConvertTo
      {
        // For SubSpaceViews::Scalar and SubSpaceViews::Vector
        template <template <class> typename SubSpaceViewsType,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode,
                  typename = typename std::enable_if<
                    is_field_solution<SpaceType>::value>::type>
        static auto
        test_function(
          const WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                              OpCode> &unary_op)
        {
          using SubSpaceViewFieldSolution_t = SubSpaceViewsType<SpaceType>;
          using UnaryOp_t =
            WeakForms::Operators::UnaryOp<SubSpaceViewFieldSolution_t, OpCode>;
          constexpr unsigned int dim      = UnaryOp_t::dimension;
          constexpr unsigned int spacedim = UnaryOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TestFunction<dim, spacedim>;
          using Op      = SubSpaceViewsType<Space_t>;
          using OpType  = WeakForms::Operators::UnaryOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = unary_op.get_operand();
          const SubSpaceExtractor_t extractor(
            field_solution_ss_op.get_extractor(),
            field_solution_ss_op.get_space().get_field_ascii_raw(),
            field_solution_ss_op.get_space().get_field_latex_raw());
          // And now apply it to the (sub)space that we wish convert to
          const Space_t space;
          const Op      operand(space[extractor]);
          return OpType(operand);
        }


        // For SubSpaceViews::Scalar and SubSpaceViews::Vector
        template <template <class> typename SubSpaceViewsType,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode,
                  typename = typename std::enable_if<
                    is_field_solution<SpaceType>::value>::type>
        static auto
        trial_solution(
          const WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                              OpCode> &unary_op)
        {
          using SubSpaceViewFieldSolution_t = SubSpaceViewsType<SpaceType>;
          using UnaryOp_t =
            WeakForms::Operators::UnaryOp<SubSpaceViewFieldSolution_t, OpCode>;
          constexpr unsigned int dim      = UnaryOp_t::dimension;
          constexpr unsigned int spacedim = UnaryOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TrialSolution<dim, spacedim>;
          using Op      = SubSpaceViewsType<Space_t>;
          using OpType  = WeakForms::Operators::UnaryOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = unary_op.get_operand();
          const SubSpaceExtractor_t extractor(
            field_solution_ss_op.get_extractor(),
            field_solution_ss_op.get_space().get_field_ascii_raw(),
            field_solution_ss_op.get_space().get_field_latex_raw());
          // And now apply it to the (sub)space that we wish convert to
          const Space_t space;
          const Op      operand(space[extractor]);
          return OpType(operand);
        }


        // For SubSpaceViews::Tensor and SubSpaceViews::SymmetricTensor
        template <template <int, class> typename SubSpaceViewsType,
                  int rank,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode,
                  typename = typename std::enable_if<
                    is_field_solution<SpaceType>::value>::type>
        static auto
        test_function(const WeakForms::Operators::UnaryOp<
                      SubSpaceViewsType<rank, SpaceType>,
                      OpCode> &unary_op)
        {
          using SubSpaceViewFieldSolution_t =
            SubSpaceViewsType<rank, SpaceType>;
          using UnaryOp_t =
            WeakForms::Operators::UnaryOp<SubSpaceViewFieldSolution_t, OpCode>;
          constexpr unsigned int dim      = UnaryOp_t::dimension;
          constexpr unsigned int spacedim = UnaryOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TestFunction<dim, spacedim>;
          using Op      = SubSpaceViewsType<rank, Space_t>;
          using OpType  = WeakForms::Operators::UnaryOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = unary_op.get_operand();
          const SubSpaceExtractor_t extractor(
            field_solution_ss_op.get_extractor(),
            field_solution_ss_op.get_space().get_field_ascii_raw(),
            field_solution_ss_op.get_space().get_field_latex_raw());
          // And now apply it to the (sub)space that we wish convert to
          const Space_t space;
          const Op      operand(space[extractor]);
          return OpType(operand);
        }


        // For SubSpaceViews::Tensor and SubSpaceViews::SymmetricTensor
        template <template <int, class> typename SubSpaceViewsType,
                  int rank,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode,
                  typename = typename std::enable_if<
                    is_field_solution<SpaceType>::value>::type>
        static auto
        trial_solution(const WeakForms::Operators::UnaryOp<
                       SubSpaceViewsType<rank, SpaceType>,
                       OpCode> &unary_op)
        {
          using SubSpaceViewFieldSolution_t =
            SubSpaceViewsType<rank, SpaceType>;
          using UnaryOp_t =
            WeakForms::Operators::UnaryOp<SubSpaceViewFieldSolution_t, OpCode>;
          constexpr unsigned int dim      = UnaryOp_t::dimension;
          constexpr unsigned int spacedim = UnaryOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TrialSolution<dim, spacedim>;
          using Op      = SubSpaceViewsType<rank, Space_t>;
          using OpType  = WeakForms::Operators::UnaryOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = unary_op.get_operand();
          const SubSpaceExtractor_t extractor(
            field_solution_ss_op.get_extractor(),
            field_solution_ss_op.get_space().get_field_ascii_raw(),
            field_solution_ss_op.get_space().get_field_latex_raw());
          // And now apply it to the (sub)space that we wish convert to
          const Space_t space;
          const Op      operand(space[extractor]);
          return OpType(operand);
        }


        // Each @p UnaryOpSubSpaceFieldSolution is expected to
        // be a
        // Operators::UnaryOp<SubSpaceViews::[Scalar/Vector/Tensor/SymmetricTensor]>>
        // Since we can't convert the underlying SubSpaceViewsType (its a fixed
        // FieldSolution) we just ask for what the expected return values of the
        // above helper functions would be.
        template <typename UnaryOpSubSpaceFieldSolution>
        using test_function_t =
          decltype(test_function(std::declval<UnaryOpSubSpaceFieldSolution>()));

        template <typename UnaryOpSubSpaceFieldSolution>
        using trial_solution_t = decltype(
          trial_solution(std::declval<UnaryOpSubSpaceFieldSolution>()));
      };


      namespace Utilities
      {
        // Something to store types of a parameter pack
        // in (instead of a Tuple)
        template <typename... Ts>
        struct TypeList
        {};

        // Something to pair up two types together
        template <typename T1, typename T2>
        struct TypePair
        {};

        // // Something to collect three types together
        // template <typename T1, typename T2, typename T3>
        // struct TypeTriple
        // {};


        // Concatenation of type lists
        template <typename... T>
        struct Concatenate;


        // // Single itemed list
        // template <typename T>
        // struct Concatenate<TypeList<T>>
        // {
        //   using type = TypeList<T>;
        // };


        template <typename... Ts, typename... Us>
        struct Concatenate<TypeList<Ts...>, TypeList<Us...>>
        {
          using type = TypeList<Ts..., Us...>;
        };


        // Print scalar, Tensor or SymmetricTensor types
        template <typename T>
        struct TypeHelper
        {
          using type = T;

          static std::string
          print()
          {
            return boost::core::demangle(typeid(T).name());
          }
        };


        // Print sub-space view:
        // Scalar or Vector sub-space
        template <template <class> typename SubSpaceViewsType,
                  typename SpaceType>
        struct TypeHelper<SubSpaceViewsType<SpaceType>>
        {
          using type = SubSpaceViewsType<SpaceType>;

          static std::string
          print()
          {
            std::string space_type = "";
            if (is_test_function<SpaceType>::value)
              space_type = "TestFunction";
            else if (is_trial_solution<SpaceType>::value)
              space_type = "TrialSolution";
            else if (is_field_solution<SpaceType>::value)
              space_type = "FieldSolution";
            else
              {
                AssertThrow(false, ExcMessage("Unknown space type."));
              }
            std::string view_type = "";
            if (std::is_same<type, SubSpaceViews::Scalar<SpaceType>>::value)
              view_type = "s";
            else if (std::is_same<type,
                                  SubSpaceViews::Vector<SpaceType>>::value)
              view_type = "V";
            else
              {
                AssertThrow(false, ExcMessage("Unknown view type."));
              }

            return space_type + "[" + view_type + "]";
          }
        };

        // Print sub-space view:
        // Tensor or SymmetricTensor sub-space
        template <template <int, class> typename SubSpaceViewsType,
                  int rank,
                  typename SpaceType>
        struct TypeHelper<SubSpaceViewsType<rank, SpaceType>>
        {
          using type = SubSpaceViewsType<rank, SpaceType>;

          static std::string
          print()
          {
            std::string space_type = "";
            if (is_test_function<SpaceType>::value)
              space_type = "TestFunction";
            else if (is_trial_solution<SpaceType>::value)
              space_type = "TrialSolution";
            else if (is_field_solution<SpaceType>::value)
              space_type = "FieldSolution";
            else
              {
                AssertThrow(false, ExcMessage("Unknown space type."));
              }

            std::string view_type = "";
            if (std::is_same<type,
                             SubSpaceViews::Tensor<rank, SpaceType>>::value)
              view_type = "T";
            else if (std::is_same<
                       type,
                       SubSpaceViews::SymmetricTensor<rank, SpaceType>>::value)
              view_type = "S";
            else
              {
                AssertThrow(false, ExcMessage("Unknown view type."));
              }

            return space_type + "[" + view_type + "]";
          }
        };


        // Print unary operation:
        // Scalar or Vector sub-space
        template <template <class> typename SubSpaceViewsType,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode>
        struct TypeHelper<
          WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>, OpCode>>
        {
          using type =
            WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>, OpCode>;

          static std::string
          print()
          {
            std::string op_code = "";
            switch (OpCode)
              {
                case WeakForms::Operators::UnaryOpCodes::value:
                  break;
                case WeakForms::Operators::UnaryOpCodes::gradient:
                  op_code = "Grad";
                  break;
                case WeakForms::Operators::UnaryOpCodes::symmetric_gradient:
                  op_code = "symm_Grad";
                  break;
                case WeakForms::Operators::UnaryOpCodes::divergence:
                  op_code = "Div";
                  break;
                case WeakForms::Operators::UnaryOpCodes::curl:
                  op_code = "Curl";
                  break;
                case WeakForms::Operators::UnaryOpCodes::laplacian:
                  op_code = "Lap";
                  break;
                case WeakForms::Operators::UnaryOpCodes::hessian:
                  op_code = "Hess";
                  break;
                case WeakForms::Operators::UnaryOpCodes::third_derivative:
                  op_code = "D3";
                  break;
                default:
                  AssertThrow(false, ExcMessage("Unknown unary op code")) break;
              }

            if (OpCode == WeakForms::Operators::UnaryOpCodes::value)
              return TypeHelper<SubSpaceViewsType<SpaceType>>::print();
            else

              return op_code + "(" +
                     TypeHelper<SubSpaceViewsType<SpaceType>>::print() + ")";
          }
        };


        // Print unary operation:
        // Tensor or SymmetricTensor sub-space
        template <template <int, class> typename SubSpaceViewsType,
                  int rank,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode>
        struct TypeHelper<WeakForms::Operators::
                            UnaryOp<SubSpaceViewsType<rank, SpaceType>, OpCode>>
        {
          using type =
            WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank, SpaceType>,
                                          OpCode>;

          static std::string
          print()
          {
            std::string op_code = "";
            switch (OpCode)
              {
                case WeakForms::Operators::UnaryOpCodes::value:
                  break;
                case WeakForms::Operators::UnaryOpCodes::gradient:
                  op_code = "Grad";
                  break;
                case WeakForms::Operators::UnaryOpCodes::divergence:
                  op_code = "Div";
                  break;
                default:
                  AssertThrow(false, ExcMessage("Unknown unary op code")) break;
              }

            if (OpCode == WeakForms::Operators::UnaryOpCodes::value)
              return TypeHelper<SubSpaceViewsType<rank, SpaceType>>::print();
            else

              return op_code + "(" +
                     TypeHelper<SubSpaceViewsType<rank, SpaceType>>::print() +
                     ")";
          }
        };


        // Print TypePair<T, U> types
        template <typename T, typename U>
        struct TypeHelper<TypePair<T, U>>
        {
          using type = TypePair<T, U>;

          static std::string
          print()
          {
            return "(" + TypeHelper<T>::print() + ", " +
                   TypeHelper<U>::print() + ")";
          }
        };


        // // Print TypeTriple<T, U, V> types
        // template <typename T, typename U, typename V>
        // struct TypeHelper<TypeTriple<T, U, V>>
        // {
        //   using type = TypeTriple<T, U, V>;
        //
        //   static std::string
        //   print()
        //   {
        //     return "(" + TypeHelper<T>::print() + ", " +
        //     TypeHelper<U>::print() + ", " + TypeHelper<V>::print() + ")";
        //   }
        // };


        // Print empty TypeList<>
        template <>
        struct TypeHelper<TypeList<>>
        {
          using type = TypeList<>;

          static std::string
          print()
          {
            return "0";
          }
        };


        template <typename T>
        struct TypeHelper<TypeList<T>>
        {
          using type = TypeList<T>;

          static std::string
          print()
          {
            return "{" + TypeHelper<T>::print() + "}";
          }

          static std::string
          print(const std::string &sep)
          {
            return sep + TypeHelper<T>::print();
          }
        };


        template <typename T, typename... Ts>
        struct TypeHelper<TypeList<T, Ts...>>
        {
          using type = TypeList<T, Ts...>;

          static std::string
          print()
          {
            return "{" + TypeHelper<T>::print() +
                   TypeHelper<TypeList<Ts...>>::print(std::string(", ")) + "}";
          }

          static std::string
          print(const std::string &sep)
          {
            return sep + TypeHelper<T>::print() +
                   TypeHelper<TypeList<Ts...>>::print(sep);
          }
        };

      } // namespace Utilities


      // Cartesian product of variadic template types
      // Adapted from https://stackoverflow.com/a/9145665
      // which seems to use the pattern described in
      // https://stackoverflow.com/a/22968432
      namespace TemplateOuterProduct
      {
        using namespace WeakForms::SelfLinearization::internal::Utilities;

        // Outer Product
        template <typename T, typename U>
        struct OuterProduct;

        // Partially specialise the empty case for the first TypeList.
        template <typename... Us>
        struct OuterProduct<TypeList<>, TypeList<Us...>>
        {
          using type = TypeList<>;
        };


        // The general case for two TypeLists. Process:
        // 1. Expand out the head of the first TypeList with the full second
        // TypeList.
        // 2. Recurse the tail of the first TypeList.
        // 3. Concatenate the two TypeLists.
        template <typename T, typename... Ts, typename... Us>
        struct OuterProduct<TypeList<T, Ts...>, TypeList<Us...>>
        {
          using type = typename Concatenate<
            TypeList<TypePair<T, Us>...>,
            typename OuterProduct<TypeList<Ts...>,
                                  TypeList<Us...>>::type>::type;
        };
      } // namespace TemplateOuterProduct



      // Ensure that template arguments contain no duplicates.
      // Adapted from https://stackoverflow.com/a/34122593
      namespace TemplateRestrictions
      {
        template <typename T, typename... List>
        struct IsContained;


        template <typename T, typename Head, typename... Tail>
        struct IsContained<T, Head, Tail...>
        {
          static constexpr bool value =
            std::is_same<T, Head>::value || IsContained<T, Tail...>::value;
        };


        template <typename T>
        struct IsContained<T>
        {
          static constexpr bool value = false;
        };


        template <typename... List>
        struct IsUnique;


        template <typename Head, typename... Tail>
        struct IsUnique<Head, Tail...>
        {
          static constexpr bool value =
            !IsContained<Head, Tail...>::value && IsUnique<Tail...>::value;
        };


        template <>
        struct IsUnique<>
        {
          static constexpr bool value = true;
        };


        template <typename... Ts>
        struct EnforceNoDuplicates
        {
          static_assert(IsUnique<Ts...>::value, "No duplicate types allowed.");

          static constexpr bool value = IsUnique<Ts...>::value;
        };

        template <typename T, typename... Us>
        struct is_unary_op_subspace_field_solution
        {
          static constexpr bool value =
            is_unary_op_subspace_field_solution<T>::value &&
            is_unary_op_subspace_field_solution<Us...>::value;
        };

        // Scalar and Vector subspaces
        template <template <class> typename SubSpaceViewsType,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode>
        struct is_unary_op_subspace_field_solution<
          WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>, OpCode>>
        {
          static constexpr bool value =
            is_field_solution<SubSpaceViewsType<SpaceType>>::value &&
            is_subspace_view<SubSpaceViewsType<SpaceType>>::value;
        };

        // Tensor and SymmetricTensor subspaces
        template <template <int, class> typename SubSpaceViewsType,
                  int rank,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode>
        struct is_unary_op_subspace_field_solution<
          WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank, SpaceType>,
                                        OpCode>>
        {
          static constexpr bool value =
            is_field_solution<SubSpaceViewsType<rank, SpaceType>>::value &&
            is_subspace_view<SubSpaceViewsType<rank, SpaceType>>::value;
        };

        template <typename T>
        struct is_unary_op_subspace_field_solution<T> : std::false_type
        {};

        template <typename... FieldArgs>
        struct EnforceIsUnaryOpSubspaceFieldSolution
        {
          static_assert(
            is_unary_op_subspace_field_solution<FieldArgs...>::value,
            "Template arguments must be unary operation subspace field solutions.");

          static constexpr bool value = true;
        };


        // Determine types resulting from differential operations
        // of scalars, tensors and symmetric tensors.
        namespace Differentiation
        {
          template <typename T, typename U>
          struct DiffOpResult;

          // Differentiate a scalar with respect to another scalar
          template <typename T, typename U>
          struct DiffOpResult<typename EnableIfScalar<T>::type,
                              typename EnableIfScalar<U>::type>
          {
            using scalar_type = typename ProductType<T, U>::type;
            using type        = scalar_type;
          };

          // Differentiate a scalar with respect to a tensor
          template <int rank, int dim, typename T, typename U>
          struct DiffOpResult<
            typename EnableIfScalar<T>::type,
            Tensor<rank, dim, typename EnableIfScalar<U>::type>>
          {
            using scalar_type = typename ProductType<T, U>::type;
            using type        = Tensor<rank, dim, scalar_type>;
          };

          // Differentiate a scalar with respect to a symmetric tensor
          template <int rank, int dim, typename T, typename U>
          struct DiffOpResult<
            typename EnableIfScalar<T>::type,
            SymmetricTensor<rank, dim, typename EnableIfScalar<U>::type>>
          {
            using scalar_type = typename ProductType<T, U>::type;
            using type        = SymmetricTensor<rank, dim, scalar_type>;
          };

          // Differentiate a tensor with respect to a scalar
          template <int rank, int dim, typename T, typename U>
          struct DiffOpResult<
            Tensor<rank, dim, typename EnableIfScalar<T>::type>,
            typename EnableIfScalar<U>::type>
          {
            using scalar_type = typename ProductType<T, U>::type;
            using type        = Tensor<rank, dim, scalar_type>;
          };

          // Differentiate a tensor with respect to another tensor
          template <int rank_1, int rank_2, int dim, typename T, typename U>
          struct DiffOpResult<
            Tensor<rank_1, dim, typename EnableIfScalar<T>::type>,
            Tensor<rank_2, dim, typename EnableIfScalar<U>::type>>
          {
            using scalar_type = typename ProductType<T, U>::type;
            using type        = Tensor<rank_1 + rank_2, dim, scalar_type>;
          };

          // Differentiate a tensor with respect to a symmetric tensor
          template <int rank_1, int rank_2, int dim, typename T, typename U>
          struct DiffOpResult<
            Tensor<rank_1, dim, typename EnableIfScalar<T>::type>,
            SymmetricTensor<rank_2, dim, typename EnableIfScalar<U>::type>>
          {
            using scalar_type = typename ProductType<T, U>::type;
            using type        = Tensor<rank_1 + rank_2, dim, scalar_type>;
          };

          // Differentiate a symmetric tensor with respect to a scalar
          template <int rank, int dim, typename T, typename U>
          struct DiffOpResult<
            SymmetricTensor<rank, dim, typename EnableIfScalar<T>::type>,
            typename EnableIfScalar<U>::type>
          {
            using scalar_type = typename ProductType<T, U>::type;
            using type        = SymmetricTensor<rank, dim, scalar_type>;
          };

          // Differentiate a symmetric tensor with respect to a tensor
          template <int rank_1, int rank_2, int dim, typename T, typename U>
          struct DiffOpResult<
            SymmetricTensor<rank_1, dim, typename EnableIfScalar<T>::type>,
            Tensor<rank_2, dim, typename EnableIfScalar<U>::type>>
          {
            using scalar_type = typename ProductType<T, U>::type;
            using type        = Tensor<rank_1 + rank_2, dim, scalar_type>;
          };

          // Differentiate a symmetric tensor with respect to another symmetric
          // tensor
          template <int rank_1, int rank_2, int dim, typename T, typename U>
          struct DiffOpResult<
            SymmetricTensor<rank_1, dim, typename EnableIfScalar<T>::type>,
            SymmetricTensor<rank_2, dim, typename EnableIfScalar<U>::type>>
          {
            using scalar_type = typename ProductType<T, U>::type;
            using type = SymmetricTensor<rank_1 + rank_2, dim, scalar_type>;
          };
        } // namespace Differentiation
      }   // namespace TemplateRestrictions



      // Forward declarations
      template <typename UnaryOpTestFunction, typename TypeListFunctorArgs>
      struct LinearFormGenerator;


      template <typename UnaryOpTestFunction, typename... FunctorArgs>
      struct LinearFormGenerator<UnaryOpTestFunction,
                                 Utilities::TypeList<FunctorArgs...>>
      {
        static_assert(
          is_test_function<UnaryOpTestFunction>::value,
          "First argument should be a unary op to a test function.");

        using test_function_t = UnaryOpTestFunction;

        template <typename NumberType>
        using value_t =
          typename test_function_t::template value_type<NumberType>;

        // template <typename NumberType>
        // using value_t = typename
        // TemplateRestrictions::Differentiation::DiffOpResult<NumberType,
        //   typename test_function_t::template value_type<NumberType>>::type;

        template <typename NumberType>
        using functor_t =
          std::function<value_t<NumberType>(const FunctorArgs &...)>;
      };


      namespace PromoteTo
      {
        template <typename TypeListFunctorArgs, typename TypeListTestFunctions>
        struct LinearFormGeneratorType;


        template <typename UnaryOpTestFunction, typename... FunctorArgs>
        struct LinearFormGeneratorType<Utilities::TypeList<FunctorArgs...>,
                                       Utilities::TypeList<UnaryOpTestFunction>>
        {
          using type = LinearFormGenerator<UnaryOpTestFunction,
                                           Utilities::TypeList<FunctorArgs...>>;
        };


        // Take a list of test functions that will have functors with the same
        // arguments, then split them up into their individual generators.
        // In the end, concatenate all of these into a single list.
        template <typename... FunctorArgs,
                  typename TestFunction,
                  typename... OtherTestFunctions>
        struct LinearFormGeneratorType<
          Utilities::TypeList<FunctorArgs...>,
          Utilities::TypeList<TestFunction, OtherTestFunctions...>>
        {
          using type = Utilities::TypeList<
            typename LinearFormGeneratorType<
              Utilities::TypeList<FunctorArgs...>,
              Utilities::TypeList<TestFunction>>::type,
            typename LinearFormGeneratorType<
              Utilities::TypeList<FunctorArgs...>,
              Utilities::TypeList<OtherTestFunctions...>>::type>;
        };
      } // namespace PromoteTo



      namespace Utilities
      {
        template <typename UnaryOpTestFunction, typename TypeListFunctorArgs>
        struct TypeHelper<
          LinearFormGenerator<UnaryOpTestFunction, TypeListFunctorArgs>>
        {
          using type =
            LinearFormGenerator<UnaryOpTestFunction, TypeListFunctorArgs>;

          static std::string
          print()
          {
            using test_function_t = typename type::test_function_t;
            using functor_t       = typename type::template functor_t<double>;
            using result_t        = typename functor_t::result_type;

            // Print something that vaguely resembles form notation
            return "<" + TypeHelper<test_function_t>::print() + ", " +
                   TypeHelper<result_t>::print() + "(" +
                   TypeHelper<TypeListFunctorArgs>::print() + ")" + ">";
          }
        };


        template <typename TypeListTestFunction, typename TypeListFunctorArgs>
        struct TypeHelper<
          PromoteTo::LinearFormGeneratorType<TypeListTestFunction,
                                             TypeListFunctorArgs>>
        {
          using type = PromoteTo::LinearFormGeneratorType<TypeListTestFunction,
                                                          TypeListFunctorArgs>;

          static std::string
          print()
          {
            return TypeHelper<typename type::type>::print();
          }
        };
      }; // namespace Utilities


      // This struct will take all of the unary operations (value, gradient,
      // divergence, curl, ...) and construct the following things with it:
      // - A type and print function to show what the input UnaryOps were
      // - A type and print function to show what the constructed test function
      // UnaryOps are
      // - A type and print function to show what the constructed trial solution
      // UnaryOps are
      // - A type and print function that are associated with the arguments
      //   that must be passed to the user-defined functor for each derivative
      //
      template <typename... UnaryOpsSubSpaceFieldSolution>
      struct SelfLinearizationHelper
      {
      private:
        // All template parameter types must be unary operators
        // for subspaces of a field solution.
        static_assert(
          TemplateRestrictions::EnforceIsUnaryOpSubspaceFieldSolution<
            UnaryOpsSubSpaceFieldSolution...>::value,
          "Template arguments must be unary operation subspace field solutions.");

        // We cannot permit multiple instance of the same unary operations
        // as a part of the template parameter pack. This would imply that
        // we want the user to define a functor that takes in multiple instances
        // of the same field variable, which does not make sense.
        static_assert(TemplateRestrictions::EnforceNoDuplicates<
                        UnaryOpsSubSpaceFieldSolution...>::value,
                      "No duplicate types allowed.");

        // A type list of the unary operators to field solutions for
        // a subspace.
        // This type is primarily to assist in verification and debugging.
        using type_list_field_solution_unary_op_t =
          Utilities::TypeList<UnaryOpsSubSpaceFieldSolution...>;

        // The product type of the solution fields with themselves.
        // This type is primarily to assist in verification and debugging.
        using field_solution_unary_op_outer_product_type =
          typename TemplateOuterProduct::OuterProduct<
            type_list_field_solution_unary_op_t,
            type_list_field_solution_unary_op_t>::type;

      public:
        // A type list of the value types for the unary op arguments.
        // These will be passed on to the functors for the value and
        // derivative(s) of self-linearizing forms.
        template <typename NumberType>
        using type_list_functor_arguments =
          Utilities::TypeList<typename UnaryOpsSubSpaceFieldSolution::
                                template value_type<NumberType>...>;

        // A type list of the unary operators to test functions for
        // a subspace.
        using type_list_test_function_unary_op_t =
          Utilities::TypeList<typename ConvertTo::test_function_t<
            UnaryOpsSubSpaceFieldSolution>...>;

        // A type list of the unary operators to trial solutions for
        // a subspace.
        using type_list_trial_solution_unary_op_t =
          Utilities::TypeList<typename ConvertTo::trial_solution_t<
            UnaryOpsSubSpaceFieldSolution>...>;

        // This function is primarily to assist in verification and debugging.
        static std::string
        print_type_list_test_function_unary_op()
        {
          return Utilities::TypeHelper<
            type_list_test_function_unary_op_t>::print();
        }

        // This function is primarily to assist in verification and debugging.
        static std::string
        print_type_list_trial_solution_unary_op()
        {
          return Utilities::TypeHelper<
            type_list_trial_solution_unary_op_t>::print();
        }

        // This function is primarily to assist in verification and debugging.
        template <typename NumberType>
        static std::string
        print_type_list_functor_arguments()
        {
          return Utilities::TypeHelper<
            type_list_functor_arguments<NumberType>>::print();
        }

        // This function is primarily to assist in verification and debugging.
        static std::string
        print_type_list_field_solution_unary_op()
        {
          return Utilities::TypeHelper<
            type_list_field_solution_unary_op_t>::print();
        }

        // This function is primarily to assist in verification and debugging.
        static std::string
        print_field_solution_unary_op_outer_product_type()
        {
          return Utilities::TypeHelper<
            field_solution_unary_op_outer_product_type>::print();
        }
      }; // class SelfLinearizationHelper

      // template<typename NumberType,
      //           typename... FunctorArguments>
      // struct LinearizedLinearForm
      // {

      // };

      // template<typename NumberType,
      //           typename... FunctorArguments>
      // struct LinearizationBilinearForm
      // {

      // };


      // template <typename... UnaryOpsSubSpaceFieldSolution>
      // struct LinearizedForm
      // {

      // };

    } // namespace internal


    /**
     * OP: (AutoDifferentiableFunctor)
     *
     * First derivatives of this form produce a ResidualForm.
     */
    template <typename... UnaryOpsSubSpaceFieldSolution>
    class SelfLinearizingEnergyFunctional
    {
      using Helper_t =
        internal::SelfLinearizationHelper<UnaryOpsSubSpaceFieldSolution...>;

      using type_list_test_function_unary_op_t =
        typename Helper_t::type_list_test_function_unary_op_t;
      using type_list_trial_solution_unary_op_t =
        typename Helper_t::type_list_trial_solution_unary_op_t;

      // Patterns to create the structs that will themselves generate linear and
      // bilinear forms: Linear form is just a list
      using linear_forms_pattern_t =
        typename Helper_t::type_list_test_function_unary_op_t;
      // The Cartesian product type of the test functions with the trial
      // solutions.
      using bilinear_forms_pattern_t =
        typename internal::TemplateOuterProduct::OuterProduct<
          type_list_test_function_unary_op_t,
          type_list_trial_solution_unary_op_t>::type;

    public:
      template <typename NumberType>
      using functor_arguments_t =
        typename Helper_t::template type_list_functor_arguments<NumberType>;

      template <typename NumberType>
      using linear_forms_generator_t = typename internal::Utilities::TypeHelper<
        internal::PromoteTo::LinearFormGeneratorType<
          functor_arguments_t<NumberType>,
          linear_forms_pattern_t>>::type;

      // This function is primarily to assist in verification and debugging.
      static std::string
      print_linear_forms_pattern()
      {
        return internal::Utilities::TypeHelper<linear_forms_pattern_t>::print();
      }

      // This function is primarily to assist in verification and debugging.
      static std::string
      print_bilinear_forms_pattern()
      {
        return internal::Utilities::TypeHelper<
          bilinear_forms_pattern_t>::print();
      }

      // This function is primarily to assist in verification and debugging.
      template <typename NumberType>
      static std::string
      print_functor_arguments()
      {
        return internal::Utilities::TypeHelper<
          functor_arguments_t<NumberType>>::print();
      }

      // This function is primarily to assist in verification and debugging.
      template <typename NumberType>
      static std::string
      print_linear_forms_generator()
      {
        return internal::Utilities::TypeHelper<
          linear_forms_generator_t<NumberType>>::print();
      }

      // // This function is primarily to assist in verification and debugging.
      // static std::string
      // print_test_function_trial_solution_unary_op_outer_product_type()
      // {
      //   return Utilities::TypeHelper<
      //     test_function_trial_solution_unary_op_outer_product_type>::print();
      // }
    };

    /**
     * OP: (Variation, SymbolicFunctor)
     *
     * This class gets converted into a LinearForm.
     * First derivatives of this form produce a BilinearForm through the
     * LinearizationForm
     */
    template <typename... UnaryOpsSubSpaceFieldSolution>
    class SelfLinearizingResidualForm
    {};

    /**
     * OP: (Variation, SymbolicFunctor, Linearization)
     *
     * This class gets converted into a LinearForm.
     * First derivatives of this form produce a BilinearForm through the
     * LinearizationForm
     */
    class LinearizationForm
    {
    private:
      // friend EnergyFunctional;
      // friend ResidualForm;
      LinearizationForm() = default;
    };
  } // namespace SelfLinearization

} // namespace WeakForms



/* ======================== Convenience functions ======================== */



// namespace WeakForms
// {
//   namespace internal
//   {
//     template <typename NumberType = double, typename...
//     UnaryOpsSubSpaceFieldSolution>
//     LinearForm<UnaryOpsSubSpaceFieldSolution...>
//     linearized_form(const LinearizedLinearForm<NumberType, typename
//     SelfLinearizationHelper<UnaryOpsSubSpaceFieldSolution...>::type_list_functor_arguments<NumberType>>
//     &linearized_linear_form,
//                     const LinearizationBilinearForm<NumberType, typename
//                     SelfLinearizationHelper<UnaryOpsSubSpaceFieldSolution...>::type_list_functor_arguments<NumberType>>
//                     &linearization_bilinear_form, const
//                     UnaryOpsSubSpaceFieldSolution
//                     &...unary_ops_field_solution)
//     {
//       return LinearizedForm<UnaryOpsSubSpaceFieldSolution...>(test_space_op,
//       functor_op);
//     }
//   }
// } // namespace WeakForms



// namespace WeakForms
// {
//   // template <typename TestSpaceOp, typename TrialSpaceOp>
//   // BilinearForm<TestSpaceOp, NoOp, TrialSpaceOp>
//   // bilinear_form(const TestSpaceOp & test_space_op,
//   //               const TrialSpaceOp &trial_space_op)
//   // {
//   //   return BilinearForm<TestSpaceOp, NoOp, TrialSpaceOp>(test_space_op,
//   //                                                        trial_space_op);
//   // }

//   template <typename ADFunctor, typename... FieldArgs>
//   SelfLinearization::EnergyFunctional<ADFunctor, FieldArgs...>
//   ad_energy_functional_form(const ADFunctor &functor_op,
//                             const FieldArgs &... dependent_fields)
//   {
//     return SelfLinearization::EnergyFunctional<ADFunctor, FieldArgs...>(
//       functor_op, dependent_fields...);
//   }

// } // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_self_linearizing_forms_h
