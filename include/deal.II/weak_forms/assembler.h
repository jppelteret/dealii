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

#ifndef dealii_weakforms_assembler_h
#define dealii_weakforms_assembler_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/weak_forms/bilinear_forms.h>
#include <deal.II/weak_forms/integral.h>
#include <deal.II/weak_forms/linear_forms.h>
#include <deal.II/weak_forms/type_traits.h>
#include <deal.II/weak_forms/unary_operators.h>

#include <functional>
#include <type_traits>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace internal
  {
    // template<int rank_result,
    //          int dim>
    // Tensor<>
    // {

    // }

    enum class AccumulationSign
    {
      plus,
      minus
    };

    // template<typename ReturnType, typename T1, typename T2, typename T = void>
    // struct FullContraction;

    // /**
    //  * Generic contraction
    //  * 
    //  * Type T1 is a scalar
    //  */
    // template<typename ReturnType, typename T1, typename T2>
    // struct FullContraction<ReturnType,T1,T2, typename std::enable_if<std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value>::type>
    // {
    //   static ReturnType
    //   contract(const T1 &t1, const T2 &t2)
    //   {
    //     return t1*t2;
    //   }
    // };


    // /**
    //  * Generic contraction
    //  * 
    //  * Type T2 is a scalar
    //  */
    // template<typename T1, typename T2>
    // struct FullContraction<T1,T2, typename std::enable_if<std::is_arithmetic<T2>::value && !std::is_arithmetic<T1>::value>::type>
    // {
    //   static ReturnType
    //   contract(const T1 &t1, const T2 &t2)
    //   {
    //     // Call other implementation
    //     return FullContraction<ReturnType,T2,T1>::contract(t2,t1);
    //   }
    // };


    template<typename T1, typename T2, typename T = void>
    struct FullContraction;

    /**
     * Contraction with a scalar or complex scalar
     * 
     * At least one of the templated types is an arithmetic type
     */
    template<typename T1, typename T2>
    struct FullContraction<T1,T2, 
      typename std::enable_if<std::is_arithmetic<T1>::value ||
                              std::is_arithmetic<T2>::value>::type>
    {
      static auto
      contract(const T1 &t1, const T2 &t2) -> decltype(t1*t2)
      {
        return t1*t2;
      }
    };
    template<typename T1, typename T2>
    struct FullContraction<std::complex<T1>,T2, 
      typename std::enable_if<std::is_arithmetic<T1>::value ||
                              std::is_arithmetic<T2>::value>::type>
    {
      static auto
      contract(const std::complex<T1> &t1, const T2 &t2) -> decltype(t1*t2)
      {
        return t1*t2;
      }
    };
    template<typename T1, typename T2>
    struct FullContraction<T1,std::complex<T2>, 
      typename std::enable_if<std::is_arithmetic<T1>::value ||
                              std::is_arithmetic<T2>::value>::type>
    {
      static auto
      contract(const T1 &t1, const std::complex<T2> &t2) -> decltype(t1*t2)
      {
        return t1*t2;
      }
    };
    template<typename T1, typename T2>
    struct FullContraction<std::complex<T1>,std::complex<T2>, 
      typename std::enable_if<std::is_arithmetic<T1>::value ||
                              std::is_arithmetic<T2>::value>::type>
    {
      static auto
      contract(const std::complex<T1> &t1, const std::complex<T2> &t2) -> decltype(t1*t2)
      {
        return t1*t2;
      }
    };

    /**
     * Contraction with a vectorized scalar
     * 
     * At least one of the templated types is a VectorizedArray
     */
    template<typename T1, typename T2>
    struct FullContraction<VectorizedArray<T1>,T2>
    {
      static auto
      contract(const VectorizedArray<T1> &t1, const T2 &t2) -> decltype(t1*t2)
      {
        return t1*t2;
      }
    };
    template<typename T1, typename T2>
    struct FullContraction<T1,VectorizedArray<T2>>
    {
      static auto
      contract(const T1 &t1, const VectorizedArray<T2> &t2) -> decltype(t1*t2)
      {
        return t1*t2;
      }
    };
    template<typename T1, typename T2>
    struct FullContraction<VectorizedArray<T1>,VectorizedArray<T2>>
    {
      static auto
      contract(const VectorizedArray<T1> &t1, const VectorizedArray<T2> &t2) -> decltype(t1*t2)
      {
        return t1*t2;
      }
    };

    /**
     * Contraction with a tensor
     * 
     * Here we recognise that the shape functions can only be
     * scalar valued (dealt with in the above specializations),
     * vector valued (Tensors of rank 1), rank-2 tensor valued or
     * rank-2 symmetric tensor valued. For the rank 1 and rank 2
     * case, we already have full contraction operations that we
     * can leverage.
     */
    template<int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<Tensor<rank_1,dim,T1>, Tensor<rank_2,dim,T2>, 
      typename std::enable_if<(rank_1 == 0 || rank_2 == 0)>::type>
    {
      static Tensor<rank_1+rank_2,dim,typename ProductType< T1, T2 >::type>
      contract(const Tensor<rank_1,dim,T1> &t1, const Tensor<rank_2,dim,T2> &t2)
      {
        return t1*t2;
      }
    };

    template<int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<Tensor<rank_1,dim,T1>, Tensor<rank_2,dim,T2>, 
      typename std::enable_if<((rank_1 == 1 && rank_2 >= 1) || (rank_2 == 1 && rank_1 >= 1))>::type>
    {
      static Tensor<rank_1+rank_2-2,dim,typename ProductType< T1, T2 >::type>
      contract(const Tensor<rank_1,dim,T1> &t1, const Tensor<rank_2,dim,T2> &t2)
      {
        return dealii::contract<rank_1-1,0>(t1,t2);
      }
    };

    template<int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<Tensor<rank_1,dim,T1>, Tensor<rank_2,dim,T2>, 
      typename std::enable_if<((rank_1 == 2 && rank_2 >= 2) || (rank_2 == 2 && rank_1 >= 2))>::type>
    {
      static Tensor<rank_1+rank_2-4, dim, typename ProductType< T1, T2 >::type>
      contract(const Tensor<rank_1,dim,T1> &t1, const Tensor<rank_2,dim,T2> &t2)
      {
        return dealii::double_contract<rank_1-2,0, rank_1-1,1>(t1,t2);
      }
    };

    template<int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<Tensor<rank_1,dim,T1>, Tensor<rank_2,dim,T2>, 
      typename std::enable_if<(rank_1 > 2 && rank_2 > 2 && rank_1 == rank_2)>::type>
    {
      static typename ProductType< T1, T2 >::type
      contract(const Tensor<rank_1,dim,T1> &t1, const Tensor<rank_2,dim,T2> &t2)
      {
        return scalar_product(t1,t2);
      }
    };

    template<int dim, typename T1, typename T2>
    struct FullContraction<SymmetricTensor<2,dim,T1>, SymmetricTensor<2,dim,T2>>
    {
      static typename ProductType< T1, T2 >::type
      contract(const SymmetricTensor<2,dim,T1> &t1, const SymmetricTensor<2,dim,T2> &t2)
      {
        // Always a double contraction
        return t1*t2;
      }
    };

    template<int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<SymmetricTensor<rank_1,dim,T1>, SymmetricTensor<rank_2,dim,T2>, 
      typename std::enable_if<(rank_1 == 2 & rank_2 > 2) || (rank_2 == 2 && rank_1 > 2)>::type>
    {
      static SymmetricTensor<rank_1+rank_2-4, dim, typename ProductType< T1, T2 >::type>
      contract(const SymmetricTensor<rank_1,dim,T1> &t1, const SymmetricTensor<rank_2,dim,T2> &t2)
      {
        // Always a double contraction
        return t1*t2;
      }
    };

    template<int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<Tensor<rank_1,dim,T1>, SymmetricTensor<rank_2,dim,T2>, typename std::enable_if<(rank_1 == 1)>::type>
    {
      static Tensor<rank_1+rank_2-2,dim,typename ProductType< T1, T2 >::type>
      contract(const Tensor<rank_1,dim,T1> &t1, const SymmetricTensor<rank_2,dim,T2> &t2)
      {
        return t1*t2;
      }
    };

    template<int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<SymmetricTensor<rank_1,dim,T1>, Tensor<rank_2,dim,T2>, typename std::enable_if<(rank_2 == 1)>::type>
    {
      static Tensor<rank_1+rank_2-2,dim,typename ProductType< T1, T2 >::type>
      contract(const SymmetricTensor<rank_1,dim,T1> &t1, const Tensor<rank_2,dim,T2> &t2)
      {
        return t1*t2;
      }
    };

    template<int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<Tensor<rank_1,dim,T1>, SymmetricTensor<rank_2,dim,T2>, typename std::enable_if<(rank_1 > 1)>::type>
    {
      // With mixed tensor types, its easier just to be defensive and not worry
      // about the symmetries of one of the tensors. The main issue comes in when
      // there are mixed ranks for the two arguments. Also, it might be more
      // expensive to do the symmetrization and subsequent contraction, as
      // opposed to this conversion and standard contraction.
      static auto
      contract(const Tensor<rank_1,dim,T1> &t1, const SymmetricTensor<rank_2,dim,T2> &t2)
        -> decltype(FullContraction<Tensor<rank_1,dim,T1>,Tensor<rank_2,dim,T2>>::contract(Tensor<rank_1,dim,T1>(),Tensor<rank_2,dim,T2>()))
      {
        using Contraction_t = FullContraction<Tensor<rank_1,dim,T1>,Tensor<rank_2,dim,T2>>;
        return Contraction_t::contract(t1, Tensor<rank_2,dim,T2>(t2));
      }
    };

    template<int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<SymmetricTensor<rank_1,dim,T1>, Tensor<rank_2,dim,T2>, typename std::enable_if<(rank_2 > 1)>::type>
    {
      static auto
      contract(const SymmetricTensor<rank_1,dim,T1> &t1, const Tensor<rank_2,dim,T2> &t2)
        -> decltype(FullContraction<Tensor<rank_1,dim,T1>,Tensor<rank_2,dim,T2>>::contract(Tensor<rank_1,dim,T1>(),Tensor<rank_2,dim,T2>()))
      {
        using Contraction_t = FullContraction<Tensor<rank_1,dim,T1>,Tensor<rank_2,dim,T2>>;
        return Contraction_t::contract(Tensor<rank_1,dim,T2>(t1), t2);
      }
    };


    // Valid for cell and face assembly
    template <enum AccumulationSign Sign,
              typename NumberType,
              int dim,
              int spacedim,
              typename ValueTypeTest,
              typename ValueTypeFunctor,
              typename ValueTypeTrial>
    void
    assemble_cell_matrix_contribution(
      FullMatrix<NumberType> &                        cell_matrix,
      const FEValuesBase<dim, spacedim> &             fe_values_dofs,
      const FEValuesBase<dim, spacedim> &             fe_values_q_points,
      const std::vector<std::vector<ValueTypeTest>> & shapes_test,
      const std::vector<ValueTypeFunctor> &           values_functor,
      const std::vector<std::vector<ValueTypeTrial>> &shapes_trial,
      const std::vector<double> &                     JxW)
    {
      Assert(shapes_test.size() == fe_values_dofs.dofs_per_cell,
             ExcDimensionMismatch(shapes_test.size(),
                                  fe_values_dofs.dofs_per_cell));
      Assert(shapes_trial.size() == fe_values_dofs.dofs_per_cell,
             ExcDimensionMismatch(shapes_trial.size(),
                                  fe_values_dofs.dofs_per_cell));
      Assert(values_functor.size() == fe_values_q_points.n_quadrature_points,
             ExcDimensionMismatch(values_functor.size(),
                                  fe_values_q_points.n_quadrature_points));
      Assert(JxW.size() == fe_values_q_points.n_quadrature_points,
             ExcDimensionMismatch(JxW.size(),
                                  fe_values_q_points.n_quadrature_points));
      for (const unsigned int k : fe_values_dofs.dof_indices())
        {
          Assert(shapes_test[k].size() ==
                   fe_values_q_points.n_quadrature_points,
                 ExcDimensionMismatch(shapes_test[k].size(),
                                      fe_values_q_points.n_quadrature_points));
          Assert(shapes_trial[k].size() ==
                   fe_values_q_points.n_quadrature_points,
                 ExcDimensionMismatch(shapes_trial[k].size(),
                                      fe_values_q_points.n_quadrature_points));
        }

      // TODO: Optimise this; precompute [ values_functor(q) *
      // shapes_trial[j][q] * JxW[q]; ]
      // TODO: Account for symmetry, if desired.
      // for (const unsigned int i : fe_values_dofs.dof_indices())
      //   for (const unsigned int j : fe_values_dofs.dof_indices())
      //     for (const unsigned int q :
      //          fe_values_q_points.quadrature_point_indices())
      //       {
      //         const auto contribution =
      //         (shapes_test[i][q] * values_functor[q] * shapes_trial[j][q]) *
      //           JxW[q];

      //         if (Sign == AccumulationSign::plus)
      //           {
      //             cell_matrix(i, j) += contribution;
      //           }
      //         else
      //           {
      //             Assert(Sign == AccumulationSign::minus, ExcInternalError());
      //             cell_matrix(i, j) -= contribution;
      //           }
      //       }

      // This is the equivalent of
      // for (q : q_points)
      //   for (i : dof_indices)
      //     for (j : dof_indices)
      //       cell_matrix(i,j) += shapes_test[i][q] * values_functor[q] * shapes_trial[j][q]) * JxW[q]
      for (const unsigned int q : fe_values_q_points.quadrature_point_indices())
      {
        for (const unsigned int j : fe_values_dofs.dof_indices())
        {
          using ContractionType_FS = FullContraction<ValueTypeFunctor,ValueTypeTrial>;
          const ValueTypeTest functor_x_shape_trial_x_JxW 
            = JxW[q] * ContractionType_FS::contract(values_functor[q],shapes_trial[j][q]);
          
          for (const unsigned int i : fe_values_dofs.dof_indices())
            {
              using ContractionType_SFS_JxW = FullContraction<ValueTypeTest,ValueTypeTest>;
              const NumberType contribution = ContractionType_SFS_JxW::contract(shapes_test[i][q],functor_x_shape_trial_x_JxW);

              if (Sign == AccumulationSign::plus)
                {
                  cell_matrix(i, j) += contribution;
                }
              else
                {
                  Assert(Sign == AccumulationSign::minus, ExcInternalError());
                  cell_matrix(i, j) -= contribution;
                }
            }
        }
      }
    }

    // Valid only for cell assembly
    template <enum AccumulationSign Sign,
              typename NumberType,
              int dim,
              int spacedim,
              typename ValueTypeTest,
              typename ValueTypeFunctor,
              typename ValueTypeTrial>
    void
    assemble_cell_matrix_contribution(
      FullMatrix<NumberType> &                        cell_matrix,
      const FEValuesBase<dim, spacedim> &             fe_values,
      const std::vector<std::vector<ValueTypeTest>> & shapes_test,
      const std::vector<ValueTypeFunctor> &           values_functor,
      const std::vector<std::vector<ValueTypeTrial>> &shapes_trial,
      const std::vector<double> &                     JxW)
    {
      assemble_cell_matrix_contribution<Sign>(cell_matrix,
                                              fe_values,
                                              fe_values,
                                              shapes_test,
                                              values_functor,
                                              shapes_trial,
                                              JxW);
    }



    // Valid for cell and face assembly
    template <enum AccumulationSign Sign,
              typename NumberType,
              int dim,
              int spacedim,
              typename ValueTypeTest,
              typename ValueTypeFunctor>
    void
    assemble_cell_vector_contribution(
      Vector<NumberType> &                           cell_vector,
      const FEValuesBase<dim, spacedim> &            fe_values_dofs,
      const FEValuesBase<dim, spacedim> &            fe_values_q_points,
      const std::vector<std::vector<ValueTypeTest>> &shapes_test,
      const std::vector<ValueTypeFunctor> &          values_functor,
      const std::vector<double> &                    JxW)
    {
      Assert(shapes_test.size() == fe_values_dofs.dofs_per_cell,
             ExcDimensionMismatch(shapes_test.size(),
                                  fe_values_dofs.dofs_per_cell));
      Assert(values_functor.size() == fe_values_q_points.n_quadrature_points,
             ExcDimensionMismatch(values_functor.size(),
                                  fe_values_q_points.n_quadrature_points));
      Assert(JxW.size() == fe_values_q_points.n_quadrature_points,
             ExcDimensionMismatch(JxW.size(),
                                  fe_values_q_points.n_quadrature_points));
      for (const unsigned int k : fe_values_dofs.dof_indices())
        {
          Assert(shapes_test[k].size() ==
                   fe_values_q_points.n_quadrature_points,
                 ExcDimensionMismatch(shapes_test[k].size(),
                                      fe_values_q_points.n_quadrature_points));
        }

      for (const unsigned int i : fe_values_dofs.dof_indices())
        for (const unsigned int q :
             fe_values_q_points.quadrature_point_indices())
          {
            const auto contribution =
              (shapes_test[i][q] * values_functor[q]) * JxW[q];

            // The sign of the accumulation is swapped for the vector, because
            // it it accumulated as a LHS quantity but then assembled onto the
            // RHS. So swapping the sign here allows us to skip negating the
            // whole cell_vector before assembly into the global vector.
            if (Sign == AccumulationSign::plus)
              {
                cell_vector(i) -= contribution;
              }
            else
              {
                Assert(Sign == AccumulationSign::minus, ExcInternalError());
                cell_vector(i) += contribution;
              }
          }
    }

    // Valid only for cell assembly
    template <enum AccumulationSign Sign,
              typename NumberType,
              int dim,
              int spacedim,
              typename ValueTypeTest,
              typename ValueTypeFunctor>
    void
    assemble_cell_vector_contribution(
      Vector<NumberType> &                           cell_vector,
      const FEValuesBase<dim, spacedim> &            fe_values,
      const std::vector<std::vector<ValueTypeTest>> &shapes_test,
      const std::vector<ValueTypeFunctor> &          values_functor,
      const std::vector<double> &                    JxW)
    {
      assemble_cell_vector_contribution<Sign>(
        cell_vector, fe_values, fe_values, shapes_test, values_functor, JxW);
    }

  } // namespace internal



  template <int dim, int spacedim = dim, typename NumberType = double>
  class AssemblerBase
  {
  public:
    using AsciiLatexOperation = std::function<std::string(const SymbolicDecorations &decorator)>;
    using StringOperation =
      std::function<std::pair<AsciiLatexOperation, enum internal::AccumulationSign>(
        void)>;

    using CellMatrixOperation =
      std::function<void(FullMatrix<NumberType> &           cell_matrix,
                         const FEValuesBase<dim, spacedim> &fe_values)>;
    using CellVectorOperation =
      std::function<void(Vector<NumberType> &               cell_vector,
                         const FEValuesBase<dim, spacedim> &fe_values)>;

    // TODO: Figure out how to get rid of this template parameter
    // We can easily do it if we exclusively use FEValuesViews, as
    // the get_function_XYZ_from_local_dof_values() functions can
    // be called without the solution vector itself. So we could decant
    // the relevant components of the solution vector into a std::vector
    // and pass those off to the functors.
    template <typename VectorType = Vector<NumberType>>
    using CellSolutionUpdateOperation =
      std::function<void(const VectorType &                 solution,
                         const FEValuesBase<dim, spacedim> &fe_values)>;
    using BoundaryMatrixOperation =
      std::function<void(FullMatrix<NumberType> &           cell_matrix,
                         const FEValuesBase<dim, spacedim> &fe_values,
                         const FEFaceValuesBase<dim, spacedim> &fe_face_values,
                         const unsigned int                  face)>;
    using BoundaryVectorOperation =
      std::function<void(Vector<NumberType> &               cell_vector,
                         const FEValuesBase<dim, spacedim> &fe_values,
                         const FEFaceValuesBase<dim, spacedim> &fe_face_values,
                         const unsigned int                  face)>;

    using InterfaceMatrixOperation =
      std::function<void(FullMatrix<NumberType> &           cell_matrix,
                         const FEValuesBase<dim, spacedim> &fe_values,
                         const FEFaceValuesBase<dim, spacedim> &fe_face_values,
                         const unsigned int                  face)>;
    using InterfaceVectorOperation =
      std::function<void(Vector<NumberType> &               cell_vector,
                         const FEValuesBase<dim, spacedim> &fe_values,
                         const FEFaceValuesBase<dim, spacedim> &fe_face_values,
                         const unsigned int                  face)>;


    virtual ~AssemblerBase() = default;


    template <typename UnaryOpType,
              typename std::enable_if<
                is_symbolic_volume_integral<UnaryOpType>::value>::type* = nullptr>
    AssemblerBase &
    operator+=(const UnaryOpType &volume_integral)
    {
      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      constexpr auto sign = internal::AccumulationSign::plus;
      add_ascii_latex_operations<sign>(volume_integral);
      add_cell_operation<sign>(volume_integral);

      const auto &form    = volume_integral.get_integrand();
      const auto &functor = form.get_functor();
      add_solution_update_operation(functor);

      return *this;
    }

    template <typename UnaryOpType,
              typename std::enable_if<
                is_symbolic_boundary_integral<UnaryOpType>::value>::type* = nullptr>
    AssemblerBase &
    operator+=(const UnaryOpType &boundary_integral)
    {
      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      constexpr auto sign = internal::AccumulationSign::plus;
      add_ascii_latex_operations<sign>(boundary_integral);
      add_boundary_face_operation<sign>(boundary_integral);

      const auto &form    = boundary_integral.get_integrand();
      const auto &functor = form.get_functor();
      add_solution_update_operation(functor);

      return *this;
    }

    template <typename UnaryOpType,
              typename std::enable_if<
                is_symbolic_interface_integral<UnaryOpType>::value>::type* = nullptr>
    AssemblerBase &
    operator+=(const UnaryOpType &interface_integral)
    {
      // static_assert(false, "Assembler: operator += not yet implemented for interface integrals");

      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      // constexpr auto sign = internal::AccumulationSign::plus;
      // add_ascii_latex_operations<sign>(boundary_integral);
      // add_cell_operation<sign>(boundary_integral);

      // const auto &form    = boundary_integral.get_integrand();
      // const auto &functor = form.get_functor();
      // add_solution_update_operation(functor);

      return *this;
    }

    template <typename UnaryOpType,
              typename std::enable_if<
                is_symbolic_volume_integral<UnaryOpType>::value>::type* = nullptr>
    AssemblerBase &
    operator-=(const UnaryOpType &volume_integral)
    {
      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      constexpr auto sign = internal::AccumulationSign::minus;
      add_ascii_latex_operations<sign>(volume_integral);
      add_cell_operation<sign>(volume_integral);

      const auto &form    = volume_integral.get_integrand();
      const auto &functor = form.get_functor();
      add_solution_update_operation(functor);

      return *this;
    }

    template <typename UnaryOpType,
              typename std::enable_if<
                is_symbolic_boundary_integral<UnaryOpType>::value>::type* = nullptr>
    AssemblerBase &
    operator-=(const UnaryOpType &boundary_integral)
    {
      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      constexpr auto sign = internal::AccumulationSign::minus;
      add_ascii_latex_operations<sign>(boundary_integral);
      add_boundary_face_operation<sign>(boundary_integral);

      const auto &form    = boundary_integral.get_integrand();
      const auto &functor = form.get_functor();
      add_solution_update_operation(functor);

      return *this;
    }

    template <typename UnaryOpType,
              typename std::enable_if<
                is_symbolic_interface_integral<UnaryOpType>::value>::type* = nullptr>
    AssemblerBase &
    operator-=(const UnaryOpType &interface_integral)
    {
      // static_assert(false, "Assembler: operator -= not yet implemented for interface integrals");

      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      // constexpr auto sign = internal::AccumulationSign::minus;
      // add_ascii_latex_operations<sign>(boundary_integral);
      // add_cell_operation<sign>(boundary_integral);

      // const auto &form    = boundary_integral.get_integrand();
      // const auto &functor = form.get_functor();
      // add_solution_update_operation(functor);

      return *this;
    }


    // TODO:
    std::string
    as_ascii(const SymbolicDecorations &decorator) const
    {
      std::string output = "0 = ";
      for (unsigned int i = 0; i < as_ascii_operations.size(); ++i)
        {
          Assert(as_ascii_operations[i], ExcNotInitialized());
          const auto &current_term_function = as_ascii_operations[i];
          const AsciiLatexOperation &string_op = current_term_function().first;
          output += string_op(decorator);
          if (i + 1 < as_ascii_operations.size())
            {
              Assert(as_ascii_operations[i + 1], ExcNotInitialized());
              const auto &next_term_function = as_ascii_operations[i + 1];
              if (next_term_function().second ==
                  internal::AccumulationSign::plus)
                {
                  output += " + ";
                }
              else
                {
                  Assert(next_term_function().second ==
                           internal::AccumulationSign::minus,
                         ExcInternalError());
                  output += " - ";
                }
            }
        }
      return output;
    }

    std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      std::string output = "0 = ";
      for (unsigned int i = 0; i < as_latex_operations.size(); ++i)
        {
          Assert(as_latex_operations[i], ExcNotInitialized());
          const auto &current_term_function = as_latex_operations[i];
          const AsciiLatexOperation &string_op = current_term_function().first;
          output += string_op(decorator);
          if (i + 1 < as_latex_operations.size())
            {
              Assert(as_latex_operations[i + 1], ExcNotInitialized());
              const auto &next_term_function = as_latex_operations[i + 1];
              if (next_term_function().second ==
                  internal::AccumulationSign::plus)
                {
                  output += " + ";
                }
              else
                {
                  Assert(next_term_function().second ==
                           internal::AccumulationSign::minus,
                         ExcInternalError());
                  output += " - ";
                }
            }
        }
      return output;
    }


    template <typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    update_solution(const VectorType &        solution_vector,
                    const DoFHandlerType &    dof_handler,
                    const CellQuadratureType &cell_quadrature)
    {
      static_assert(DoFHandlerType::dimension == dim,
                    "Dimension is incompatible");
      static_assert(DoFHandlerType::space_dimension == spacedim,
                    "Space dimension is incompatible");

      using CellIteratorType = typename DoFHandlerType::active_cell_iterator;
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CopyData         = MeshWorker::CopyData<0, 0, 0>; // Empty copier

      // Define a cell worker
      const auto &cell_solution_update_operations =
        this->cell_solution_update_operations;
      auto cell_worker = [&cell_solution_update_operations,
                          &solution_vector](const CellIteratorType &cell,
                                            ScratchData &scratch_data,
                                            CopyData &   copy_data) {
        const auto &fe_values = scratch_data.reinit(cell);

        // Perform all operations that contribute to the local cell matrix
        for (const auto &cell_solution_update_op :
             cell_solution_update_operations)
          {
            cell_solution_update_op(solution_vector, fe_values);
          }

        // TODO:
        // boundary_matrix_operations
        // interface_matrix_operations
      };

      auto dummy_copier = [](const CopyData &copy_data) {};

      const ScratchData sample_scratch_data(dof_handler.get_fe(),
                                            cell_quadrature,
                                            this->get_cell_update_flags());
      const CopyData    sample_copy_data(dof_handler.get_fe().dofs_per_cell);

      MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                            cell_worker,
                            dummy_copier,
                            sample_scratch_data,
                            sample_copy_data,
                            MeshWorker::assemble_own_cells);
    }

  protected:
    explicit AssemblerBase()
      : cell_update_flags(update_default)
      , cell_solution_update_flags(update_default)
      , boundary_face_update_flags(update_default)
      , boundary_face_solution_update_flags(update_default)
      , interface_face_update_flags(update_default)
      , interface_face_solution_update_flags(update_default)
    {}


    template <enum internal::AccumulationSign Sign, typename IntegralType>
    typename std::enable_if<is_symbolic_integral<IntegralType>::value>::type
    add_ascii_latex_operations(const IntegralType &integral)
    {
      // Augment the composition of the operation
      // Important note: All operations must be captured by copy!
      as_ascii_operations.push_back(
        [integral]() { return std::make_pair([integral](const SymbolicDecorations &decorator){ return integral.as_ascii(decorator); }, Sign); });
      as_latex_operations.push_back(
        [integral]() { return std::make_pair([integral](const SymbolicDecorations &decorator){ return integral.as_latex(decorator); }, Sign); });
    }

    /**
     * Cell operations for bilinear forms
     *
     * @tparam UnaryOpVolumeIntegral
     * @tparam std::enable_if<is_bilinear_form<
     * typename UnaryOpVolumeIntegral::IntegrandType>::value>::type
     * @param volume_integral
     */
    template <enum internal::AccumulationSign Sign,
              typename UnaryOpVolumeIntegral,
              typename std::enable_if<is_bilinear_form<
      typename UnaryOpVolumeIntegral::IntegrandType>::value>::type* = nullptr>
    void
    add_cell_operation(const UnaryOpVolumeIntegral &volume_integral)
    {
      static_assert(is_symbolic_volume_integral<UnaryOpVolumeIntegral>::value, "Expected a volume integral type.");

      // We need to update the flags that need to be set for
      // cell operations. The flags from the composite operation
      // that composes the integrand will be bubbled down to the
      // integral itself.
      cell_update_flags |= volume_integral.get_update_flags();

      // Extract some information about the form that we'll be
      // constructing and integrating
      const auto &form = volume_integral.get_integrand();
      static_assert(
        is_bilinear_form<typename std::decay<decltype(form)>::type>::value,
        "Incompatible integrand type.");

      const auto &test_space_op  = form.get_test_space_operation();
      const auto &functor        = form.get_functor();
      const auto &trial_space_op = form.get_trial_space_operation();

      using TestSpaceOp  = typename std::decay<decltype(test_space_op)>::type;
      using Functor      = typename std::decay<decltype(functor)>::type;
      using TrialSpaceOp = typename std::decay<decltype(trial_space_op)>::type;

      using ValueTypeTest =
        typename TestSpaceOp::template value_type<NumberType>;
      using ValueTypeFunctor =
        typename Functor::template value_type<NumberType>;
      using ValueTypeTrial =
        typename TrialSpaceOp::template value_type<NumberType>;

      // Now, compose all of this into a bespoke operation for this
      // contribution.
      //
      // Important note: All operations must be captured by copy!
      // We do this in case someone inlines a call to bilinear_form()
      // with operator+= , e.g.
      //   MatrixBasedAssembler<dim, spacedim> assembler;
      //   assembler += bilinear_form(test_val, coeff_func, trial_val).dV();
      auto f = [volume_integral,
                test_space_op,
                functor,
                trial_space_op](FullMatrix<NumberType> &           cell_matrix,
                                const FEValuesBase<dim, spacedim> &fe_values) {
        // Skip this cell if it doesn't match the criteria set for the
        // integration domain.
        if (!volume_integral.get_integral_operation().integrate_on_cell(
              fe_values.get_cell()))
          {
            return;
          }

        const unsigned int n_dofs_per_cell = fe_values.dofs_per_cell;
        const unsigned int n_q_points      = fe_values.n_quadrature_points;

        // Get all values at the quadrature points
        // TODO: Can we use std::array here?
        const std::vector<double> &         JxW =
          volume_integral.template          operator()<NumberType>(fe_values);
        const std::vector<ValueTypeFunctor> values_functor =
          functor.template                  operator()<NumberType>(fe_values);

        // Get the shape function data (value, gradients, curls, etc.)
        // for all quadrature points at all DoFs. We construct it in this
        // manner (with the q_point indices fast) so that we can perform
        // contractions in an optimal manner.
        // TODO: Can we use std::array here?
        std::vector<std::vector<ValueTypeTest>> shapes_test(
          n_dofs_per_cell, std::vector<ValueTypeTest>(n_q_points));
        std::vector<std::vector<ValueTypeTest>> shapes_trial(
          n_dofs_per_cell, std::vector<ValueTypeTest>(n_q_points));
        for (const unsigned int k : fe_values.dof_indices())
          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              shapes_test[k][q] =
                test_space_op.template operator()<NumberType>(fe_values, k, q);
              shapes_trial[k][q] =
                trial_space_op.template operator()<NumberType>(fe_values, k, q);
            }

        internal::assemble_cell_matrix_contribution<Sign>(cell_matrix,
                                                          fe_values,
                                                          shapes_test,
                                                          values_functor,
                                                          shapes_trial,
                                                          JxW);
      };
      cell_matrix_operations.emplace_back(f);
    }


    template <enum internal::AccumulationSign Sign,
              typename UnaryOpBoundaryIntegral,
              typename std::enable_if<is_bilinear_form<
      typename UnaryOpBoundaryIntegral::IntegrandType>::value>::type* = nullptr>
    void
    add_boundary_face_operation(const UnaryOpBoundaryIntegral &boundary_integral)
    {
      static_assert(is_symbolic_boundary_integral<UnaryOpBoundaryIntegral>::value, "Expected a boundary integral type.");
      // static_assert(false, "Assembler: Boundary face operations not yet implemented for bilinear forms.")
    }

    
    template <enum internal::AccumulationSign Sign,
              typename UnaryOpInterfaceIntegral,
              typename std::enable_if<is_bilinear_form<
      typename UnaryOpInterfaceIntegral::IntegrandType>::value>::type* = nullptr>
    void
    add_internal_face_operation(const UnaryOpInterfaceIntegral &interface_integral)
    {
      static_assert(is_symbolic_interface_integral<UnaryOpInterfaceIntegral>::value, "Expected an interface integral type.");
      // static_assert(false, "Assembler: Internal face operations not yet implemented for bilinear forms.")
    }


    /**
     * Cell operations for linear forms
     *
     * @tparam UnaryOpVolumeIntegral
     * @tparam std::enable_if<is_linear_form<
     * typename UnaryOpVolumeIntegral::IntegrandType>::value>::type
     * @param volume_integral
     */
    template <enum internal::AccumulationSign Sign,
              typename UnaryOpVolumeIntegral,
              typename std::enable_if<is_linear_form<
      typename UnaryOpVolumeIntegral::IntegrandType>::value>::type* = nullptr>
    void
    add_cell_operation(const UnaryOpVolumeIntegral &volume_integral)
    {
      static_assert(is_symbolic_volume_integral<UnaryOpVolumeIntegral>::value, "Expected a volume integral type.");

      // We need to update the flags that need to be set for
      // cell operations. The flags from the composite operation
      // that composes the integrand will be bubbled down to the
      // integral itself.
      cell_update_flags |= volume_integral.get_update_flags();

      // Extract some information about the form that we'll be
      // constructing and integrating
      const auto &form = volume_integral.get_integrand();
      static_assert(
        is_linear_form<typename std::decay<decltype(form)>::type>::value,
        "Incompatible integrand type.");

      const auto &test_space_op = form.get_test_space_operation();
      const auto &functor       = form.get_functor();

      using TestSpaceOp  = typename std::decay<decltype(test_space_op)>::type;
      using Functor      = typename std::decay<decltype(functor)>::type;

      using ValueTypeTest =
        typename TestSpaceOp::template value_type<NumberType>;
      using ValueTypeFunctor =
        typename Functor::template value_type<NumberType>;

      // Now, compose all of this into a bespoke operation for this
      // contribution.
      //
      // Important note: All operations must be captured by copy!
      // We do this in case someone inlines a call to bilinear_form()
      // with operator+= , e.g.
      //   MatrixBasedAssembler<dim, spacedim> assembler;
      //   assembler += linear_form(test_val, coeff_func).dV();
      auto f = [volume_integral,
                test_space_op,
                functor](Vector<NumberType> &               cell_vector,
                         const FEValuesBase<dim, spacedim> &fe_values) {
        // Skip this cell if it doesn't match the criteria set for the
        // integration domain.
        if (!volume_integral.get_integral_operation().integrate_on_cell(
              fe_values.get_cell()))
          {
            return;
          }

        const unsigned int n_dofs_per_cell = fe_values.dofs_per_cell;
        const unsigned int n_q_points      = fe_values.n_quadrature_points;

        // Get all values at the quadrature points
        // TODO: Can we use std::array here?
        const std::vector<double> &         JxW =
          volume_integral.template          operator()<NumberType>(fe_values);
        const std::vector<ValueTypeFunctor> values_functor =
          functor.template                  operator()<NumberType>(fe_values);

        // Get the shape function data (value, gradients, curls, etc.)
        // for all quadrature points at all DoFs. We construct it in this
        // manner (with the q_point indices fast) so that we can perform
        // contractions in an optimal manner.
        // TODO: Can we use std::array here?
        std::vector<std::vector<ValueTypeTest>> shapes_test(
          n_dofs_per_cell, std::vector<ValueTypeTest>(n_q_points));
        for (const unsigned int k : fe_values.dof_indices())
          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              shapes_test[k][q] =
                test_space_op.template operator()<NumberType>(fe_values, k, q);
            }

        internal::assemble_cell_vector_contribution<Sign>(
          cell_vector, fe_values, shapes_test, values_functor, JxW);
      };
      cell_vector_operations.emplace_back(f);
    }


    template <enum internal::AccumulationSign Sign,
              typename UnaryOpBoundaryIntegral,
              typename std::enable_if<is_linear_form<
      typename UnaryOpBoundaryIntegral::IntegrandType>::value>::type* = nullptr>
    void
    add_boundary_face_operation(const UnaryOpBoundaryIntegral &boundary_integral)
    {
      static_assert(is_symbolic_boundary_integral<UnaryOpBoundaryIntegral>::value, "Expected a boundary integral type.");
      // static_assert(false, "Assembler: Boundary face operations not yet implemented for linear forms.")
    
      // We need to update the flags that need to be set for
      // cell operations. The flags from the composite operation
      // that composes the integrand will be bubbled down to the
      // integral itself.
      boundary_face_update_flags |= boundary_integral.get_update_flags();

      // Extract some information about the form that we'll be
      // constructing and integrating
      const auto &form = boundary_integral.get_integrand();
      static_assert(
        is_linear_form<typename std::decay<decltype(form)>::type>::value,
        "Incompatible integrand type.");

      const auto &test_space_op = form.get_test_space_operation();
      const auto &functor       = form.get_functor();

      using TestSpaceOp  = typename std::decay<decltype(test_space_op)>::type;
      using Functor      = typename std::decay<decltype(functor)>::type;

      using ValueTypeTest =
        typename TestSpaceOp::template value_type<NumberType>;
      using ValueTypeFunctor =
        typename Functor::template value_type<NumberType>;

      // Now, compose all of this into a bespoke operation for this
      // contribution.
      //
      // Important note: All operations must be captured by copy!
      // We do this in case someone inlines a call to bilinear_form()
      // with operator+= , e.g.
      //   MatrixBasedAssembler<dim, spacedim> assembler;
      //   assembler += linear_form(test_val, boundary_func).dA();
      auto f = [boundary_integral,
                test_space_op,
                functor](Vector<NumberType> &               cell_vector,
                         const FEValuesBase<dim, spacedim> &fe_values,
                         const FEFaceValuesBase<dim, spacedim> &fe_face_values,
                         const unsigned int                 face) {
        // Skip this cell if it doesn't match the criteria set for the
        // integration domain.
        if (!boundary_integral.get_integral_operation().integrate_on_face(
              fe_values.get_cell(), face))
          {
            return;
          }

        const unsigned int n_dofs_per_cell = fe_values.dofs_per_cell;
        const unsigned int n_q_points      = fe_face_values.n_quadrature_points;

        // Get all values at the quadrature points
        // TODO: Can we use std::array here?
        const std::vector<double> &         JxW =
          boundary_integral.template          operator()<NumberType>(fe_face_values);
        const std::vector<ValueTypeFunctor> values_functor =
          functor.template                  operator()<NumberType>(fe_face_values);

        // Get the shape function data (value, gradients, curls, etc.)
        // for all quadrature points at all DoFs. We construct it in this
        // manner (with the q_point indices fast) so that we can perform
        // contractions in an optimal manner.
        // TODO: Can we use std::array here?
        std::vector<std::vector<ValueTypeTest>> shapes_test(
          n_dofs_per_cell, std::vector<ValueTypeTest>(n_q_points));
        for (const unsigned int k : fe_values.dof_indices())
          for (const unsigned int q : fe_face_values.quadrature_point_indices())
            {
              shapes_test[k][q] =
                test_space_op.template operator()<NumberType>(fe_face_values, k, q);
            }

        internal::assemble_cell_vector_contribution<Sign>(
          cell_vector, fe_values, fe_face_values, shapes_test, values_functor, JxW);
      };
      boundary_face_vector_operations.emplace_back(f);
    }

    
    template <enum internal::AccumulationSign Sign,
              typename UnaryOpInterfaceIntegral,
              typename std::enable_if<is_symbolic_interface_integral<UnaryOpInterfaceIntegral>::value && is_linear_form<
      typename UnaryOpInterfaceIntegral::IntegrandType>::value>::type* = nullptr>
    void
    add_internal_face_operation(const UnaryOpInterfaceIntegral &interface_integral)
    {
      static_assert(is_symbolic_interface_integral<UnaryOpInterfaceIntegral>::value, "Expected an interface integral type.");
      // static_assert(false, "Assembler: Internal face operations not yet implemented for linear forms.")
    }


    template <typename FunctorType>
    typename std::enable_if<!is_ad_functor<FunctorType>::value>::type
    add_solution_update_operation(FunctorType &functor)
    {
      // Do nothing
    }


    template <typename VectorType = Vector<double>, typename FunctorType>
    typename std::enable_if<is_ad_functor<FunctorType>::value>::type
    add_solution_update_operation(FunctorType &functor)
    {
      cell_solution_update_flags |= functor.get_update_flags();

      auto f = [&functor](const VectorType &               solution_vector,
                        const FEValuesBase<dim, spacedim> &fe_values) {
          functor.update_from_solution(solution_vector,fe_values);
      };
      cell_solution_update_operations.emplace_back(f);
    }

    UpdateFlags
    get_cell_update_flags() const
    {
      return cell_update_flags | cell_solution_update_flags;
    }

    UpdateFlags
    get_face_update_flags() const
    {
      return boundary_face_update_flags | boundary_face_solution_update_flags |
             interface_face_update_flags | interface_face_solution_update_flags;
    }

    std::vector<StringOperation> as_ascii_operations;
    std::vector<StringOperation> as_latex_operations;

    UpdateFlags                      cell_update_flags;
    std::vector<CellMatrixOperation> cell_matrix_operations;
    std::vector<CellVectorOperation> cell_vector_operations;

    UpdateFlags                                cell_solution_update_flags;
    std::vector<CellSolutionUpdateOperation<>> cell_solution_update_operations;

    UpdateFlags                          boundary_face_update_flags;
    std::vector<BoundaryMatrixOperation> boundary_face_matrix_operations;
    std::vector<BoundaryVectorOperation> boundary_face_vector_operations;

    UpdateFlags                                boundary_face_solution_update_flags;
    std::vector<CellSolutionUpdateOperation<>> boundary_face_solution_update_operations;

    UpdateFlags                           interface_face_update_flags;
    std::vector<InterfaceMatrixOperation> interface_face_matrix_operations;
    std::vector<InterfaceVectorOperation> interface_face_vector_operations;

    UpdateFlags                                interface_face_solution_update_flags;
    std::vector<CellSolutionUpdateOperation<>> interface_face_solution_update_operations;
  };



  // TODO: Put in another header
  template <int dim, int spacedim = dim, typename NumberType = double>
  class MatrixBasedAssembler : public AssemblerBase<dim, spacedim, NumberType>
  {
    template<typename CellIteratorType, typename ScratchData, typename CopyData> 
    using CellWorkerType = std::function<void(const CellIteratorType &,
                                              ScratchData &, 
                                              CopyData &)>;
  
  template<typename CellIteratorType, typename ScratchData, typename CopyData> 
  using BoundaryWorkerType = std::function<void(const CellIteratorType &, 
                                                const unsigned int &, 
                                                ScratchData &,
                                                CopyData &)>;
  
  template<typename CellIteratorType, typename ScratchData, typename CopyData> 
  using FaceWorkerType = std::function< void(const CellIteratorType &, 
                                             const unsigned int, 
                                             const unsigned int, 
                                             const CellIteratorType &, 
                                             const unsigned int, 
                                             const unsigned int,
                                             ScratchData &, 
                                             CopyData &)>;

  public:
    explicit MatrixBasedAssembler()
      : AssemblerBase<dim, spacedim, NumberType>(){};

    /**
     * Assemble the linear system matrix, excluding boundary and internal
     * face contributions.
     *
     * @tparam NumberType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @ Note: Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename MatrixType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    assemble_matrix(MatrixType &                         system_matrix,
             const AffineConstraints<NumberType> &constraints,
             const DoFHandlerType &               dof_handler,
             const CellQuadratureType &           cell_quadrature) const
    {
      static_assert(DoFHandlerType::dimension == dim,
                    "Dimension is incompatible");
      static_assert(DoFHandlerType::space_dimension == spacedim,
                    "Space dimension is incompatible");

      Assert(this->boundary_face_matrix_operations.empty(), 
             ExcMessage("Assembly with only cell contributions has been selected, "
                        "while there are boundary face contributions in to the "
                        "bilinear form. You should use the other assemble_matrix() "
                        "function that takes in face quadrature as an argument so "
                        "that all contributions are considered."));

      Assert(this->interface_face_matrix_operations.empty(), 
             ExcMessage("Assembly with only cell contributions has been selected, "
                        "while there are internal face contributions in to the "
                        "bilinear form. You should use the other assemble_matrix() "
                        "function that takes in face quadrature as an argument so "
                        "that all contributions are considered."));

      using CellIteratorType = typename DoFHandlerType::active_cell_iterator;
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CopyData         = MeshWorker::CopyData<1, 1, 1>;

      // Define a cell worker
      const auto &cell_matrix_operations = this->cell_matrix_operations;
      auto cell_worker = CellWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!cell_matrix_operations.empty())
      {
        cell_worker = [&cell_matrix_operations](const CellIteratorType &cell,
                                                    ScratchData &scratch_data,
                                                    CopyData &   copy_data) {
          const auto &fe_values          = scratch_data.reinit(cell);
          copy_data                      = CopyData(fe_values.dofs_per_cell);
          copy_data.local_dof_indices[0] = scratch_data.get_local_dof_indices();

          FullMatrix<NumberType> &cell_matrix = copy_data.matrices[0];

          // Perform all operations that contribute to the local cell matrix
          for (const auto &cell_matrix_op : cell_matrix_operations)
            {
              cell_matrix_op(cell_matrix, fe_values);
            }
        };
      }

      auto copier = [&constraints, &system_matrix](const CopyData &copy_data) {
        const FullMatrix<NumberType> &cell_matrix = copy_data.matrices[0];
        const std::vector<types::global_dof_index> &local_dof_indices =
          copy_data.local_dof_indices[0];

        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               system_matrix);
      };

      const ScratchData sample_scratch_data(dof_handler.get_fe(),
                                            cell_quadrature,
                                            this->get_cell_update_flags());
      const CopyData    sample_copy_data(dof_handler.get_fe().dofs_per_cell);


      MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_nothing;
      if (!cell_matrix_operations.empty())
        assemble_flags |= MeshWorker::assemble_own_cells;

      if (assemble_flags)
      {
        MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                              cell_worker,
                              copier,
                              sample_scratch_data,
                              sample_copy_data,
                              assemble_flags);

        system_matrix.compress(VectorOperation::add);
      }

      // DEBUGGING!
      // ScratchData scratch = sample_scratch_data;
      // for (const auto cell : dof_handler.active_cell_iterators())
      //   {
      //     CopyData copy = sample_copy_data;

      //     cell_worker(cell, scratch, copy);
      //     copier(copy);
      //   }
    }

    /**
     * Assemble the linear system matrix, including boundary and internal
     * face contributions.
     *
     * @tparam NumberType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @ Note: Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename MatrixType,
              typename DoFHandlerType,
              typename CellQuadratureType,
              typename FaceQuadratureType>
    void
    assemble_matrix(MatrixType &                         system_matrix,
             const AffineConstraints<NumberType> &constraints,
             const DoFHandlerType &               dof_handler,
             const CellQuadratureType &           cell_quadrature,
             const FaceQuadratureType &           face_quadrature) const
    {
      static_assert(DoFHandlerType::dimension == dim,
                    "Dimension is incompatible");
      static_assert(DoFHandlerType::space_dimension == spacedim,
                    "Space dimension is incompatible");

      using CellIteratorType = typename DoFHandlerType::active_cell_iterator;
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CopyData         = MeshWorker::CopyData<1, 1, 1>;

      // Define a cell worker
      const auto &cell_matrix_operations = this->cell_matrix_operations;
      auto cell_worker = CellWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!cell_matrix_operations.empty())
      {
        cell_worker = [&cell_matrix_operations](const CellIteratorType &cell,
                                                    ScratchData &scratch_data,
                                                    CopyData &   copy_data) {
          const auto &fe_values          = scratch_data.reinit(cell);
          copy_data                      = CopyData(fe_values.dofs_per_cell);
          copy_data.local_dof_indices[0] = scratch_data.get_local_dof_indices();

          FullMatrix<NumberType> &cell_matrix = copy_data.matrices[0];

          // Perform all operations that contribute to the local cell matrix
          for (const auto &cell_matrix_op : cell_matrix_operations)
            {
              cell_matrix_op(cell_matrix, fe_values);
            }
        };
      }

      // Define a boundary worker
      const auto &boundary_face_matrix_operations = this->boundary_face_matrix_operations;
      auto boundary_worker = BoundaryWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!boundary_face_matrix_operations.empty())
      {
        boundary_worker = [&boundary_face_matrix_operations](const CellIteratorType &cell,
                                              const unsigned int face,
                                              ScratchData &scratch_data,
                                              CopyData &   copy_data) 
        {
          Assert((cell->face(face)->at_boundary()), ExcMessage("Cell face is not at the boundary."));

          const auto &fe_values = scratch_data.reinit(cell);
          const auto &fe_face_values = scratch_data.reinit(cell,face);
          copy_data             = CopyData(fe_values.dofs_per_cell);
          copy_data.local_dof_indices[0] = scratch_data.get_local_dof_indices();

          FullMatrix<NumberType> &cell_matrix = copy_data.matrices[0];

          // Perform all operations that contribute to the local cell matrix
          for (const auto &boundary_face_matrix_op : boundary_face_matrix_operations)
            {
              boundary_face_matrix_op(cell_matrix, fe_values, fe_face_values, face);
            }
        };
      }

      // Define a face / interface worker
      const auto &interface_face_matrix_operations = this->interface_face_matrix_operations;
      auto face_worker = FaceWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!interface_face_matrix_operations.empty())
      {
        // interface_matrix_operations
        AssertThrow(false, ExcMessage("Internal face cell matrix contributions have not yet been implemented."));
      }

      auto copier = [&constraints, &system_matrix](const CopyData &copy_data) {
        const FullMatrix<NumberType> &cell_matrix = copy_data.matrices[0];
        const std::vector<types::global_dof_index> &local_dof_indices =
          copy_data.local_dof_indices[0];

        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               system_matrix);
      };

      const ScratchData sample_scratch_data(dof_handler.get_fe(),
                                            cell_quadrature,
                                            this->get_cell_update_flags(),
                                            face_quadrature,
                                            this->get_face_update_flags());
      const CopyData    sample_copy_data(dof_handler.get_fe().dofs_per_cell);

      MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_nothing;
      if (!cell_matrix_operations.empty())
        assemble_flags |= MeshWorker::assemble_own_cells;
      if (!boundary_face_matrix_operations.empty())
        assemble_flags |= MeshWorker::assemble_boundary_faces;
      if (!interface_face_matrix_operations.empty())
        assemble_flags |= MeshWorker::assemble_own_interior_faces_once;

      if (assemble_flags)
      {
        MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                              cell_worker,
                              copier,
                              sample_scratch_data,
                              sample_copy_data,
                              assemble_flags);

        system_matrix.compress(VectorOperation::add);
      }

      // DEBUGGING!
      // ScratchData scratch = sample_scratch_data;
      // for (const auto cell : dof_handler.active_cell_iterators())
      //   {
      //     CopyData copy = sample_copy_data;

      //     cell_worker(cell, scratch, copy);
      //     copier(copy);
      //   }
    }

    /**
     * Assemble a RHS vector, boundary and internal face contributions.
     *
     * @tparam NumberType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @ Note: Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    assemble_rhs_vector(VectorType &                         system_vector,
             const AffineConstraints<NumberType> &constraints,
             const DoFHandlerType &               dof_handler,
             const CellQuadratureType &           cell_quadrature) const
    {
      static_assert(DoFHandlerType::dimension == dim,
                    "Dimension is incompatible");
      static_assert(DoFHandlerType::space_dimension == spacedim,
                    "Space dimension is incompatible");

      Assert(this->boundary_face_vector_operations.empty(), 
             ExcMessage("Assembly with only cell contributions has been selected, "
                        "while there are boundary face contributions in to the "
                        "linear form. You should use the other assemble_rhs_vector() "
                        "function that takes in face quadrature as an argument so "
                        "that all contributions are considered."));

      Assert(this->interface_face_vector_operations.empty(), 
             ExcMessage("Assembly with only cell contributions has been selected, "
                        "while there are internal face contributions in to the "
                        "linear form. You should use the other assemble_rhs_vector() "
                        "function that takes in face quadrature as an argument so "
                        "that all contributions are considered."));

      using CellIteratorType = typename DoFHandlerType::active_cell_iterator;
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CopyData         = MeshWorker::CopyData<1, 1, 1>;

      // Define a cell worker
      const auto &cell_vector_operations = this->cell_vector_operations;
      auto cell_worker = CellWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!cell_vector_operations.empty())
      {
        cell_worker = [&cell_vector_operations](const CellIteratorType &cell,
                                              ScratchData &scratch_data,
                                              CopyData &   copy_data) 
        {
          const auto &fe_values = scratch_data.reinit(cell);
          copy_data             = CopyData(fe_values.dofs_per_cell);
          copy_data.local_dof_indices[0] = scratch_data.get_local_dof_indices();

          Vector<NumberType> &    cell_vector = copy_data.vectors[0];

          // Perform all operations that contribute to the local cell vector
          for (const auto &cell_vector_op : cell_vector_operations)
            {
              cell_vector_op(cell_vector, fe_values);
            }
        };
      }

      auto copier = [&constraints, &system_vector](
                      const CopyData &copy_data) {
        const Vector<NumberType> &    cell_vector = copy_data.vectors[0];
        const std::vector<types::global_dof_index> &local_dof_indices =
          copy_data.local_dof_indices[0];

        constraints.distribute_local_to_global(cell_vector,
                                               local_dof_indices,
                                               system_vector);
      };

      const ScratchData sample_scratch_data(dof_handler.get_fe(),
                                            cell_quadrature,
                                            this->get_cell_update_flags());
      const CopyData    sample_copy_data(dof_handler.get_fe().dofs_per_cell);

      MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_nothing;
      if (!cell_vector_operations.empty())
        assemble_flags |= MeshWorker::assemble_own_cells;

      if (assemble_flags)
      {
        MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                              cell_worker,
                              copier,
                              sample_scratch_data,
                              sample_copy_data,
                              assemble_flags);

        system_vector.compress(VectorOperation::add);
      }

      // DEBUGGING!
      // ScratchData scratch = sample_scratch_data;
      // for (const auto cell : dof_handler.active_cell_iterators())
      //   {
      //     CopyData copy = sample_copy_data;

      //     cell_worker(cell, scratch, copy);
      //     copier(copy);
      //   }
    }

    /**
     * Assemble a RHS vector, including boundary and internal face contributions.
     *
     * @tparam NumberType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @ Note: Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType,
              typename FaceQuadratureType>
    void
    assemble_rhs_vector(VectorType &                         system_vector,
             const AffineConstraints<NumberType> &constraints,
             const DoFHandlerType &               dof_handler,
             const CellQuadratureType &           cell_quadrature,
             const FaceQuadratureType &           face_quadrature) const
    {
      static_assert(DoFHandlerType::dimension == dim,
                    "Dimension is incompatible");
      static_assert(DoFHandlerType::space_dimension == spacedim,
                    "Space dimension is incompatible");

      using CellIteratorType = typename DoFHandlerType::active_cell_iterator;
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CopyData         = MeshWorker::CopyData<1, 1, 1>;

      // Define a cell worker
      const auto &cell_vector_operations = this->cell_vector_operations;
      auto cell_worker = CellWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!cell_vector_operations.empty())
      {
        cell_worker = [&cell_vector_operations](const CellIteratorType &cell,
                                              ScratchData &scratch_data,
                                              CopyData &   copy_data) 
        {
          const auto &fe_values = scratch_data.reinit(cell);
          copy_data             = CopyData(fe_values.dofs_per_cell);
          copy_data.local_dof_indices[0] = scratch_data.get_local_dof_indices();

          Vector<NumberType> &    cell_vector = copy_data.vectors[0];

          // Perform all operations that contribute to the local cell vector
          for (const auto &cell_vector_op : cell_vector_operations)
            {
              cell_vector_op(cell_vector, fe_values);
            }
        };
      }

      // Define a boundary worker
      const auto &boundary_face_vector_operations = this->boundary_face_vector_operations;
      auto boundary_worker = BoundaryWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!boundary_face_vector_operations.empty())
      {
        boundary_worker = [&boundary_face_vector_operations](const CellIteratorType &cell,
                                              const unsigned int face,
                                              ScratchData &scratch_data,
                                              CopyData &   copy_data) 
        {
          Assert((cell->face(face)->at_boundary()), ExcMessage("Cell face is not at the boundary."));

          const auto &fe_values = scratch_data.reinit(cell);
          const auto &fe_face_values = scratch_data.reinit(cell,face);
          copy_data             = CopyData(fe_values.dofs_per_cell);
          copy_data.local_dof_indices[0] = scratch_data.get_local_dof_indices();

          Vector<NumberType> &    cell_vector = copy_data.vectors[0];

          // Perform all operations that contribute to the local cell vector
          for (const auto &boundary_face_vector_op : boundary_face_vector_operations)
            {
              boundary_face_vector_op(cell_vector, fe_values, fe_face_values, face);
            }
        };
      }

      // Define a face / interface worker
      const auto &interface_face_vector_operations = this->interface_face_vector_operations;
      auto face_worker = FaceWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!interface_face_vector_operations.empty())
      {
        // interface_vector_operations
        AssertThrow(false, ExcMessage("Internal face cell vector contributions have not yet been implemented."));
      }

      auto copier = [&constraints, &system_vector](
                      const CopyData &copy_data) {
        const Vector<NumberType> &    cell_vector = copy_data.vectors[0];
        const std::vector<types::global_dof_index> &local_dof_indices =
          copy_data.local_dof_indices[0];

        constraints.distribute_local_to_global(cell_vector,
                                               local_dof_indices,
                                               system_vector);
      };

      const ScratchData sample_scratch_data(dof_handler.get_fe(),
                                            cell_quadrature,
                                            this->get_cell_update_flags(),
                                            face_quadrature,
                                            this->get_face_update_flags());
      const CopyData    sample_copy_data(dof_handler.get_fe().dofs_per_cell);

      MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_nothing;
      if (!cell_vector_operations.empty())
        assemble_flags |= MeshWorker::assemble_own_cells;
      if (!boundary_face_vector_operations.empty())
        assemble_flags |= MeshWorker::assemble_boundary_faces;
      if (!interface_face_vector_operations.empty())
        assemble_flags |= MeshWorker::assemble_own_interior_faces_once;

      if (assemble_flags)
      {
        MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                              cell_worker,
                              copier,
                              sample_scratch_data,
                              sample_copy_data,
                              assemble_flags,
                              boundary_worker,
                              face_worker);

        system_vector.compress(VectorOperation::add);
      }

      // DEBUGGING!
      // ScratchData scratch = sample_scratch_data;
      // for (const auto cell : dof_handler.active_cell_iterators())
      //   {
      //     CopyData copy = sample_copy_data;

      //     cell_worker(cell, scratch, copy);
      //     copier(copy);
      //   }
    }

    /**
     * Assemble a system matrix and a RHS vector, excluding boundary and internal
     * face contributions.
     *
     * @tparam NumberType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @ Note: Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename MatrixType,
              typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType>
    void
    assemble_system(MatrixType &                         system_matrix,
             VectorType &                         system_vector,
             const AffineConstraints<NumberType> &constraints,
             const DoFHandlerType &               dof_handler,
             const CellQuadratureType &           cell_quadrature) const
    {
      static_assert(DoFHandlerType::dimension == dim,
                    "Dimension is incompatible");
      static_assert(DoFHandlerType::space_dimension == spacedim,
                    "Space dimension is incompatible");

      Assert(this->boundary_face_matrix_operations.empty(), 
             ExcMessage("Assembly with only cell contributions has been selected, "
                        "while there are boundary face contributions in to the "
                        "bilinear form. You should use the other assemble_matrix() "
                        "function that takes in face quadrature as an argument so "
                        "that all contributions are considered."));

      Assert(this->boundary_face_vector_operations.empty(), 
             ExcMessage("Assembly with only cell contributions has been selected, "
                        "while there are boundary face contributions in to the "
                        "linear form. You should use the other assemble_rhs_vector() "
                        "function that takes in face quadrature as an argument so "
                        "that all contributions are considered."));

      Assert(this->interface_face_matrix_operations.empty(), 
             ExcMessage("Assembly with only cell contributions has been selected, "
                        "while there are internal face contributions in to the "
                        "bilinear form. You should use the other assemble_matrix() "
                        "function that takes in face quadrature as an argument so "
                        "that all contributions are considered."));

      Assert(this->interface_face_vector_operations.empty(), 
             ExcMessage("Assembly with only cell contributions has been selected, "
                        "while there are internal face contributions in to the "
                        "linear form. You should use the other assemble_rhs_vector() "
                        "function that takes in face quadrature as an argument so "
                        "that all contributions are considered."));

      using CellIteratorType = typename DoFHandlerType::active_cell_iterator;
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CopyData         = MeshWorker::CopyData<1, 1, 1>;

      // Define a cell worker
      const auto &cell_matrix_operations = this->cell_matrix_operations;
      const auto &cell_vector_operations = this->cell_vector_operations;
      auto cell_worker = CellWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!cell_matrix_operations.empty() || !cell_vector_operations.empty())
      {
          cell_worker            = [&cell_matrix_operations,
                            &cell_vector_operations](const CellIteratorType &cell,
                                                    ScratchData &scratch_data,
                                                    CopyData &   copy_data) {
          const auto &fe_values = scratch_data.reinit(cell);
          copy_data             = CopyData(fe_values.dofs_per_cell);
          copy_data.local_dof_indices[0] = scratch_data.get_local_dof_indices();

          FullMatrix<NumberType> &cell_matrix = copy_data.matrices[0];
          Vector<NumberType> &    cell_vector = copy_data.vectors[0];

          // Perform all operations that contribute to the local cell matrix
          for (const auto &cell_matrix_op : cell_matrix_operations)
            {
              cell_matrix_op(cell_matrix, fe_values);
            }
          // Perform all operations that contribute to the local cell vector
          for (const auto &cell_vector_op : cell_vector_operations)
            {
              cell_vector_op(cell_vector, fe_values);
            }
        };
      }

      auto copier = [&constraints, &system_matrix, &system_vector](
                      const CopyData &copy_data) {
        const FullMatrix<NumberType> &cell_matrix = copy_data.matrices[0];
        const Vector<NumberType> &    cell_vector = copy_data.vectors[0];
        const std::vector<types::global_dof_index> &local_dof_indices =
          copy_data.local_dof_indices[0];

        constraints.distribute_local_to_global(cell_matrix,
                                               cell_vector,
                                               local_dof_indices,
                                               system_matrix,
                                               system_vector);
      };

      const ScratchData sample_scratch_data(dof_handler.get_fe(),
                                            cell_quadrature,
                                            this->get_cell_update_flags());
      const CopyData    sample_copy_data(dof_handler.get_fe().dofs_per_cell);

      MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_nothing;
      if (!cell_matrix_operations.empty() || !cell_vector_operations.empty())
        assemble_flags |= MeshWorker::assemble_own_cells;

      if (assemble_flags)
      {
        MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                              cell_worker,
                              copier,
                              sample_scratch_data,
                              sample_copy_data,
                              assemble_flags);

        if (!cell_matrix_operations.empty())
          system_matrix.compress(VectorOperation::add);
        
        if (!cell_vector_operations.empty())
          system_vector.compress(VectorOperation::add);
      }

      // DEBUGGING!
      // ScratchData scratch = sample_scratch_data;
      // for (const auto cell : dof_handler.active_cell_iterators())
      //   {
      //     CopyData copy = sample_copy_data;

      //     cell_worker(cell, scratch, copy);
      //     copier(copy);
      //   }
    }

    /**
     * Assemble a system matrix and a RHS vector, including boundary and internal
     * face contributions.
     *
     * @tparam NumberType
     * @tparam MatrixType
     * @param system_matrix
     * @param constraints
     *
     * @ Note: Does not reset the matrix, so one can assemble from multiple
     * Assemblers into one matrix. This is useful if you want different
     * quadrature rules for different contributions on the same cell.
     */
    template <typename MatrixType,
              typename VectorType,
              typename DoFHandlerType,
              typename CellQuadratureType,
              typename FaceQuadratureType>
    void
    assemble_system(MatrixType &                         system_matrix,
             VectorType &                         system_vector,
             const AffineConstraints<NumberType> &constraints,
             const DoFHandlerType &               dof_handler,
             const CellQuadratureType &           cell_quadrature,
             const FaceQuadratureType &           face_quadrature) const
    {
      static_assert(DoFHandlerType::dimension == dim,
                    "Dimension is incompatible");
      static_assert(DoFHandlerType::space_dimension == spacedim,
                    "Space dimension is incompatible");

      using CellIteratorType = typename DoFHandlerType::active_cell_iterator;
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CopyData         = MeshWorker::CopyData<1, 1, 1>;

      // Define a cell worker
      const auto &cell_matrix_operations = this->cell_matrix_operations;
      const auto &cell_vector_operations = this->cell_vector_operations;
      auto cell_worker = CellWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!cell_matrix_operations.empty() || !cell_vector_operations.empty())
      {
          cell_worker            = [&cell_matrix_operations,
                            &cell_vector_operations](const CellIteratorType &cell,
                                                    ScratchData &scratch_data,
                                                    CopyData &   copy_data) {
          const auto &fe_values = scratch_data.reinit(cell);
          copy_data             = CopyData(fe_values.dofs_per_cell);
          copy_data.local_dof_indices[0] = scratch_data.get_local_dof_indices();

          FullMatrix<NumberType> &cell_matrix = copy_data.matrices[0];
          Vector<NumberType> &    cell_vector = copy_data.vectors[0];

          // Perform all operations that contribute to the local cell matrix
          for (const auto &cell_matrix_op : cell_matrix_operations)
            {
              cell_matrix_op(cell_matrix, fe_values);
            }
          // Perform all operations that contribute to the local cell vector
          for (const auto &cell_vector_op : cell_vector_operations)
            {
              cell_vector_op(cell_vector, fe_values);
            }
        };
      }

      // Define a boundary worker
      const auto &boundary_face_matrix_operations = this->boundary_face_matrix_operations;
      const auto &boundary_face_vector_operations = this->boundary_face_vector_operations;
      auto boundary_worker = BoundaryWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!boundary_face_matrix_operations.empty() || !boundary_face_vector_operations.empty())
      {
        boundary_worker = [&boundary_face_vector_operations](const CellIteratorType &cell,
                                              const unsigned int face,
                                              ScratchData &scratch_data,
                                              CopyData &   copy_data) 
        {
          Assert((cell->face(face)->at_boundary()), ExcMessage("Cell face is not at the boundary."));

          const auto &fe_values = scratch_data.reinit(cell);
          const auto &fe_face_values = scratch_data.reinit(cell,face);
          copy_data             = CopyData(fe_values.dofs_per_cell);
          copy_data.local_dof_indices[0] = scratch_data.get_local_dof_indices();

          Vector<NumberType> &    cell_vector = copy_data.vectors[0];


          // Perform all operations that contribute to the local cell matrix
          for (const auto &boundary_face_matrix_op : boundary_face_matrix_operations)
            {
              boundary_face_matrix_op(cell_vector, fe_values, fe_face_values, face);
            }
          // Perform all operations that contribute to the local cell vector
          for (const auto &boundary_face_vector_op : boundary_face_vector_operations)
            {
              boundary_face_vector_op(cell_vector, fe_values, fe_face_values, face);
            }
        };
      }

      // Define a face / interface worker
      const auto &interface_face_matrix_operations = this->interface_face_matrix_operations;
      const auto &interface_face_vector_operations = this->interface_face_vector_operations;
      auto face_worker = FaceWorkerType<CellIteratorType,ScratchData,CopyData>();
      if (!interface_face_matrix_operations.empty() || !interface_face_vector_operations.empty())
      {
        // interface_matrix_operations
        // interface_vector_operations
        AssertThrow(false, ExcMessage("Internal face cell matrix and vector contributions have not yet been implemented."));
      }

      auto copier = [&constraints, &system_matrix, &system_vector](
                      const CopyData &copy_data) {
        const FullMatrix<NumberType> &cell_matrix = copy_data.matrices[0];
        const Vector<NumberType> &    cell_vector = copy_data.vectors[0];
        const std::vector<types::global_dof_index> &local_dof_indices =
          copy_data.local_dof_indices[0];

        constraints.distribute_local_to_global(cell_matrix,
                                               cell_vector,
                                               local_dof_indices,
                                               system_matrix,
                                               system_vector);
      };

      const ScratchData sample_scratch_data(dof_handler.get_fe(),
                                            cell_quadrature,
                                            this->get_cell_update_flags(),
                                            face_quadrature,
                                            this->get_face_update_flags());
      const CopyData    sample_copy_data(dof_handler.get_fe().dofs_per_cell);

      MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_nothing;
      if (!cell_matrix_operations.empty() || !cell_vector_operations.empty())
        assemble_flags |= MeshWorker::assemble_own_cells;
      if (!boundary_face_matrix_operations.empty() || !boundary_face_vector_operations.empty())
        assemble_flags |= MeshWorker::assemble_boundary_faces;
      if (!interface_face_matrix_operations.empty() || !interface_face_vector_operations.empty())
        assemble_flags |= MeshWorker::assemble_own_interior_faces_once;

      if (assemble_flags)
      {
        MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                              cell_worker,
                              copier,
                              sample_scratch_data,
                              sample_copy_data,
                              assemble_flags);

        if (!cell_matrix_operations.empty() || !boundary_face_matrix_operations.empty() || !interface_face_matrix_operations.empty())
          system_matrix.compress(VectorOperation::add);
        
        if (!cell_vector_operations.empty() || !boundary_face_vector_operations.empty() || !interface_face_vector_operations.empty())
          system_vector.compress(VectorOperation::add);
      }

      // DEBUGGING!
      // ScratchData scratch = sample_scratch_data;
      // for (const auto cell : dof_handler.active_cell_iterators())
      //   {
      //     CopyData copy = sample_copy_data;

      //     cell_worker(cell, scratch, copy);
      //     copier(copy);
      //   }
    }
  };

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_assembler_h
