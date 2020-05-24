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

#include <deal.II/base/types.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

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

    // Valid for cell and face assembly
    template <typename NumberType,
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
      for (const unsigned int i : fe_values_dofs.dof_indices())
        for (const unsigned int j : fe_values_dofs.dof_indices())
          for (const unsigned int q :
               fe_values_q_points.quadrature_point_indices())
            {
              cell_matrix(i, j) +=
                (shapes_test[i][q] * values_functor[q] * shapes_trial[j][q]) *
                JxW[q];
            }
    }

    template <typename NumberType,
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
      assemble_cell_matrix_contribution(cell_matrix,
                                        fe_values,
                                        fe_values,
                                        shapes_test,
                                        values_functor,
                                        shapes_trial,
                                        JxW);
    }

  } // namespace internal



  template <int dim, int spacedim = dim, typename NumberType = double>
  class AssemblerBase
  {
  public:
    using StringOperation = std::function<std::string(void)>;

    using CellMatrixOperation =
      std::function<void(FullMatrix<NumberType> &           cell_matrix,
                         const FEValuesBase<dim, spacedim> &fe_values)>;
    using CellVectorOperation =
      std::function<void(Vector<NumberType> &               cell_vector,
                         const FEValuesBase<dim, spacedim> &fe_values)>;
    using BoundaryMatrixOperation =
      std::function<void(FullMatrix<NumberType> &           cell_matrix,
                         const FEValuesBase<dim, spacedim> &fe_values,
                         const FEValuesBase<dim, spacedim> &fe_face_values)>;
    using BoundaryVectorOperation =
      std::function<void(Vector<NumberType> &               cell_vector,
                         const FEValuesBase<dim, spacedim> &fe_values,
                         const FEValuesBase<dim, spacedim> &fe_face_values)>;


    virtual ~AssemblerBase() = default;


    template <typename UnaryOpType,
              typename = typename std::enable_if<
                is_symbolic_volume_integral<UnaryOpType>::value>::type>
    AssemblerBase &
    operator+=(const UnaryOpType &volume_integral)
    {
      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      add_cell_operation(volume_integral);

      // TODO: Add to as_ascii() and as_latex()

      return *this;
    }

    // TODO:
    std::string
    as_ascii() const
    {
      std::string output = "0 = ";
      for (unsigned int i = 0; i < as_ascii_operations.size(); ++i)
        {
          output += as_ascii_operations[i]();
          if (i + 1 < as_ascii_operations.size())
            output += " + ";
        }
      return output;
    }

    std::string
    as_latex() const
    {
      std::string output = "0 = ";
      for (unsigned int i = 0; i < as_latex_operations.size(); ++i)
        {
          output += as_latex_operations[i]();
          if (i + 1 < as_latex_operations.size())
            output += " + ";
        }
      return output;
    }

  protected:
    explicit AssemblerBase()
      : cell_update_flags(update_default)
      , face_update_flags(update_default)
    {}

    template <typename UnaryOpVolumeIntegral,
              typename = typename std::enable_if<is_bilinear_form<
                typename UnaryOpVolumeIntegral::IntegrandType>::value>::type>
    void
    add_cell_operation(const UnaryOpVolumeIntegral &volume_integral)
    {
      // We need to update the flags that need to be set for
      // cell operations. The flags from the composite operation
      // that composes the integrand will be bubbled down to the
      // integral itself.
      cell_update_flags |= volume_integral.get_update_flags();

      // Augment the composition of the operation
      as_ascii_operations.push_back(
        [&volume_integral]() { return volume_integral.as_ascii(); });
      as_latex_operations.push_back(
        [&volume_integral]() { return volume_integral.as_latex(); });

      // Extract some information about the form that we'll be
      // constructing and integrating
      const auto &form = volume_integral.get_integrand();
      static_assert(
        is_bilinear_form<typename std::decay<decltype(form)>::type>::value,
        "Incompatible integrand type.");

      const auto test_space_op  = form.get_test_space_operation();
      const auto functor        = form.get_functor();
      const auto trial_space_op = form.get_trial_space_operation();

      using TestSpaceOp  = decltype(test_space_op);
      using Functor      = decltype(functor);
      using TrialSpaceOp = decltype(trial_space_op);

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
          return;

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
              shapes_trial[k][q] =
                trial_space_op.template operator()<NumberType>(fe_values, k, q);
              shapes_test[k][q] =
                test_space_op.template operator()<NumberType>(fe_values, k, q);
            }

        internal::assemble_cell_matrix_contribution(cell_matrix,
                                                    fe_values,
                                                    shapes_test,
                                                    values_functor,
                                                    shapes_trial,
                                                    JxW);
      };
      cell_matrix_operations.emplace_back(f);
    }

    // template<typename UnaryOpVolumeIntegral,
    // typename = typename std::enable_if<is_linear_form<typename
    // UnaryOpVolumeIntegral::IntegrandType>::value>::type> void
    // add_cell_operation (const UnaryOpVolumeIntegral &volume_integral)
    // {
    //   // cell_operations.emplace_back();
    // }

    UpdateFlags
    get_cell_update_flags() const
    {
      return cell_update_flags;
    }

    std::vector<StringOperation> as_ascii_operations;
    std::vector<StringOperation> as_latex_operations;

    UpdateFlags                      cell_update_flags;
    std::vector<CellMatrixOperation> cell_matrix_operations;
    std::vector<CellVectorOperation> cell_vector_operations;

    UpdateFlags                          face_update_flags;
    std::vector<BoundaryMatrixOperation> boundary_matrix_operations;
    std::vector<BoundaryVectorOperation> boundary_vector_operations;
    // std::vector<BoundaryWorkerType> interface_matrix_operations;
    // std::vector<BoundaryWorkerType> interface_vector_operations;
  };



  // TODO: Put in another header
  template <int dim, int spacedim = dim, typename NumberType = double>
  class MatrixBasedAssembler : public AssemblerBase<dim, spacedim, NumberType>
  {
  public:
    explicit MatrixBasedAssembler()
      : AssemblerBase<dim, spacedim, NumberType>(){};

    /**
     * @brief
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
    assemble(MatrixType &                         system_matrix,
             const AffineConstraints<NumberType> &constraints,
             const DoFHandlerType &               dof_handler,
             const CellQuadratureType &           cell_quadrature) const
    {
      static_assert(DoFHandlerType::dimension == dim,
                    "Dimension is incompatible");
      static_assert(DoFHandlerType::space_dimension == spacedim,
                    "Space dimension is incompatible");

      using CellIteratorType = typename DoFHandlerType::active_cell_iterator;
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CopyData         = MeshWorker::CopyData<1, 1, 1>;

      //       using CellWorkerType     = std::function<void(const
      //       CellIteratorType &,
      // ScratchData &, CopyData &)>;
      // using BoundaryWorkerType =
      // std::function<void(
      //     const CellIteratorType &, const unsigned int &, ScratchData &,
      //     CopyData
      //     &)>;
      // using InterfaceWorkerType = std::function< void(const
      // CellIteratorBaseType &, const unsigned int, const unsigned int, const
      // CellIteratorBaseType &, const unsigned int, const unsigned int,
      // ScratchData &, CopyData &)>;

      const auto &cell_matrix_operations = this->cell_matrix_operations;
      auto cell_worker = [&cell_matrix_operations](const CellIteratorType &cell,
                                                   ScratchData &scratch_data,
                                                   CopyData &   copy_data) {
        const auto &fe_values          = scratch_data.reinit(cell);
        copy_data                      = CopyData(fe_values.dofs_per_cell);
        copy_data.local_dof_indices[0] = scratch_data.get_local_dof_indices();

        FullMatrix<NumberType> &cell_matrix = copy_data.matrices[0];
        Vector<NumberType> &    cell_vector = copy_data.vectors[0];

        // Perform all operations that contribute to the local cell matrix
        for (const auto &cell_matrix_op : cell_matrix_operations)
          {
            cell_matrix_op(cell_matrix, fe_values);
          }

        // cell_vector_operations
        // boundary_matrix_operations
        // boundary_vector_operations
        // interface_matrix_operations
        // interface_vector_operations
      };

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

      MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                            cell_worker,
                            copier,
                            sample_scratch_data,
                            sample_copy_data,
                            MeshWorker::assemble_own_cells);

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