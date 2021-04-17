/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Jean-Paul Pelteret, 2021
 * Based on step-15, authored by Sven Wetterauer, University of Heidelberg, 2012
 */


// @sect3{Include files}

// The first few files have already been covered in previous examples and will
// thus not be further commented on.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>


#include <fstream>
#include <iostream>

// We will use adaptive mesh refinement between Newton iterations. To do so,
// we need to be able to work with a solution on the new mesh, although it was
// computed on the old one. The SolutionTransfer class transfers the solution
// from the old to the new mesh:

#include <deal.II/numerics/solution_transfer.h>

// We then open a namespace for this program and import everything from the
// dealii namespace into it, as in previous programs:
namespace Step72
{
  using namespace dealii;

  // @sect3{The <code>MinimalSurfaceProblemParameters</code> class}

  class MinimalSurfaceProblemParameters : public ParameterAcceptor
  {
  public:
    MinimalSurfaceProblemParameters();

    // Selection for the formulation and corresponding AD framework to be used.
    //   formulation = 0 : Unassisted implementation (full hand linearization)
    //   formulation = 1 : Automated linearization of the finite element
    //                     residual
    //   formulation = 2 : Automated computation of finite element
    //                     residual and linearization using a
    //                     variational formulation
    int formulation = 0;

    // The maximum acceptable tolerance for the linear system residual.
    double tolerance = 1e-2;
  };


  MinimalSurfaceProblemParameters::MinimalSurfaceProblemParameters()
    : ParameterAcceptor("Minimal Surface Problem/")
  {
    add_parameter(
      "Formulation", formulation, "", this->prm, Patterns::Integer(0, 2));
    add_parameter("Tolerance", tolerance, "", this->prm, Patterns::Double(0.0));
  }



  // @sect3{The <code>MinimalSurfaceProblem</code> class template}

  // The class template is basically the same as in step-6.  Three additions
  // are made:
  // - There are two solution vectors, one for the Newton update
  //   $\delta u^n$, and one for the current iterate $u^n$.
  // - The <code>setup_system</code> function takes an argument that denotes
  //   whether this is the first time it is called or not. The difference is
  //   that the first time around we need to distribute the degrees of freedom
  //   and set the solution vector for $u^n$ to the correct size. The following
  //   times, the function is called after we have already done these steps as
  //   part of refining the mesh in <code>refine_mesh</code>.
  // - We then also need new functions: <code>set_boundary_values()</code>
  //   takes care of setting the boundary values on the solution vector
  //   correctly, as discussed at the end of the
  //   introduction. <code>compute_residual()</code> is a function that computes
  //   the norm of the nonlinear (discrete) residual. We use this function to
  //   monitor convergence of the Newton iteration. The function takes a step
  //   length $\alpha^n$ as argument to compute the residual of $u^n + \alpha^n
  //   \; \delta u^n$. This is something one typically needs for step length
  //   control, although we will not use this feature here. Finally,
  //   <code>determine_step_length()</code> computes the step length $\alpha^n$
  //   in each Newton iteration. As discussed in the introduction, we here use a
  //   fixed step length and leave implementing a better strategy as an
  //   exercise.

  template <int dim>
  class MinimalSurfaceProblem
  {
  public:
    MinimalSurfaceProblem();

    void run(const int formulation, const double tolerance);

  private:
    void   setup_system(const bool initial_step);
    void   assemble_system_unassisted();
    void   assemble_system_with_residual_linearization();
    void   assemble_system_using_energy_functional();
    void   solve();
    void   refine_mesh();
    void   set_boundary_values();
    double compute_residual(const double alpha) const;
    double determine_step_length() const;
    void   output_results(const unsigned int refinement_cycle) const;

    Triangulation<dim> triangulation;

    DoFHandler<dim> dof_handler;
    FE_Q<dim>       fe;
    QGauss<dim>     quadrature_formula;

    AffineConstraints<double> hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> current_solution;
    Vector<double> newton_update;
    Vector<double> system_rhs;
  };

  // @sect3{Boundary condition}

  // The boundary condition is implemented just like in step-4.  It is chosen
  // as $g(x,y)=\sin(2 \pi (x+y))$:

  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };


  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> &p,
                                    const unsigned int /*component*/) const
  {
    return std::sin(2 * numbers::PI * (p[0] + p[1]));
  }

  // @sect3{The <code>MinimalSurfaceProblem</code> class implementation}

  // @sect4{MinimalSurfaceProblem::MinimalSurfaceProblem}

  // The constructor and destructor of the class are the same as in the first
  // few tutorials.

  template <int dim>
  MinimalSurfaceProblem<dim>::MinimalSurfaceProblem()
    : dof_handler(triangulation)
    , fe(2)
    , quadrature_formula(fe.degree + 1)
  {}

  // @sect4{MinimalSurfaceProblem::setup_system}

  // As always in the setup-system function, we setup the variables of the
  // finite element method. There are same differences to step-6, because
  // there we start solving the PDE from scratch in every refinement cycle
  // whereas here we need to take the solution from the previous mesh onto the
  // current mesh. Consequently, we can't just reset solution vectors. The
  // argument passed to this function thus indicates whether we can
  // distributed degrees of freedom (plus compute constraints) and set the
  // solution vector to zero or whether this has happened elsewhere already
  // (specifically, in <code>refine_mesh()</code>).

  template <int dim>
  void MinimalSurfaceProblem<dim>::setup_system(const bool initial_step)
  {
    if (initial_step)
      {
        dof_handler.distribute_dofs(fe);
        current_solution.reinit(dof_handler.n_dofs());

        hanging_node_constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                hanging_node_constraints);
        hanging_node_constraints.close();
      }


    // The remaining parts of the function are the same as in step-6.

    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    hanging_node_constraints.condense(dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
  }

  // @sect4{MinimalSurfaceProblem::assemble_system}

  // This function does the same as in the previous tutorials except that now,
  // of course, the matrix and right hand side functions depend on the
  // previous iteration's solution. As discussed in the introduction, we need
  // to use zero boundary values for the Newton updates; we compute them at
  // the end of this function.
  //
  // The top of the function contains the usual boilerplate code, setting up
  // the objects that allow us to evaluate shape functions at quadrature
  // points and temporary storage locations for the local matrices and
  // vectors, as well as for the gradients of the previous solution at the
  // quadrature points. We then start the loop over all cells:


  template <int dim>
  void MinimalSurfaceProblem<dim>::assemble_system_unassisted()
  {
    system_matrix = 0;
    system_rhs    = 0;

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    using ScratchData      = MeshWorker::ScratchData<dim>;
    using CopyData         = MeshWorker::CopyData<1, 1, 1>;
    using CellIteratorType = decltype(dof_handler.begin_active());

    const ScratchData sample_scratch_data(fe,
                                          quadrature_formula,
                                          update_gradients |
                                            update_quadrature_points |
                                            update_JxW_values);
    const CopyData    sample_copy_data(dofs_per_cell);

    auto cell_worker = [dofs_per_cell, this](const CellIteratorType &cell,
                                             ScratchData &scratch_data,
                                             CopyData &   copy_data) {
      const auto &fe_values = scratch_data.reinit(cell);

      FullMatrix<double> &                  cell_matrix = copy_data.matrices[0];
      Vector<double> &                      cell_rhs    = copy_data.vectors[0];
      std::vector<types::global_dof_index> &local_dof_indices =
        copy_data.local_dof_indices[0];
      cell->get_dof_indices(local_dof_indices);

      // For the assembly of the linear system, we have to obtain the values
      // of the previous solution's gradients at the quadrature
      // points. There is a standard way of doing this: the
      // FEValues::get_function_gradients function takes a vector that
      // represents a finite element field defined on a DoFHandler, and
      // evaluates the gradients of this field at the quadrature points of the
      // cell with which the FEValues object has last been reinitialized.
      // The values of the gradients at all quadrature points are then written
      // into the second argument:
      std::vector<Tensor<1, dim>> old_solution_gradients(
        fe_values.n_quadrature_points);
      fe_values.get_function_gradients(current_solution,
                                       old_solution_gradients);

      // With this, we can then do the integration loop over all quadrature
      // points and shape functions.  Having just computed the gradients of
      // the old solution in the quadrature points, we are able to compute
      // the coefficients $a_{n}$ in these points.  The assembly of the
      // system itself then looks similar to what we always do with the
      // exception of the nonlinear terms, as does copying the results from
      // the local objects into the global ones:
      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          const double coeff = 1.0 / std::sqrt(1 + old_solution_gradients[q] *
                                                     old_solution_gradients[q]);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) +=
                  (((fe_values.shape_grad(i, q)      // ((\nabla \phi_i
                     * coeff                         //   * a_n
                     * fe_values.shape_grad(j, q))   //   * \nabla \phi_j)
                    -                                //  -
                    (fe_values.shape_grad(i, q)      //  (\nabla \phi_i
                     * coeff * coeff * coeff         //   * a_n^3
                     * (fe_values.shape_grad(j, q)   //   * (\nabla \phi_j
                        * old_solution_gradients[q]) //      * \nabla u_n)
                     * old_solution_gradients[q]))   //   * \nabla u_n)))
                   * fe_values.JxW(q));              // * dx

              cell_rhs(i) -= (fe_values.shape_grad(i, q)  // \nabla \phi_i
                              * coeff                     // * a_n
                              * old_solution_gradients[q] // * u_n
                              * fe_values.JxW(q));        // * dx
            }
        }
    };

    auto copier = [dofs_per_cell, this](const CopyData &copy_data) {
      const FullMatrix<double> &cell_matrix = copy_data.matrices[0];
      const Vector<double> &    cell_rhs    = copy_data.vectors[0];
      const std::vector<types::global_dof_index> &local_dof_indices =
        copy_data.local_dof_indices[0];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    };

    MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                          cell_worker,
                          copier,
                          sample_scratch_data,
                          sample_copy_data,
                          MeshWorker::assemble_own_cells);

    // Finally, we remove hanging nodes from the system and apply zero
    // boundary values to the linear system that defines the Newton updates
    // $\delta u^n$:
    hanging_node_constraints.condense(system_matrix);
    hanging_node_constraints.condense(system_rhs);

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       newton_update,
                                       system_rhs);
  }


  template <int dim>
  void MinimalSurfaceProblem<dim>::assemble_system_with_residual_linearization()
  {
    system_matrix = 0;
    system_rhs    = 0;

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    using ScratchData      = MeshWorker::ScratchData<dim>;
    using CopyData         = MeshWorker::CopyData<1, 1, 1>;
    using CellIteratorType = decltype(dof_handler.begin_active());

    const ScratchData sample_scratch_data(fe,
                                          quadrature_formula,
                                          update_gradients |
                                            update_quadrature_points |
                                            update_JxW_values);
    const CopyData    sample_copy_data(dofs_per_cell);

    // Define the AD data structures that we'll be using.
    // In this case, we choose the helper class that will automatically compute
    // the linearization of the finite element residual using Sacado forward
    // automatic differentiation types. These number types can be used to
    // compute first derivatives only. This is exactly what we want, because we
    // know that we'll only be linearizing the residual, which implies only
    // computing first-order derivatives.
    using ADHelper = Differentiation::AD::ResidualLinearization<
      Differentiation::AD::NumberTypes::sacado_dfad,
      double>;
    using ADNumberType = typename ADHelper::ad_type;

    // We need an extractor to help interpret and retrieve some data from
    // the AD helper.
    const FEValuesExtractors::Scalar u_fe(0);

    auto cell_worker = [&, this](const CellIteratorType &cell,
                                 ScratchData &           scratch_data,
                                 CopyData &              copy_data) {
      const auto &fe_values = scratch_data.reinit(cell);

      FullMatrix<double> &                  cell_matrix = copy_data.matrices[0];
      Vector<double> &                      cell_rhs    = copy_data.vectors[0];
      std::vector<types::global_dof_index> &local_dof_indices =
        copy_data.local_dof_indices[0];
      cell->get_dof_indices(local_dof_indices);

      // Create and initialize an instance of the helper class.
      const unsigned int n_independent_variables = local_dof_indices.size();
      const unsigned int n_dependent_variables   = dofs_per_cell;
      ADHelper ad_helper(n_independent_variables, n_dependent_variables);

      // First, we set the values for all DoFs.
      ad_helper.register_dof_values(current_solution, local_dof_indices);

      // Then we get the complete set of degree of freedom values as
      // represented by auto-differentiable numbers. The operations
      // performed with these variables are tracked by the AD library
      // from this point until the object goes out of scope.
      const std::vector<ADNumberType> &dof_values_ad =
        ad_helper.get_sensitive_dof_values();

      // Then we do some problem specific tasks, the first being to
      // compute all values, gradients, etc. based on sensitive AD DoF
      // values. Here we are fetching the solution gradients at each
      // quadrature point. We'll be linearizing around this solution.
      // Solution gradients are now sensitive to the values of the
      // degrees of freedom.
      std::vector<Tensor<1, dim, ADNumberType>> old_solution_gradients(
        fe_values.n_quadrature_points);
      fe_values[u_fe].get_function_gradients_from_local_dof_values(
        dof_values_ad, old_solution_gradients);

      // This variable stores the cell residual vector contributions.
      // IMPORTANT: Note that each entry is hand-initialized with a value
      // of zero. This is a highly recommended practise, as some AD
      // numbers appear not to safely initialize their internal data
      // structures.
      std::vector<ADNumberType> residual_ad(n_dependent_variables,
                                            ADNumberType(0.0));
      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          // The coefficient now encodes its dependence on the FE DoF values.
          const ADNumberType coeff =
            1.0 / std::sqrt(1.0 + old_solution_gradients[q] *
                                    old_solution_gradients[q]);

          // Finally we may assemble the components of the residual vector.
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              residual_ad[i] += (fe_values.shape_grad(i, q)   // \nabla \phi_i
                                 * coeff                      // * a_n
                                 * old_solution_gradients[q]) // * u_n
                                * fe_values.JxW(q);           // * dx
            }
        }

      // Register the definition of the cell residual
      ad_helper.register_residual_vector(residual_ad);

      // Compute the residual values and their Jacobian at the
      // evaluation point
      ad_helper.compute_residual(cell_rhs);
      cell_rhs *= -1.0; // RHS = - residual
      ad_helper.compute_linearization(cell_matrix);
    };

    auto copier = [dofs_per_cell, this](const CopyData &copy_data) {
      const FullMatrix<double> &cell_matrix = copy_data.matrices[0];
      const Vector<double> &    cell_rhs    = copy_data.vectors[0];
      const std::vector<types::global_dof_index> &local_dof_indices =
        copy_data.local_dof_indices[0];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    };

    MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                          cell_worker,
                          copier,
                          sample_scratch_data,
                          sample_copy_data,
                          MeshWorker::assemble_own_cells);

    // Finally, we remove hanging nodes from the system and apply zero
    // boundary values to the linear system that defines the Newton updates
    // $\delta u^n$:
    hanging_node_constraints.condense(system_matrix);
    hanging_node_constraints.condense(system_rhs);

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       newton_update,
                                       system_rhs);
  }


  template <int dim>
  void MinimalSurfaceProblem<dim>::assemble_system_using_energy_functional()
  {
    system_matrix = 0;
    system_rhs    = 0;

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    using ScratchData      = MeshWorker::ScratchData<dim>;
    using CopyData         = MeshWorker::CopyData<1, 1, 1>;
    using CellIteratorType = decltype(dof_handler.begin_active());

    const ScratchData sample_scratch_data(fe,
                                          quadrature_formula,
                                          update_gradients |
                                            update_quadrature_points |
                                            update_JxW_values);
    const CopyData    sample_copy_data(dofs_per_cell);

    // Define the AD data structures that we'll be using.
    // In this case, we choose the helper class that will automatically compute
    // both the residual and its linearization from the cell contribution to an
    // energy functional using Sacado forward automatic differentiation types.
    // The selected number types can be used to compute both first and
    // second derivatives. We require this, as the residual defined as the
    // sensitivity of the potential energy with respect to the DoF values (i.e.
    // its gradient). We'll then need to linearize the residual, implying that
    // second derivatives of the potential energy must be computed.
    using ADHelper = Differentiation::AD::EnergyFunctional<
      Differentiation::AD::NumberTypes::sacado_dfad_dfad,
      double>;
    using ADNumberType = typename ADHelper::ad_type;

    // We need an extractor to help interpret and retrieve some data from
    // the AD helper.
    const FEValuesExtractors::Scalar u_fe(0);

    auto cell_worker = [&, this](const CellIteratorType &cell,
                                 ScratchData &           scratch_data,
                                 CopyData &              copy_data) {
      const auto &fe_values = scratch_data.reinit(cell);

      FullMatrix<double> &                  cell_matrix = copy_data.matrices[0];
      Vector<double> &                      cell_rhs    = copy_data.vectors[0];
      std::vector<types::global_dof_index> &local_dof_indices =
        copy_data.local_dof_indices[0];
      cell->get_dof_indices(local_dof_indices);

      // Create and initialize an instance of the helper class.
      const unsigned int n_independent_variables = local_dof_indices.size();
      ADHelper           ad_helper(n_independent_variables);

      // First, we set the values for all DoFs.
      ad_helper.register_dof_values(current_solution, local_dof_indices);

      // Then we get the complete set of degree of freedom values as
      // represented by auto-differentiable numbers. The operations
      // performed with these variables are tracked by the AD library
      // from this point until the object goes out of scope.
      const std::vector<ADNumberType> &dof_values_ad =
        ad_helper.get_sensitive_dof_values();

      // Then we do some problem specific tasks, the first being to
      // compute all values, gradients, etc. based on sensitive AD DoF
      // values. Here we are fetching the solution gradients at each
      // quadrature point. We'll be linearizing around this solution.
      // Again, the solution gradients are sensitive to the values of the
      // degrees of freedom.
      std::vector<Tensor<1, dim, ADNumberType>> old_solution_gradients(
        fe_values.n_quadrature_points);
      fe_values[u_fe].get_function_gradients_from_local_dof_values(
        dof_values_ad, old_solution_gradients);

      // This variable stores the cell total energy.
      // IMPORTANT: Note that it is hand-initialized with a value of
      // zero. This is a highly recommended practise, as some AD numbers
      // appear not to safely initialize their internal data structures.
      ADNumberType energy_ad = ADNumberType(0.0);

      // Compute the cell total energy = (internal + external) energies
      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          // We compute the configuration-dependent material energy, namely
          // the integrand for the area of the surface to be minimized...
          const ADNumberType psi = std::sqrt(1.0 + old_solution_gradients[q] *
                                                     old_solution_gradients[q]);

          // ... which we then integrate to increment the total cell energy.
          energy_ad += psi * fe_values.JxW(q);
        }

      // Register the definition of the total cell energy
      ad_helper.register_energy_functional(energy_ad);

      // Compute the residual values and their Jacobian at the
      // evaluation point
      ad_helper.compute_residual(cell_rhs);
      cell_rhs *= -1.0; // RHS = - residual
      ad_helper.compute_linearization(cell_matrix);
    };

    auto copier = [dofs_per_cell, this](const CopyData &copy_data) {
      const FullMatrix<double> &cell_matrix = copy_data.matrices[0];
      const Vector<double> &    cell_rhs    = copy_data.vectors[0];
      const std::vector<types::global_dof_index> &local_dof_indices =
        copy_data.local_dof_indices[0];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    };

    MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
                          cell_worker,
                          copier,
                          sample_scratch_data,
                          sample_copy_data,
                          MeshWorker::assemble_own_cells);

    // Finally, we remove hanging nodes from the system and apply zero
    // boundary values to the linear system that defines the Newton updates
    // $\delta u^n$:
    hanging_node_constraints.condense(system_matrix);
    hanging_node_constraints.condense(system_rhs);

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       newton_update,
                                       system_rhs);
  }


  // @sect4{MinimalSurfaceProblem::solve}

  // The solve function is the same as always. At the end of the solution
  // process we update the current solution by setting
  // $u^{n+1}=u^n+\alpha^n\;\delta u^n$.
  template <int dim>
  void MinimalSurfaceProblem<dim>::solve()
  {
    SolverControl            solver_control(system_rhs.size(),
                                 system_rhs.l2_norm() * 1e-6);
    SolverCG<Vector<double>> solver(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    solver.solve(system_matrix, newton_update, system_rhs, preconditioner);

    hanging_node_constraints.distribute(newton_update);

    const double alpha = determine_step_length();
    current_solution.add(alpha, newton_update);
  }


  // @sect4{MinimalSurfaceProblem::refine_mesh}

  // The first part of this function is the same as in step-6... However,
  // after refining the mesh we have to transfer the old solution to the new
  // one which we do with the help of the SolutionTransfer class. The process
  // is slightly convoluted, so let us describe it in detail:
  template <int dim>
  void MinimalSurfaceProblem<dim>::refine_mesh()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      current_solution,
      estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.03);

    // Then we need an additional step: if, for example, you flag a cell that
    // is once more refined than its neighbor, and that neighbor is not
    // flagged for refinement, we would end up with a jump of two refinement
    // levels across a cell interface.  To avoid these situations, the library
    // will silently also have to refine the neighbor cell once. It does so by
    // calling the Triangulation::prepare_coarsening_and_refinement function
    // before actually doing the refinement and coarsening.  This function
    // flags a set of additional cells for refinement or coarsening, to
    // enforce rules like the one-hanging-node rule.  The cells that are
    // flagged for refinement and coarsening after calling this function are
    // exactly the ones that will actually be refined or coarsened. Usually,
    // you don't have to do this by hand
    // (Triangulation::execute_coarsening_and_refinement does this for
    // you). However, we need to initialize the SolutionTransfer class and it
    // needs to know the final set of cells that will be coarsened or refined
    // in order to store the data from the old mesh and transfer to the new
    // one. Thus, we call the function by hand:
    triangulation.prepare_coarsening_and_refinement();

    // With this out of the way, we initialize a SolutionTransfer object with
    // the present DoFHandler and attach the solution vector to it, followed
    // by doing the actual refinement and distribution of degrees of freedom
    // on the new mesh
    SolutionTransfer<dim> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(current_solution);

    triangulation.execute_coarsening_and_refinement();

    dof_handler.distribute_dofs(fe);

    // Finally, we retrieve the old solution interpolated to the new
    // mesh. Since the SolutionTransfer function does not actually store the
    // values of the old solution, but rather indices, we need to preserve the
    // old solution vector until we have gotten the new interpolated
    // values. Thus, we have the new values written into a temporary vector,
    // and only afterwards write them into the solution vector object. Once we
    // have this solution we have to make sure that the $u^n$ we now have
    // actually has the correct boundary values. As explained at the end of
    // the introduction, this is not automatically the case even if the
    // solution before refinement had the correct boundary values, and so we
    // have to explicitly make sure that it now has:
    Vector<double> tmp(dof_handler.n_dofs());
    solution_transfer.interpolate(current_solution, tmp);
    current_solution = tmp;

    set_boundary_values();

    // On the new mesh, there are different hanging nodes, which we have to
    // compute again. To ensure there are no hanging nodes of the old mesh in
    // the object, it's first cleared.  To be on the safe side, we then also
    // make sure that the current solution's vector entries satisfy the
    // hanging node constraints (see the discussion in the documentation of
    // the SolutionTransfer class for why this is necessary):
    hanging_node_constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();

    hanging_node_constraints.distribute(current_solution);

    // We end the function by updating all the remaining data structures,
    // indicating to <code>setup_dofs()</code> that this is not the first
    // go-around and that it needs to preserve the content of the solution
    // vector:
    setup_system(false);
  }



  // @sect4{MinimalSurfaceProblem::set_boundary_values}

  // The next function ensures that the solution vector's entries respect the
  // boundary values for our problem.  Having refined the mesh (or just
  // started computations), there might be new nodal points on the
  // boundary. These have values that are simply interpolated from the
  // previous mesh (or are just zero), instead of the correct boundary
  // values. This is fixed up by setting all boundary nodes explicit to the
  // right value:
  template <int dim>
  void MinimalSurfaceProblem<dim>::set_boundary_values()
  {
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryValues<dim>(),
                                             boundary_values);
    for (auto &boundary_value : boundary_values)
      current_solution(boundary_value.first) = boundary_value.second;
  }


  // @sect4{MinimalSurfaceProblem::compute_residual}

  // In order to monitor convergence, we need a way to compute the norm of the
  // (discrete) residual, i.e., the norm of the vector
  // $\left<F(u^n),\varphi_i\right>$ with $F(u)=-\nabla \cdot \left(
  // \frac{1}{\sqrt{1+|\nabla u|^{2}}}\nabla u \right)$ as discussed in the
  // introduction. It turns out that (although we don't use this feature in
  // the current version of the program) one needs to compute the residual
  // $\left<F(u^n+\alpha^n\;\delta u^n),\varphi_i\right>$ when determining
  // optimal step lengths, and so this is what we implement here: the function
  // takes the step length $\alpha^n$ as an argument. The original
  // functionality is of course obtained by passing a zero as argument.
  //
  // In the function below, we first set up a vector for the residual, and
  // then a vector for the evaluation point $u^n+\alpha^n\;\delta u^n$. This
  // is followed by the same boilerplate code we use for all integration
  // operations:
  template <int dim>
  double MinimalSurfaceProblem<dim>::compute_residual(const double alpha) const
  {
    Vector<double> residual(dof_handler.n_dofs());

    Vector<double> evaluation_point(dof_handler.n_dofs());
    evaluation_point = current_solution;
    evaluation_point.add(alpha, newton_update);

    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_gradients | update_quadrature_points |
                              update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double>              cell_residual(dofs_per_cell);
    std::vector<Tensor<1, dim>> gradients(n_q_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_residual = 0;
        fe_values.reinit(cell);

        // The actual computation is much as in
        // <code>assemble_system()</code>. We first evaluate the gradients of
        // $u^n+\alpha^n\,\delta u^n$ at the quadrature points, then compute
        // the coefficient $a_n$, and then plug it all into the formula for
        // the residual:
        fe_values.get_function_gradients(evaluation_point, gradients);


        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double coeff =
              1. / std::sqrt(1 + gradients[q] * gradients[q]);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_residual(i) -= (fe_values.shape_grad(i, q) // \nabla \phi_i
                                   * coeff                    // * a_n
                                   * gradients[q]             // * u_n
                                   * fe_values.JxW(q));       // * dx
          }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          residual(local_dof_indices[i]) += cell_residual(i);
      }

    // At the end of this function we also have to deal with the hanging node
    // constraints and with the issue of boundary values. With regard to the
    // latter, we have to set to zero the elements of the residual vector for
    // all entries that correspond to degrees of freedom that sit at the
    // boundary. The reason is that because the value of the solution there is
    // fixed, they are of course no "real" degrees of freedom and so, strictly
    // speaking, we shouldn't have assembled entries in the residual vector
    // for them. However, as we always do, we want to do exactly the same
    // thing on every cell and so we didn't not want to deal with the question
    // of whether a particular degree of freedom sits at the boundary in the
    // integration above. Rather, we will simply set to zero these entries
    // after the fact. To this end, we need to determine which degrees
    // of freedom do in fact belong to the boundary and then loop over all of
    // those and set the residual entry to zero. This happens in the following
    // lines which we have already seen used in step-11, using the appropriate
    // function from namespace DoFTools:
    hanging_node_constraints.condense(residual);

    for (types::global_dof_index i :
         DoFTools::extract_boundary_dofs(dof_handler))
      residual(i) = 0;

    // At the end of the function, we return the norm of the residual:
    return residual.l2_norm();
  }



  // @sect4{MinimalSurfaceProblem::determine_step_length}

  // As discussed in the introduction, Newton's method frequently does not
  // converge if we always take full steps, i.e., compute $u^{n+1}=u^n+\delta
  // u^n$. Rather, one needs a damping parameter (step length) $\alpha^n$ and
  // set $u^{n+1}=u^n+\alpha^n\delta u^n$. This function is the one called
  // to compute $\alpha^n$.
  //
  // Here, we simply always return 0.1. This is of course a sub-optimal
  // choice: ideally, what one wants is that the step size goes to one as we
  // get closer to the solution, so that we get to enjoy the rapid quadratic
  // convergence of Newton's method. We will discuss better strategies below
  // in the results section.
  template <int dim>
  double MinimalSurfaceProblem<dim>::determine_step_length() const
  {
    return 0.1;
  }



  // @sect4{MinimalSurfaceProblem::output_results}

  // This last function to be called from `run()` outputs the current solution
  // (and the Newton update) in graphical form as a VTU file. It is entirely the
  // same as what has been used in previous tutorials.
  template <int dim>
  void MinimalSurfaceProblem<dim>::output_results(
    const unsigned int refinement_cycle) const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, "solution");
    data_out.add_data_vector(newton_update, "update");
    data_out.build_patches();

    const std::string filename =
      "solution-" + Utilities::int_to_string(refinement_cycle, 2) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }


  // @sect4{MinimalSurfaceProblem::run}

  // In the run function, we build the first grid and then have the top-level
  // logic for the Newton iteration.
  //
  // As described in the introduction, the domain is the unit disk around
  // the origin, created in the same way as shown in step-6. The mesh is
  // globally refined twice followed later on by several adaptive cycles.
  //
  // Before starting the Newton loop, we also need to do a bit of
  // setup work: We need to create the basic data structures and
  // ensure that the first Newton iterate already has the correct
  // boundary values, as discussed in the introduction.
  //
  // TODO[JPP]: Changes: Tolerance; formulation
  template <int dim>
  void MinimalSurfaceProblem<dim>::run(const int    formulation,
                                       const double tolerance)
  {
    std::cout << "******** Assembly approach ********" << std::endl;
    switch (formulation)
      {
        case (0):
          std::cout << "Unassisted implementation (full hand linearization).\n"
                    << std::endl;
          break;
        case (1):
          std::cout
            << "Automated linearization of the finite element residual.\n"
            << std::endl;
          break;
        case (2):
          std::cout
            << "Automated computation of finite element residual and linearization using a variational formulation.\n"
            << std::endl;
          break;
        default:
          AssertThrow(false, ExcNotImplemented());
          break;
      }

    TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

    GridGenerator::hyper_ball(triangulation);
    triangulation.refine_global(2);

    setup_system(/*first time=*/true);
    set_boundary_values();

    // The Newton iteration starts next. We iterate until the (norm of the)
    // residual computed at the end of the previous iteration is less than
    // $10^{-3}$, as checked at the end of the `do { ... } while` loop that
    // starts here. Because we don't have a reasonable value to initialize
    // the variable, we just use the largest value that can be represented
    // as a `double`.
    double       last_residual_norm = std::numeric_limits<double>::max();
    unsigned int refinement_cycle   = 0;
    do
      {
        std::cout << "Mesh refinement step " << refinement_cycle << std::endl;

        if (refinement_cycle != 0)
          refine_mesh();

        // On every mesh we do exactly five Newton steps. We print the initial
        // residual here and then start the iterations on this mesh.
        //
        // In every Newton step the system matrix and the right hand side have
        // to be computed first, after which we store the norm of the right
        // hand side as the residual to check against when deciding whether to
        // stop the iterations. We then solve the linear system (the function
        // also updates $u^{n+1}=u^n+\alpha^n\;\delta u^n$) and output the
        // norm of the residual at the end of this Newton step.
        //
        // After the end of this loop, we then also output the solution on the
        // current mesh in graphical form and increment the counter for the
        // mesh refinement cycle.
        std::cout << "  Initial residual: " << compute_residual(0) << std::endl;

        for (unsigned int inner_iteration = 0; inner_iteration < 5;
             ++inner_iteration)
          {
            // Here we change the method of assembly based on the input
            // parameter.
            timer.enter_subsection("Assemble");
            if (formulation == 0)
              assemble_system_unassisted();
            else if (formulation == 1)
              assemble_system_with_residual_linearization();
            else if (formulation == 2)
              assemble_system_using_energy_functional();
            else
              AssertThrow(false, ExcNotImplemented());
            timer.leave_subsection();

            last_residual_norm = system_rhs.l2_norm();

            timer.enter_subsection("Solve");
            solve();
            timer.leave_subsection();

            std::cout << "  Residual: " << compute_residual(0) << std::endl;
          }

        output_results(refinement_cycle);

        ++refinement_cycle;
        std::cout << std::endl;
      }
    while (last_residual_norm > tolerance);
  }
} // namespace Step72

// @sect4{The main function}

// Finally the main function. This follows the scheme of all other main
// functions:
int main(int argc, char *argv[])
{
  try
    {
      using namespace Step72;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

      std::string prm_file;
      if (argc > 1)
        prm_file = argv[1];
      else
        prm_file = "parameters.prm";

      const MinimalSurfaceProblemParameters parameters;
      ParameterAcceptor::initialize(prm_file);

      MinimalSurfaceProblem<2> laplace_problem_2d;
      laplace_problem_2d.run(parameters.formulation, parameters.tolerance);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}