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


// Check that integrator works for volumes and boundaries

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/weakforms/integrator.h>

#include "../tests.h"


template<int dim, int spacedim = dim>
void run()
{
  deallog << "Dim: " << dim << std::endl;

  const FE_Q<dim, spacedim> fe (1);
  const QGauss<dim>         cell_quadrature (fe.degree+1);
  const QGauss<dim-1>       face_quadrature (fe.degree+1);
  const UpdateFlags         update_flags_cell = update_quadrature_points | update_JxW_values;
  const UpdateFlags         update_flags_face = update_quadrature_points | update_JxW_values;

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 4, 0.0, 1.0);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  Functions::ConstantFunction<dim,double> unity (1.0);

  using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
  using CopyData         = MeshWorker::CopyData<1, 1, 1>;
  using CellIteratorType = decltype(dof_handler.begin_active());

  // Volume integral
  {
    // ScratchData scratch(fe, cell_quadrature, update_flags_cell);
    // CopyData    copy(1);

    // double vol = 0.0;

    // auto cell_worker = [&unity] (const CellIteratorType &cell,
    //                              ScratchData            &scratch_data,
    //                              CopyData               &copy_data)
    // {
    //   const auto &fe_values = scratch_data.reinit(cell);
    //   double      &cell_vol = copy_data.vectors[0][0];

    //   for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
    //     cell_vol += unity.value(fe_values.quadrature_point(q_point)) * fe_values.JxW(q_point);
    // };
    // auto copier = [&vol](const CopyData &copy_data)
    // {
    //   vol += copy_data.vectors[0][0];
    // };

    // const auto filtered_iterator_range =
    //   filter_iterators(dof_handler.active_cell_iterators(),
    //                    IteratorFilters::LocallyOwnedCell());
    // MeshWorker::mesh_loop(filtered_iterator_range,
    //                       cell_worker, copier,
    //                       scratch, copy,
    //                       MeshWorker::assemble_own_cells);

    const double vol = WeakForms::Integral<dim,double>(unity).dV(dof_handler, cell_quadrature);
    deallog << "Volume: " << vol << std::endl;
  }

  // Boundary integral
  {

  }

  deallog << "OK" << std::endl;
}


int
main()
{
  initlog();

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
