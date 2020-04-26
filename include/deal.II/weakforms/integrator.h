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

#ifndef dealii_weakforms_integrator_h
#define dealii_weakforms_integrator_h

#include <deal.II/base/config.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/meshworker/copy_data.h>

#include <functional>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{

  template<int dim, typename ReturnType = void>
  class Integral
  {
    using IntegrandPositionIndependent = std::function<ReturnType(const unsigned int &q_point)>;
    using IntegrandPositionDependent   = std::function<ReturnType(const unsigned int &q_point, const Point<dim> &position)>;

  public:

    /**
     * Construct a new Integral object
     * 
     * @param integrand 
     */
    Integral(const IntegrandPositionIndependent &integrand)
      : integrand_position_independent(integrand)
      , integrand_position_dependent(nullptr)
    {}

    /**
     * Construct a new Integral object
     * 
     * @param integrand 
     */
    Integral(const IntegrandPositionDependent &integrand)
      : integrand_position_independent(nullptr)
      , integrand_position_dependent(integrand)
    {}

    /**
     * Construct a new Integral object
     * 
     * @param function 
     */
    Integral(const Function<dim, ReturnType> &function)
      : integrand_position_independent(nullptr)
      , integrand_position_dependent([&function](
          const unsigned int &q_point, const Point<dim> &position) 
          {
            return function.value(position);
          })
    {
    }

    /**
     * Integrate on a volume.
     * 
     * @tparam spacedim 
     * @tparam DoFHandlerType 
     * @param dof_handler 
     * @param cell_quadrature 
     * @return ReturnType 
     */
    template<int spacedim, template< int, int > class DoFHandlerType>
    ReturnType
    dV (const DoFHandlerType<dim, spacedim> &dof_handler,
        const Quadrature<dim>               &cell_quadrature)
    {
      using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
      using CopyData         = MeshWorker::CopyData<1, 1, 1>;
      using CellIteratorType = decltype(dof_handler.begin_active());

      const UpdateFlags update_flags_cell = update_quadrature_points | update_JxW_values;

      ScratchData scratch(dof_handler.get_fe(), 
                          cell_quadrature, 
                          update_flags_cell);
      CopyData    copy(1);

      const auto &integrand_pd = this->integrand_position_dependent;
      const auto &integrand_pi = this->integrand_position_independent;
      auto cell_worker = [&integrand_pd, &integrand_pi] (
        const CellIteratorType &cell,
        ScratchData            &scratch_data,
        CopyData               &copy_data)
      {
        const auto &fe_values = scratch_data.reinit(cell);
        double     &cell_integral = copy_data.vectors[0][0];

        if (integrand_pd)
        {
          Assert(!integrand_pi, ExcInternalError());

          for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
            cell_integral += integrand_pd(q_point, fe_values.quadrature_point(q_point)) * fe_values.JxW(q_point);
        }
        else if (integrand_pi)
        {
          Assert(!integrand_pd, ExcInternalError());

          for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
            cell_integral += integrand_pi(q_point) * fe_values.JxW(q_point);
        }
      };

      ReturnType integral;
      auto copier = [&integral](const CopyData &copy_data)
      {
        const double &cell_integral = copy_data.vectors[0][0];
        integral += cell_integral;
      };

      const auto filtered_iterator_range =
        filter_iterators(dof_handler.active_cell_iterators(),
                        IteratorFilters::LocallyOwnedCell());
      MeshWorker::mesh_loop(filtered_iterator_range,
                            cell_worker, copier,
                            scratch, copy,
                            MeshWorker::assemble_own_cells);

      return integral;
    }

  private:

    /**
     * 
     * 
     */
    const IntegrandPositionDependent   integrand_position_dependent;

    /**
     * 
     * 
     */
    const IntegrandPositionIndependent integrand_position_independent;
  };

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE


#endif // dealii_weakforms_integrator_h
