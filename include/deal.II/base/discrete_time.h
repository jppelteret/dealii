// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_discrete_time_h
#define dealii_discrete_time_h

#include <deal.II/base/config.h>

DEAL_II_NAMESPACE_OPEN

/**
 * Provides a means to keep track of the simulation time in a time-dependent
 * simulation. It manages stepping forward from an initial time to a final
 * time. It also allows adjusting the time step size during the simulation.
 * It is guaranteed that at all times the current simulation time is in the
 * closed interval between the start time and the end time.
 *
 * You can loop over all of the time steps by using a for loop
 * @code
 *   for (DiscreteTime time(0., 1., 0.3);
 *        time.is_at_end() == false;
 *        time.advance_time())
 *   {
 *     // Insert simulation code here
 *   }
 * @code
 *
 * In the above example the time starts at $t = 0$. Assuming the time step
 * $dt = 0.3$ is not modified inside the loop, the time is advanced to
 * $t = 0.3$, $t = 0.6$, $t = 0.9$ and finally it reaches the end time at
 * $t = 1.0$. Note that the final step size is automatically reduced to
 * $dt = 0.1$ in order to ensure that we finish the simulation exactly at
 * the specified end time.
 *
 * @author Reza Rastak, 2019
 */
class DiscreteTime
{
public:
  /**
   * Constructor
   */
  DiscreteTime(const double start_time,
               const double end_time,
               const double start_step_size);

  /**
   * Return the current time.
   */
  double
  get_current_time() const;

  /**
   * Return the next time that we reach if we advance time by one step.
   */
  double
  get_next_time() const;

  /**
   * Return the previous time.
   */
  double
  get_previous_time() const;

  /**
   * Return the start time.
   */
  double
  get_start_time() const;

  /**
   * Return the end of the time interval.
   * The final time step ends exactly at this point. This exact floating-point
   * equality is very important because it allows us to equality-compare
   * current time with end time and decide whether we have reached the end of
   * the simulation.
   */
  double
  get_end_time() const;

  /**
   * Return whether no step has taken place yet.
   */
  bool
  is_at_start() const;

  /**
   * Return whether time has reached the end time.
   */
  bool
  is_at_end() const;

  /**
   * Return the size of the step from current time step to the next.
   */
  double
  get_next_step_size() const;

  /**
   * Return the step size of the previous step.
   */
  double
  get_previous_step_size() const;

  /**
   * Return the number of times the simuation time is incremented.
   */
  unsigned int
  get_step_number() const;

  /**
   * Set the value of the next time step size. The next time advance_time()
   * is called, the newly set @p time_step_size will be used to advance
   * the simulation time. However, if the step is too large such that the next
   * simulation time exceeds the end time, the step size is truncated.
   * Additionally, if the step size is such that the next simulation time
   * approximates the end time (but falls just slightly short of it), the step
   * size is adjusted such that the next simulation time exactly matches the
   * end time.
   */
  void
  set_next_step_size(const double time_step_size);

  /**
   * Advance the current time based on the value of the current step.
   * If you want to adjust the next time step size, call the method
   * set_next_step_size() before calling this method.
   * If you call this function repeatedly, the time
   * is increased with the same step size until it reaches the end
   * time. See the documentation of set_next_step_size() for explanation
   * of the rules for automatic adjustment of the step size.
   *
   * @pre Current time must be smaller than the end time. The object cannot
   * advance time if it is already at the end time. This rule is created to
   * avoid the creation of an infinite loop when advance_time() is called
   * inside a loop.
   */
  void
  advance_time();

  /**
   * Set the current time equal to start time and set the step size to the
   * initial step size.
   */
  void
  restart();

private:
  /**
   * The beginning of the time interval.
   */
  const double start_time;

  /**
   *The end of the time interval.
   */
  const double end_time;

  /**
   * The size of the first step.
   */
  const double start_step_size;

  /**
   * The current time.
   */
  double current_time;

  /**
   * The time at the next step.
   *
   * @note Internally, the next simulation time is stored instead of the
   * current step size. For example, when the method set_next_step_size()
   * is called, it computes the appropriate next simulation time and stores
   * it. When advance_time() is called, the current_time is replaced by
   * next_time. This choice for the internal state allows for simpler code
   * and ensures than when we call advance_time() at the last step, the
   * floating-point value of the time exactly matches the end time.
   */
  double next_time;

  /**
   * The previous time.
   */
  double previous_time;

  /**
   * The step number i.e. the number of times the simulation time ha been
   * incremented.
   */
  unsigned int step_number;
};


/*---------------------- Inline functions ------------------------------*/


inline double
DiscreteTime::get_start_time() const
{
  return start_time;
}



inline double
DiscreteTime::get_end_time() const
{
  return end_time;
}



inline bool
DiscreteTime::is_at_start() const
{
  return step_number == 0;
}



inline bool
DiscreteTime::is_at_end() const
{
  return current_time == end_time;
}



inline double
DiscreteTime::get_next_step_size() const
{
  return next_time - current_time;
}



inline double
DiscreteTime::get_previous_step_size() const
{
  return current_time - previous_time;
}



inline double
DiscreteTime::get_current_time() const
{
  return current_time;
}



inline double
DiscreteTime::get_next_time() const
{
  return next_time;
}



inline double
DiscreteTime::get_previous_time() const
{
  return previous_time;
}



inline unsigned int
DiscreteTime::get_step_number() const
{
  return step_number;
}


DEAL_II_NAMESPACE_CLOSE

#endif
