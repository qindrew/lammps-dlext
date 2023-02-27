// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef PY_LAMMPS_FIX_H_
#define PY_LAMMPS_FIX_H_

#include "fix.h"
#include <pybind11/pybind11.h>
#include "Sampler.h"

namespace LAMMPS_NS
{
namespace dlext
{

//! Trampoline class to allow overriding Fix methods from within Python.
//!
//! References:
//! - https://pybind11.readthedocs.io/en/stable/advanced/classes.html
//!
class DEFAULT_VISIBILITY PyFix : public Fix {
public:
    using Fix::Fix;

    void setmask(int mask) override
    {
        PYBIND11_OVERLOAD_PURE(void, Fix, setmask, mask);
    }
};

}  // namespace dlext
}  // namespace LAMMPS_NS

#endif  // PY_LAMMPS_FIX_H_
