// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef PY_LAMMPS_DLPACK_EXTENSION_H_
#define PY_LAMMPS_DLPACK_EXTENSION_H_


#include "DLExt.h"

#include "pybind11/pybind11.h"


namespace dlext
{

/*
using PropertyExtractor =
    DLManagedTensorPtr (*)(const SystemView&, AccessLocation, AccessMode)
;
*/
using PropertyExtractor =
    DLManagedTensorPtr (*)(const SystemView&)
;

const char* const kDLTensorCapsuleName = "dltensor";


template <PropertyExtractor property>
inline pybind11::capsule encapsulate(
    const SystemView& sysview
    //const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    //auto dl_managed_tensor = property(sysview, location, mode);
    auto dl_managed_tensor = property(sysview);
    return pybind11::capsule(
        dl_managed_tensor, kDLTensorCapsuleName,
        [](PyObject* obj) {  // PyCapsule_Destructor
            auto dlmt = static_cast<DLManagedTensorPtr>(
                PyCapsule_GetPointer(obj, kDLTensorCapsuleName)
            );
            if (dlmt && dlmt->deleter) {
                dlmt->deleter(dlmt);
            } else {
                PyErr_Clear();
            }
        }
    );
}


} // namespace dlext


#endif // PY_LAMMPS_DLPACK_EXTENSION_H_
