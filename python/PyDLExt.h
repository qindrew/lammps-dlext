// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef PY_LAMMPS_DLPACK_EXTENSION_H_
#define PY_LAMMPS_DLPACK_EXTENSION_H_

#include "Sampler.h"
#include "pybind11/pybind11.h"

namespace LAMMPS_NS
{
namespace dlext
{

using PyCapsule = pybind11::capsule;
using PyTensorBundle = std::tuple<PyObject*, DLManagedTensorPtr, DLManagedTensorDeleter>;

const char* const kDLTensorCapsuleName = "dltensor";
const char* const kUsedDLTensorCapsuleName = "used_dltensor";

static std::vector<PyTensorBundle> kPyCapsulesPool;

inline PyCapsule pyencapsulate(DLManagedTensorPtr tensor, bool autodestruct = true)
{
    auto capsule = PyCapsule(tensor, kDLTensorCapsuleName, nullptr);
    if (autodestruct)
        PyCapsule_SetDestructor(
            capsule.ptr(),
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
    return capsule;
}

template <typename ExternalUpdater, template <typename> class Wrapper, class DeviceType, typename Property>
struct DEFAULT_VISIBILITY PyUnsafeEncapsulator final {
    static PyCapsule wrap_property(const Sampler<ExternalUpdater, Wrapper, DeviceType>& sampler,
     AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        DLManagedTensorPtr tensor = Property::from(sampler, location, mode);
        return pyencapsulate(tensor);
    }
};

// wrap_property is used in lammps_dlext.cc and exposed to the dlpack_extension interface
//   Property can be Positions, Types, Velocities, NetForces, Tags and Images structs defined in Sampler.h

template <typename Property>
struct DEFAULT_VISIBILITY PyEncapsulator final {
    static PyCapsule wrap_property(AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        // here Property would be Positions, Types, Velocities, NetForces, Tags and Images
        //   that are structs defined in Sampler.h
        //   from() returns a tensor (DLManagedTensorPtr)
        auto tensor = Property::from(location, mode);

        // create a capsule from the tensor
        auto capsule = pyencapsulate(tensor, /* autodestruct = */ false);
        kPyCapsulesPool.push_back(std::make_tuple(capsule.ptr(), tensor, tensor->deleter));

        // We manually delete the tensor when exiting the context manager,
        // so we need to prevent others from grabbing the default deleter.
        //   do_not_delete(DLManagedTensorPtr tensor) is defined in DLExt.h
        tensor->deleter = do_not_delete;
        return capsule;
    }
};

void invalidate(PyTensorBundle& bundle)
{
    auto obj = std::get<0>(bundle);
    auto tensor = std::get<1>(bundle);
    auto shred = std::get<2>(bundle);

    shred(tensor);

    if (PyCapsule_IsValid(obj, kDLTensorCapsuleName)) {
        PyCapsule_SetName(obj, kUsedDLTensorCapsuleName);
        PyCapsule_SetPointer(obj, opaque(&kInvalidDLManagedTensor));
    } else if (PyCapsule_IsValid(obj, kUsedDLTensorCapsuleName)) {
        PyCapsule_SetPointer(obj, opaque(&kInvalidDLManagedTensor));
    }
}

} // namespace dlext
} // namspace LAMMPS_NS

#endif  // PY_LAMMPS_DLPACK_EXTENSION_H_
