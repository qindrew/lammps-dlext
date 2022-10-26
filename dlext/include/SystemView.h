// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef LAMMPS_SYSVIEW_H_
#define LAMMPS_SYSVIEW_H_

#include "DLExt.h"
#include "lammps.h"
#include "fix.h"
#include "KOKKOS/kokkos_type.h"

using namespace LAMMPS_NS;

namespace dlext
{

namespace cxx11 = cxx11utils;

class SystemView;

//{ // Aliases

//! Specifies where to acquire the data
struct access_location {
    //! The enum
    enum Enum {
        host,   //!< Ask to acquire the data on the host
#ifdef ENABLE_CUDA
        device  //!< Ask to acquire the data on the device
#endif
    };
};
using AccessLocation = access_location::Enum;
const auto kOnHost = access_location::host;
#ifdef ENABLE_CUDA
const auto kOnDevice = access_location::device;
#endif

//! Specify how the data is to be accessed
struct access_mode  {
    //! The enum
    enum Enum {
        read,       //!< Data will be accessed read only
        readwrite,  //!< Data will be accessed for read and write
        overwrite   //!< The data is to be completely overwritten during this acquire
    };
};
using AccessMode = access_mode::Enum;
const auto kRead = access_mode::read;
const auto kReadWrite = access_mode::readwrite;
const auto kOverwrite = access_mode::overwrite;

//using ParticleDataSPtr = std::shared_ptr<ParticleData>;
//using SystemDefinitionSPtr = std::shared_ptr<SystemDefinition>;
//using ExecutionConfigurationSPtr = std::shared_ptr<const ExecutionConfiguration>;
/*
template <template <typename> class Array, typename T, typename Object>
using ArrayPropertyGetter = const Array<T>& (Object::*)() const;

template <typename T>
using PropertyGetter = T (*)(const SystemView&, AccessLocation, AccessMode);
*/

//} // Aliases

class DEFAULT_VISIBILITY SystemView : public Fix {
public:
    SystemView(LAMMPS_NS::LAMMPS* lmp, int, char**);
/*
    ParticleDataSPtr particle_data() const;
    ExecutionConfigurationSPtr exec_config() const;
    bool is_gpu_enabled() const;
    bool in_context_manager() const;
    unsigned int local_particle_number() const;
    unsigned int global_particle_number() const;
    int get_device_id(bool gpu_flag) const;
    void synchronize();
    void enter();
    void exit();
*/
protected:
/*
    SystemDefinitionSPtr _sysdef;
    ParticleDataSPtr _pdata;
    ExecutionConfigurationSPtr _exec_conf;
    bool _in_context_manager = false;
*/    
};

inline DLDevice dldevice(const SystemView& sysview, bool gpu_flag)
{
    return DLDevice { gpu_flag ? kDLCUDA : kDLCPU, 
                      0 //sysview.get_device_id(gpu_flag)
                       };
}

inline unsigned int particle_number(const SystemView& sysview)
{
    return 0; //sysview.local_particle_number();
}


// see atom_kokkos.h for executation space and datamask
/*
*/
template <template <typename> class A, typename T, typename O>
DLManagedTensorPtr wrap(const SystemView& sysview,
                        const AccessLocation location, const AccessMode mode,
                        int64_t size2 = 1, uint64_t offset = 0, uint64_t stride1_offset = 0)
{
    assert((size2 >= 1));
/*
    auto location = sysview.is_gpu_enabled() ? requested_location : kOnHost;
    auto handle = cxx11utils::make_unique<ArrayHandle<T>>(
        INVOKE(*(sysview.particle_data()), getter)(), location, mode
    );
    auto bridge = cxx11utils::make_unique<DLDataBridge<T>>(handle);

#ifdef ENABLE_CUDA
    auto gpu_flag = (location == kOnDevice);
#else
    auto gpu_flag = false;
#endif

    bridge->tensor.manager_ctx = bridge.get();
    bridge->tensor.deleter = delete_bridge<T>;

    auto& dltensor = bridge->tensor.dl_tensor;
    dltensor.data = opaque(bridge->handle->data);
    dltensor.device = dldevice(sysview, gpu_flag);
    dltensor.dtype = dtype<T>();

    auto& shape = bridge->shape;
    int n
    //shape.push_back(particle_number<A>(sysview));
    if (size2 > 1)
        shape.push_back(size2);

    auto& strides = bridge->strides;
    strides.push_back(stride1<T>() + stride1_offset);
    if (size2 > 1)
        strides.push_back(1);

    dltensor.ndim = shape.size();
    dltensor.shape = reinterpret_cast<std::int64_t*>(shape.data());
    dltensor.strides = reinterpret_cast<std::int64_t*>(strides.data());
    dltensor.byte_offset = offset;
*/
    return nullptr; //&(bridge.release()->tensor);
}
/*
struct PositionsTypes final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, const AccessLocation location, AccessMode mode
    )
    {
        return wrap(sysview, location, mode, 4);
    }
};
*/
/*
struct VelocitiesMasses final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, location, mode, 4);
    }
};
*/
/*
struct Orientations final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, &ParticleData::getOrientationArray, location, mode, 4);
    }
};
*/
/*
struct AngularMomenta final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, &ParticleData::getAngularMomentumArray, location, mode, 4);
    }
};
*/
/*
struct MomentsOfInertia final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, &ParticleData::getMomentsOfInertiaArray, location, mode, 3);
    }
};
*/
/*
struct Charges final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, &ParticleData::getCharges, location, mode);
    }
};

struct Diameters final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, &ParticleData::getDiameters, location, mode);
    }
};

struct Images final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, &ParticleData::getImages, location, mode, 3);
    }
};

struct Tags final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, &ParticleData::getTags, location, mode);
    }
};

struct RTags final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, &ParticleData::getRTags, location, mode);
    }
};

struct NetForces final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, &ParticleData::getNetForce, location, mode, 4);
    }
};

struct NetTorques final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, &ParticleData::getNetTorqueArray, location, mode, 4);
    }
};

struct NetVirial final {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return wrap(sysview, &ParticleData::getNetVirial, location, mode, 6);
    }
};
*/
}  // namespace dlext

#endif  // LAMMPS_SYSVIEW_H_
