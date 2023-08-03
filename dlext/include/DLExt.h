// SPDX-License-Identifier: MIT
// This file is part of `lammps-dlext`, see LICENSE.md

#ifndef LAMMPS_DLPACK_EXTENSION_H_
#define LAMMPS_DLPACK_EXTENSION_H_

#include "LAMMPSView.h"
#include "atom.h"

#ifdef LMP_KOKKOS
#include "atom_kokkos.h"
#endif

#include <type_traits>
#include <vector>

namespace LAMMPS_NS
{
namespace dlext
{

#ifndef LMP_KOKKOS
using LMP_FLOAT = double;
using X_FLOAT = double;
using V_FLOAT = double;
using F_FLOAT = double;
#endif

static struct Positions { } kPositions;
static struct Velocities { } kVelocities;
static struct Masses { } kMasses;
static struct Forces { } kForces;
static struct Images { } kImages;
static struct Tags { } kTags;
static struct TagsMap { } kTagsMap;
static struct Types { } kTypes;

static struct SecondDim { } kSecondDim;

struct DLDataBridge {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLManagedTensor tensor;
};

void delete_bridge(DLManagedTensor* tensor)
{
    if (tensor)
        delete static_cast<DLDataBridge*>(tensor->manager_ctx);
}

template <typename T>
inline void* opaque(T* data)
{
    return static_cast<void*>(data);
}

template <typename T>
inline void* opaque(const T* data)
{
    return const_cast<void*>(data);
}

#ifdef LMP_KOKKOS
#define DLEXT_OPAQUE_ATOM_KOKKOS(PROPERTY, ACCESSOR)       \
    inline void* opaque(const AtomKokkos* atom, PROPERTY)  \
    {                                                      \
        return opaque(atom->ACCESSOR.d_view.data());       \
    }

DLEXT_OPAQUE_ATOM_KOKKOS(Positions, k_x)
DLEXT_OPAQUE_ATOM_KOKKOS(Velocities, k_v)
DLEXT_OPAQUE_ATOM_KOKKOS(Masses, k_mass)
DLEXT_OPAQUE_ATOM_KOKKOS(Forces, k_f)
DLEXT_OPAQUE_ATOM_KOKKOS(Images, k_image)
DLEXT_OPAQUE_ATOM_KOKKOS(Tags, k_tag)
DLEXT_OPAQUE_ATOM_KOKKOS(TagsMap, k_map_array)
DLEXT_OPAQUE_ATOM_KOKKOS(Types, k_type)

#undef DLEXT_OPAQUE_ATOM_KOKKOS
#endif

inline void* opaque(const Atom* atom, Positions) { return opaque(atom->x[0]); }
inline void* opaque(const Atom* atom, Velocities) { return opaque(atom->v[0]); }
inline void* opaque(const Atom* atom, Masses) { return opaque(atom->mass); }
inline void* opaque(const Atom* atom, Forces) { return opaque(atom->f[0]); }
inline void* opaque(const Atom* atom, Images) { return opaque(atom->image); }
inline void* opaque(const Atom* atom, Tags) { return opaque(atom->tag); }
inline void* opaque(const Atom* atom, Types) { return opaque(atom->type); }
inline void* opaque(const Atom* atom, TagsMap)
{
    return opaque(const_cast<Atom*>(atom)->get_map_array());
}

template <typename Property>
inline void* opaque(const LAMMPSView& view, DLDeviceType device_type, Property p)
{
#ifdef LMP_KOKKOS
    if (device_type == kDLCUDA)
        return opaque(view.atom_kokkos_ptr(), p);
#endif
    return opaque(view.atom_ptr(), p);
}

inline DLDevice device_info(const LAMMPSView& view, DLDeviceType device_type)
{
    return DLDevice { device_type, view.device_id() };
}

constexpr DLDataTypeCode dtype_code(Positions) { return kDLFloat; }
constexpr DLDataTypeCode dtype_code(Velocities) { return kDLFloat; }
constexpr DLDataTypeCode dtype_code(Masses) { return kDLFloat; }
constexpr DLDataTypeCode dtype_code(Forces) { return kDLFloat; }
constexpr DLDataTypeCode dtype_code(Images) { return kDLInt; }
constexpr DLDataTypeCode dtype_code(Tags) { return kDLInt; }
constexpr DLDataTypeCode dtype_code(TagsMap) { return kDLInt; }
constexpr DLDataTypeCode dtype_code(Types) { return kDLInt; }

#define DLEXT_BITS_FLOAT_ARRAY(PROPERTY, TYPE)                                          \
    inline uint8_t bits(DLDeviceType device_type, PROPERTY)                             \
    {                                                                                   \
        return (device_type == kDLCPU || std::is_same<TYPE, double>::value) ? 64 : 32;  \
    }

DLEXT_BITS_FLOAT_ARRAY(Positions, X_FLOAT)
DLEXT_BITS_FLOAT_ARRAY(Velocities, V_FLOAT)
DLEXT_BITS_FLOAT_ARRAY(Masses, LMP_FLOAT)
DLEXT_BITS_FLOAT_ARRAY(Forces, F_FLOAT)

#undef DLEXT_BITS_FLOAT_ARRAY

#define DLEXT_BITS_INT_ARRAY(PROPERTY, TYPE)                  \
    inline uint8_t bits(DLDeviceType device_type, PROPERTY)   \
    {                                                         \
        cxx11::maybe_unused(device_type);                     \
        return std::is_same<TYPE, int64_t>::value ? 64 : 32;  \
    }

DLEXT_BITS_INT_ARRAY(Images, imageint)
DLEXT_BITS_INT_ARRAY(Tags, tagint)

#undef DLEXT_BITS_INT_ARRAY

inline uint8_t bits(DLDeviceType device_type, TagsMap) { return 32; }
inline uint8_t bits(DLDeviceType device_type, Types) { return 32; }

template <typename Property>
inline DLDataType dtype(DLDeviceType device_type, Property p)
{
    return DLDataType { dtype_code(p), bits(device_type, p), 1 };
}

template <typename Property>
inline int64_t size(const LAMMPSView& view, Property)
{
    return view.local_particle_number();
}
inline int64_t size(const LAMMPSView& view, Masses) { return view.atom_ptr()->ntypes + 1; }
inline int64_t size(const LAMMPSView& view, TagsMap) { return view.atom_ptr()->get_map_size(); }

template <typename Property>
inline int64_t size(const LAMMPSView& view, Property, SecondDim)
{
    return 1;
}
inline int64_t size(const LAMMPSView& view, Positions, SecondDim) { return 3; }
inline int64_t size(const LAMMPSView& view, Velocities, SecondDim) { return 3; }
inline int64_t size(const LAMMPSView& view, Forces, SecondDim) { return 3; }

template <typename Property>
constexpr uint64_t offset(const LAMMPSView& view, Property p)
{
    return 0;
}

template <typename Property>
DLManagedTensor* wrap(const LAMMPSView& view, Property property, ExecutionSpace exec_space)
{
    auto bridge = std::make_unique<DLDataBridge>();
    bridge->tensor.manager_ctx = bridge.get();
    bridge->tensor.deleter = delete_bridge;

    auto& dltensor = bridge->tensor.dl_tensor;
    auto device_type = view.device_type(exec_space);
    dltensor.data = opaque(view, device_type, property);
    dltensor.device = device_info(view, device_type);
    dltensor.dtype = dtype(device_type, property);

    auto& shape = bridge->shape;
    auto size2 = size(view, property, kSecondDim);
    shape.push_back(size(view, property));
    if (size2 > 1)
        shape.push_back(size2);

    auto& strides = bridge->strides;
    strides.push_back(size2);
    if (size2 > 1)
        strides.push_back(1);

    dltensor.ndim = shape.size();
    dltensor.shape = reinterpret_cast<std::int64_t*>(shape.data());
    dltensor.strides = reinterpret_cast<std::int64_t*>(strides.data());
    dltensor.byte_offset = offset(view, property);

    return &(bridge.release()->tensor);
}

#define DLEXT_PROPERTY_FROM_VIEW(FN, SELECTOR)                                \
    inline DLManagedTensor* FN(const LAMMPSView& view, ExecutionSpace space)  \
    {                                                                         \
        return wrap(view, SELECTOR, space);                                   \
    }

DLEXT_PROPERTY_FROM_VIEW(positions, kPositions)
DLEXT_PROPERTY_FROM_VIEW(velocities, kVelocities)
DLEXT_PROPERTY_FROM_VIEW(masses, kMasses)
DLEXT_PROPERTY_FROM_VIEW(forces, kForces)
DLEXT_PROPERTY_FROM_VIEW(images, kImages)
DLEXT_PROPERTY_FROM_VIEW(tags, kTags)
DLEXT_PROPERTY_FROM_VIEW(tags_map, kTagsMap)
DLEXT_PROPERTY_FROM_VIEW(types, kTypes)

#undef DLEXT_PROPERTY

}  // namespace dlext
}  // namespace LAMMPS_NS

#endif  // LAMMPS_DLPACK_EXTENSION_H_
