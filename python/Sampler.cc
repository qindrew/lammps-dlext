#include "Sampler.h"
#include "KOKKOS/kokkos_type.h"  // this requires LAMMPS KOKKOS essential headers in LAMMPS_INCLUDE_DIR (e.g. $HOME/miniconda3/envs/pysages/include/lammps),
                                 // importantly, kokkos_type.h includes the core KOKKOS headers which are installed in virtual env
                                 // Scalar, Scalar3, Scalar4, int3 are needed
#include "KOKKOS/atom_kokkos.h"
#include "atom_masks.h"
#include <stdexcept>

using namespace std;
using namespace Kokkos;
namespace py = pybind11;

const char* const kDLTensorCapsuleName = "dltensor";
template<class Scalar>
constexpr uint8_t kBits = std::is_same<Scalar, float>::value ? 32 : 64;

// KOKKOS supports double precision for now

template <typename>
constexpr DLDataType dtype();
template <>
constexpr DLDataType dtype<double4>() { return DLDataType {kDLFloat, 64, 1}; }
template <>
constexpr DLDataType dtype<double3>() { return DLDataType {kDLFloat, 64, 1}; }
template <>
constexpr DLDataType dtype<double>() { return DLDataType {kDLFloat, 64, 1}; }

template <>
constexpr DLDataType dtype<int3>() { return DLDataType {kDLInt, 32, 1}; }
template <>
constexpr DLDataType dtype<unsigned int>() { return DLDataType {kDLUInt, 32, 1}; }
template <>
constexpr DLDataType dtype<int>() { return DLDataType {kDLInt, 32, 1}; }

template <typename>
constexpr int64_t stride1();
template <>
constexpr int64_t stride1<double4>() { return 4; }
template <>
constexpr int64_t stride1<double3>() { return 3; }
template <>
constexpr int64_t stride1<double>() { return 1; }
template <>
constexpr int64_t stride1<int3>() { return 3; }
template <>
constexpr int64_t stride1<unsigned int>() { return 1; }

template <typename T>
inline void* opaque(T* data) { return static_cast<void*>(data); }

inline py::capsule encapsulate(DLManagedTensor* dl_managed_tensor)
{
  return py::capsule(dl_managed_tensor, kDLTensorCapsuleName);
}

template<class DeviceType>
Sampler<DeviceType>::Sampler(LAMMPS_NS::LAMMPS* lmp, int narg, char** arg, py::function python_update)
  : Fix(lmp, narg, arg), m_python_update(python_update)
{
  this->setSimulator(lmp);

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;

  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read =  V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK;
  datamask_modify = F_MASK;
}

template<class DeviceType>
void Sampler<DeviceType>::setSimulator(LAMMPS_NS::LAMMPS* lmp)
{
  m_lmp = lmp;
}


template<class DeviceType>
void Sampler<DeviceType>::run_on_data(py::function py_exec)
{
  bool on_device = true;
  if (execution_space == LAMMPS_NS::Host)
    on_device = false;

  atomKK->sync(execution_space,datamask_read);
  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  image = atomKK->k_image.view<DeviceType>();
  atomKK->sync(execution_space,datamask_modify);
  f = atomKK->k_f.view<DeviceType>();

  auto pos_bridge = wrap<double4, double>(x, on_device, 4 );
  auto pos_capsule = encapsulate(&pos_bridge.tensor);
  auto vel_bridge = wrap<double4, double>(v, on_device, 4 );
  auto vel_capsule = encapsulate(&vel_bridge.tensor);
  auto img_bridge = wrap<int3, double>(image, on_device, 3 );
  auto img_capsule = encapsulate(&img_bridge.tensor);
  auto force_bridge = wrap<double4, double>(f, on_device, 4 );
  auto force_capsule = encapsulate(&force_bridge.tensor);

  py_exec(pos_capsule, vel_capsule, img_capsule, force_capsule); 
}

template<class DeviceType>
void Sampler<DeviceType>::post_force(int)
{

  // Accessing the handles here holds them valid until the block of this function.
  // This keeps them valid for the python function call
  bool on_device = true;
  if (execution_space == LAMMPS_NS::Host)
    on_device = false;

  auto pos_bridge = wrap<double4, double>(x, on_device, 4 );
  auto pos_capsule = encapsulate(&pos_bridge.tensor);
  auto vel_bridge = wrap<double4, double>(v, on_device, 4 );
  auto vel_capsule = encapsulate(&vel_bridge.tensor);
  auto img_bridge = wrap<int3, double>(image, on_device, 3 );
  auto img_capsule = encapsulate(&img_bridge.tensor);
  auto force_bridge = wrap<double4, double>(f, on_device, 4 );
  auto force_capsule = encapsulate(&force_bridge.tensor);


  // const ArrayHandle<Scalar4> pos(m_pdata->getPositions(), location, access_mode::read);
  // auto pos_tensor = wrap<Scalar4, Scalar>(pos.data, 4 );
  // ArrayHandle<Scalar4> vel(m_pdata->getVelocities(), location, access_mode::read);
  // auto vel_tensor = wrap<Scalar4, Scalar>(vel.data, 4);
  // ArrayHandle<unsigned int> rtags(m_pdata->getRTags(), location, access_mode::read);
  // auto rtag_tensor = wrap<unsigned int, unsigned int>(rtags.data, 1);
  // ArrayHandle<int3> img(m_pdata->getImages(), location, access_mode::read);
  // auto img_tensor = wrap<int3, int>(img.data, 3);

  // ArrayHandle<Scalar4> net_forces(m_pdata->getNetForce(), location, access_mode::readwrite);
  // auto force_tensor = wrap<Scalar4, Scalar>(net_forces.data, 4);

  // m_python_update(pos_tensor, vel_tensor, rtag_tensor, img_tensor, force_tensor,
  //                 m_pdata->getGlobalBox());
  this->run_on_data(m_python_update);

}

template<class DeviceType>
template <typename TV, typename TS>
DLDataBridge Sampler<DeviceType>::wrap(TV* ptr,
                           const bool on_device,
                           const int64_t size2,
                           const uint64_t offset,
                           uint64_t stride1_offset) {
  assert((size2 >= 1)); // assert is a macro so the extra parentheses are requiered here

  const unsigned int particle_number = atom->nlocal;
  const int gpu_id = 0;

  DLDataBridge bridge;
  bridge.tensor.manager_ctx = NULL;
  bridge.tensor.deleter = NULL;

  bridge.tensor.dl_tensor.data = opaque(ptr);
  bridge.tensor.dl_tensor.device = DLDevice{on_device ? kDLCUDA : kDLCPU, gpu_id};
  bridge.tensor.dl_tensor.dtype = dtype<TS>();

  bridge.shape.push_back(particle_number);
  if (size2 > 1)
    bridge.shape.push_back(size2);

  bridge.strides.push_back(stride1<TV>() + stride1_offset);
  if (size2 > 1)
    bridge.strides.push_back(1);

  bridge.tensor.dl_tensor.ndim = bridge.shape.size();
  bridge.tensor.dl_tensor.dtype = dtype<TS>();
  bridge.tensor.dl_tensor.shape = reinterpret_cast<std::int64_t*>(bridge.shape.data());
  bridge.tensor.dl_tensor.strides = reinterpret_cast<std::int64_t*>(bridge.strides.data());
  bridge.tensor.dl_tensor.byte_offset = offset;

  return bridge;
}


void export_Sampler(py::module& m)
{
/*
  py::class_<Sampler, std::shared_ptr<Sampler> >(m, "DLextSampler", py::base<HalfStepHook>())
    .def(py::init<std::shared_ptr<SystemDefinition>, py::function>())
    .def("run_on_data", &Sampler::run_on_data)
    ;
*/    
}
