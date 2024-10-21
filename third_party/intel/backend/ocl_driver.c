//===- driver.c -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <CL/cl.h>
#include <sycl/sycl.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
// #include <numpy/arrayobject.h>

#include "ocl_functions.h"
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "opencl.hpp"

static std::vector<std::unique_ptr<sycl::device>> g_sycl_devices;
static std::once_flag g_sycl_devices_flag;

// TODO: auto run after library loaded
// NOTE: must keep logic same with pytorch `c10/xpu/XPUFunctions.cpp`
void enumDevices() {
  auto platform_list = sycl::platform::get_platforms();
  for (const auto &platform : platform_list) {
    auto device_list = platform.get_devices();
    for (const auto &device : device_list) {
      g_sycl_devices.push_back(std::make_unique<sycl::device>(device));
    }
  }
}

static inline void gpuAssert(cl_int code) {
  if (code != CL_SUCCESS) {
    auto str = parseOclResultCode(code);
    char err[1024] = {0};
    strncat(err, str.c_str(), std::min(str.size(), size_t(1024)));
    PyGILState_STATE gil_state;
    gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, err);
    PyGILState_Release(gil_state);
  }
}

template <typename T>
static inline T checkSyclErrors(const std::tuple<T, cl_int> tuple) {
  gpuAssert(std::get<1>(tuple));
  if (PyErr_Occurred())
    return nullptr;
  else
    return std::get<0>(tuple);
}

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  std::call_once(g_sycl_devices_flag, enumDevices);
  if (device_id > g_sycl_devices.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }
  const auto &sycl_device = g_sycl_devices[device_id];

  // Get device handle
  cl_device_id phDevice = sycl::get_native<sycl::backend::opencl>(*sycl_device);

  // use clhpp
  cl::Device d = cl::Device(phDevice);
  int multiprocessor_count = d.getInfo<CL_DEVICE_MAX_NUM_SUB_GROUPS>();
  int sm_clock_rate = d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
  int max_shared_mem = d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
  int max_group_size = d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

  // TODO
  // CL_DEVICE_SUB_GROUP_SIZES_INTEL
  // int num_subgroup_sizes = d.getInfo<>();
  int num_subgroup_sizes = 4;
  PyObject *subgroup_sizes = PyTuple_New(num_subgroup_sizes);
  for (int i = 0; i < num_subgroup_sizes; i++) {
    PyTuple_SetItem(subgroup_sizes, i, PyLong_FromLong(1));
  }

  // TODO
  // int mem_clock_rate = pMemoryProperties[0].maxClockRate;
  // int mem_bus_width = pMemoryProperties[0].maxBusWidth;

  return Py_BuildValue(
      "{s:i, s:i, s:i, s:i, s:i, s:i, s:N}", "max_shared_mem", max_shared_mem,
      "multiprocessor_count", multiprocessor_count, "sm_clock_rate",
      sm_clock_rate, "mem_clock_rate", -1, "mem_bus_width", -1,
      "max_work_group_size", max_group_size, "sub_group_sizes", subgroup_sizes);
}

void freeKernel(PyObject *p) {
  delete reinterpret_cast<sycl::kernel *>(PyCapsule_GetPointer(p, "kernel"));
}

void freeKernelBundle(PyObject *p) {
  delete reinterpret_cast<
      sycl::kernel_bundle<sycl::bundle_state::executable> *>(
      PyCapsule_GetPointer(p, "kernel_bundle"));
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  PyObject *quene;
  const char *name, *build_flags;
  int shared;
  PyObject *py_bytes;
  int devId;

  if (!PyArg_ParseTuple(args, "OsSisi", &quene, &name, &py_bytes, &shared,
                        &build_flags, &devId)) {
    std::cerr << "loadBinary arg parse failed" << std::endl;
    return NULL;
  }

  void *queue_ptr = NULL;
  if (!(queue_ptr = PyLong_AsVoidPtr(quene)))
    return NULL;
  sycl::queue *sycl_queue = static_cast<sycl::queue *>(queue_ptr);

  std::call_once(g_sycl_devices_flag, enumDevices);
  if (devId > g_sycl_devices.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }

  const auto &sycl_device = g_sycl_devices[devId];

  std::string kernel_name = name;
  const size_t binary_size = PyBytes_Size(py_bytes);

  uint8_t *binary_ptr = (uint8_t *)PyBytes_AsString(py_bytes);
  // const sycl::context sycl_context =
  //     sycl_device->get_platform().ext_oneapi_get_default_context();
  const sycl::context sycl_context = sycl_queue->get_context();

  const auto ocl_device = sycl::get_native<sycl::backend::opencl>(*sycl_device);
  const auto ocl_context =
      sycl::get_native<sycl::backend::opencl, sycl::context>(sycl_context);

  auto ocl_module = checkSyclErrors(create_module(
      ocl_context, ocl_device, binary_ptr, binary_size, build_flags, true));

  auto checkOCLErrors = [&](auto ocl_module) -> cl_kernel {
    if (PyErr_Occurred()) {
      // check for errors from module creation
      return NULL;
    }
    cl_kernel ocl_kernel =
        checkSyclErrors(create_function(ocl_module, kernel_name));
    if (PyErr_Occurred()) {
      // check for errors from kernel creation
      return NULL;
    }
    return ocl_kernel;
  };

  // Retrieve the kernel properties (e.g. register spills).
  cl_kernel ocl_kernel = checkOCLErrors(ocl_module);

  // auto mod = new sycl::kernel_bundle<sycl::bundle_state::executable>(
  //     sycl::make_kernel_bundle<sycl::backend::opencl,
  //                              sycl::bundle_state::executable>(ocl_module,
  //                                                              sycl_context));
  sycl::kernel *fun = new sycl::kernel(
      sycl::make_kernel<sycl::backend::opencl>(ocl_kernel, sycl_context));
  auto kernel_py =
      PyCapsule_New(reinterpret_cast<void *>(fun), "kernel", freeKernel);
  // auto kernel_bundle_py = PyCapsule_New(reinterpret_cast<void *>(mod),
  //                                       "kernel_bundle", freeKernelBundle);
  // TODO: support `kernel_bundle_py`
  PyObject *kernel_bundle_py = PyTuple_New(0);

  // TODO: support `n_spills` and `n_regs`
  int32_t n_spills = 0;
  const int32_t n_regs = 0;
  return Py_BuildValue("(OOii)", kernel_bundle_py, kernel_py, n_regs, n_spills);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided SPV into ZE driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "spirv_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_spirv_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
