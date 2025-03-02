//===- driver.c -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// TODO: rm
// #include <CL/cl.h>
// #include <sycl/sycl.hpp>

#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __attribute__((visibility("default")))
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
// #include <numpy/arrayobject.h>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

// TODO: print more debug infos if env `TRITON_DEBUG=1`
// TODO: release cl* objects correctly

static std::vector<std::unique_ptr<cl::Device>> g_cl_devices;

// TODO: add & keep same logic in pytorch
void enumDevices() {
  g_cl_devices.push_back(
      std::make_unique<cl::Device>(cl::Device::getDefault()));
}

static auto _tmp_func = []() { enumDevices(); };
static int _tmp_v = (_tmp_func(), 0);

extern "C" EXPORT_FUNC PyObject *get_device_properties(int device_id) {
  if (device_id > g_cl_devices.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }
  const auto &device = g_cl_devices[device_id];

  int multiprocessor_count = device->getInfo<CL_DEVICE_MAX_NUM_SUB_GROUPS>();
  int sm_clock_rate = device->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
  int max_shared_mem = device->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
  int max_group_size = device->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  int mem_clock_rate = device->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
  int mem_bus_width = -1;

  std::vector<size_t> cl_subgroup_sizes =
      device->getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
  int num_subgroup_sizes = cl_subgroup_sizes.size();
  PyObject *subgroup_sizes = PyTuple_New(num_subgroup_sizes);
  for (int i = 0; i < num_subgroup_sizes; i++) {
    PyTuple_SetItem(subgroup_sizes, i, PyLong_FromLong(cl_subgroup_sizes[i]));
  }

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:N}", "max_shared_mem",
                       max_shared_mem, "multiprocessor_count",
                       multiprocessor_count, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width, "max_work_group_size", max_group_size,
                       "sub_group_sizes", subgroup_sizes);
}

extern "C" EXPORT_FUNC PyObject *load_binary(PyObject *args) {
  PyObject *quene_capsule;
  const char *name, *build_flags_ptr;
  int shared;
  PyObject *py_bytes;
  int devId;

  if (!PyArg_ParseTuple(args, "OsSisi", &quene_capsule, &name, &py_bytes,
                        &shared, &build_flags_ptr, &devId)) {
    std::cerr << "loadBinary arg parse failed" << std::endl;
    return NULL;
  }

  if (!PyCapsule_CheckExact(quene_capsule)) {
    return NULL;
  }
  cl::CommandQueue *cl_queue = static_cast<cl::CommandQueue *>(
      PyCapsule_GetPointer(quene_capsule, "clCommandQueue"));
  if (!cl_queue) {
    return nullptr;
  }

  std::cerr << "cl_queue: " << cl_queue << std::endl;
  // auto cl_queue = *cl_queue_ptr;

  // TODO: why `getInfo` fails ?
  cl::Context cl_context = cl_queue->getInfo<CL_QUEUE_CONTEXT>();
  cl::Device cl_dev = cl_queue->getInfo<CL_QUEUE_DEVICE>();
  // cl::Context cl_context = cl::Context::getDefault();
  // cl::Device cl_dev = cl::Device::getDefault();

  std::string kernel_name = name;
  const size_t binary_size = PyBytes_Size(py_bytes);
  uint8_t *binary_ptr = (uint8_t *)PyBytes_AsString(py_bytes);

  assert(binary_ptr != nullptr && "binary_ptr should not be NULL");
  assert(build_flags_ptr != nullptr && "build_flags_ptr should not be NULL");
  cl_program prog =
      clCreateProgramWithIL(cl_context.get(), binary_ptr, binary_size, NULL);
  auto cl_prog = cl::Program(prog, true);
  cl_prog.build(cl_dev, build_flags_ptr);
  auto cl_kernel = new cl::Kernel(cl_prog, kernel_name);
  if (!cl_kernel) {
    delete cl_kernel;
    return nullptr;
  }

  auto free_cl_kernel = [](PyObject *p) {
    delete reinterpret_cast<cl::Kernel *>(PyCapsule_GetPointer(p, "kernel"));
  };
  auto kernel_py = PyCapsule_New(reinterpret_cast<void *>(&cl_kernel), "kernel",
                                 free_cl_kernel);

  // TODO: support `kernel_bundle_py`
  PyObject *kernel_bundle_py = PyTuple_New(0);

  // TODO: support `n_spills` and `n_regs`
  int32_t n_spills = 0;
  const int32_t n_regs = 0;
  return Py_BuildValue("(OOii)", kernel_bundle_py, kernel_py, n_regs, n_spills);
}

extern "C" EXPORT_FUNC PyObject *init_context(PyObject *cap) {
  // Do nothing for now
  auto context = -1;
  return Py_BuildValue("(K)", (uint64_t)context);
}

extern "C" EXPORT_FUNC PyObject *init_devices(PyObject *cap) {
  // Do nothing for now
  const uint32_t deviceCount = g_cl_devices.size();
  return Py_BuildValue("(i)", deviceCount);
}

extern "C" EXPORT_FUNC PyObject *wait_on_sycl_queue(PyObject *cap) {
  void *queue = NULL;
  if (!(queue = PyLong_AsVoidPtr(cap)))
    return NULL;

  auto cl_queue = static_cast<cl::CommandQueue *>(queue);
  cl_queue->finish();

  return Py_None;
}
