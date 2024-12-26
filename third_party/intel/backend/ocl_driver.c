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
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <CL/cl.h>
#include <sycl/sycl.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
// #include <numpy/arrayobject.h>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include "opencl.hpp"

// TODO: print more debug infos if env `TRITON_DEBUG=1`
// TODO: release cl* objects correctly

static std::vector<std::unique_ptr<sycl::device>> g_sycl_devices;

#define CL_CHECK(code)                                                         \
  {                                                                            \
    if (code != CL_SUCCESS) {                                                  \
      return std::make_tuple(nullptr, code, #code);                            \
    }                                                                          \
  }

std::tuple<cl_program, cl_int, std::string>
create_module(cl_context context, cl_device_id device, uint8_t *binary_ptr,
              size_t binary_size, const char *build_flags,
              const bool is_spv = true) {
  assert(binary_ptr != nullptr && "binary_ptr should not be NULL");
  assert(build_flags != nullptr && "build_flags should not be NULL");
  assert(is_spv == true && "is_spv should be true");

  cl_int error_no;
  cl_program module =
      clCreateProgramWithIL(context, binary_ptr, binary_size, &error_no);
  CL_CHECK(error_no);
  CL_CHECK(clBuildProgram(module, 1, &device, nullptr, nullptr, nullptr));
  return std::make_tuple(module, error_no, __FUNCTION__);
}

std::tuple<cl_kernel, cl_int, std::string>
create_function(cl_program module, std::string_view func_name) {
  cl_int error_no;
  cl_kernel kernel = clCreateKernel(module, func_name.data(), &error_no);
  CL_CHECK(error_no);
  return std::make_tuple(kernel, error_no, __FUNCTION__);
}

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

static auto _tmp_func = []() { enumDevices(); };
static int _tmp_v = (_tmp_func(), 0);

template <typename T>
static inline T
checkSyclErrors(const std::tuple<T, cl_int, std::string> tuple) {
  const auto code = std::get<1>(tuple);
  if (code != CL_SUCCESS) {
    const auto msg = std::get<2>(tuple);
    std::stringstream ss;
    ss << "Triton Error [OCL]: " << "0x" << std::hex << code << ", " << msg
       << "\n";
    auto str = ss.str();
    throw std::runtime_error(ss.str());
  }
  return std::get<0>(tuple);
}

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  if (device_id > g_sycl_devices.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }
  const auto &sycl_device = g_sycl_devices[device_id];
  cl_device_id ocl_device =
      sycl::get_native<sycl::backend::opencl>(*sycl_device);

  cl::Device d = cl::Device(ocl_device);
  int multiprocessor_count = d.getInfo<CL_DEVICE_MAX_NUM_SUB_GROUPS>();
  int sm_clock_rate = d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
  int max_shared_mem = d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
  int max_group_size = d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  int mem_clock_rate = d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
  int mem_bus_width = -1;

  std::vector<size_t> cl_subgroup_sizes =
      d.getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
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

  if (devId > g_sycl_devices.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }
  const auto &sycl_device = g_sycl_devices[devId];

  std::string kernel_name = name;
  const size_t binary_size = PyBytes_Size(py_bytes);
  uint8_t *binary_ptr = (uint8_t *)PyBytes_AsString(py_bytes);
  const sycl::context sycl_context = sycl_queue->get_context();
  const auto ocl_context =
      sycl::get_native<sycl::backend::opencl, sycl::context>(sycl_context);
  const auto ocl_device = sycl::get_native<sycl::backend::opencl>(*sycl_device);

  auto ocl_module = checkSyclErrors(create_module(
      ocl_context, ocl_device, binary_ptr, binary_size, build_flags, true));
  auto ocl_kernel = checkSyclErrors(create_function(ocl_module, kernel_name));

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
     "Load provided SPV into OpenCL driver"},
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
