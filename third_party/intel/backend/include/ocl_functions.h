#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

#include <sycl/sycl.hpp>

inline std::string parseOclResultCode(const cl_int code) {
  const std::string prefix = "Triton Error [OCL]: ";
  std::stringstream ss;
  ss << prefix << "0x" << std::hex << code << "\n";
  return ss.str();
}

#define CL_CHECK(code)                                                         \
  {                                                                            \
    if (code != CL_SUCCESS) {                                                  \
      return std::make_tuple(nullptr, code);                                   \
    }                                                                          \
  }

std::tuple<cl_program, cl_int>
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
  // clRetainProgram(module);
  CL_CHECK(clBuildProgram(module, 1, &device, nullptr, nullptr, nullptr));
  return std::make_tuple(module, error_no);
}

std::tuple<cl_kernel, cl_int> create_function(cl_program module,
                                              std::string_view func_name) {
  cl_int error_no;
  cl_kernel kernel = clCreateKernel(module, func_name.data(), &error_no);
  CL_CHECK(error_no);
  // clRetainKernel(kernel);
  return std::make_tuple(kernel, CL_SUCCESS);
}
