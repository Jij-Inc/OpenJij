############################################################################
# Copyright (c) 2019, Martin Renou                                         #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

# pybind11_json cmake module
# This module sets the following variables in your project::
#
#   pybind11_json_FOUND - true if pybind11_json found on the system
#   pybind11_json_INCLUDE_DIRS - the directory containing pybind11_json headers
#   pybind11_json_LIBRARY - empty


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was pybind11_jsonConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################



include(CMakeFindDependencyMacro)
find_dependency(pybind11 2.2.4)
find_dependency(nlohmann_json 3.2.0)

if(NOT TARGET pybind11_json)
  include("${CMAKE_CURRENT_LIST_DIR}/pybind11_jsonTargets.cmake")

  get_target_property(pybind11_json_INCLUDE_DIRS pybind11_json INTERFACE_INCLUDE_DIRECTORIES)
endif()
