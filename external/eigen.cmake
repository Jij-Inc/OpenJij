include(FetchContent)

message(CHECK_START "Fetching Eigen3")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

set(FETCHCONTENT_QUIET OFF)

set(BUILD_TESTING OFF) 
set(TEST_LIB OFF)

set(EIGEN_MPL2_ONLY ON CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_DOC_USE_MATHJAX OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_TEST_NOQT OFF CACHE BOOL "" FORCE)
set(EIGEN_LEAVE_TEST_IN_ALL_TARGET OFF CACHE BOOL "" FORCE)


# Disable installation of Eigen3 headers to prevent them from being included in the wheel
set(EIGEN_INSTALL_HEADERS OFF CACHE BOOL "" FORCE)


if(MSVC)
  set(EIGEN_Fortran_COMPILER_WORKS OFF CACHE BOOL "" FORCE)
endif()

if(BLAS_FOUND) 
  set(EIGEN_USE_BLAS ON) 
endif()

if(LAPACK_FOUND) 
  set(EIGEN_USE_LAPACKE ON)
endif()

#### Eigen ####
FetchContent_Declare(
    eigen3
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen
    GIT_TAG         3.4.1
    GIT_SHALLOW     TRUE
    )

# Prevent Eigen from being installed by using FetchContent_Populate + add_subdirectory with EXCLUDE_FROM_ALL
FetchContent_GetProperties(eigen3)
if(NOT eigen3_POPULATED)
    FetchContent_Populate(eigen3)
    add_subdirectory(${eigen3_SOURCE_DIR} ${eigen3_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_library(openjij-eigen_lib INTERFACE)
target_include_directories(openjij-eigen_lib INTERFACE ${eigen3_SOURCE_DIR})
target_compile_definitions(openjij-eigen_lib INTERFACE 
    EIGEN_MPL2_ONLY
    BUILD_TESTING=OFF
    TEST_LIB=OFF
    EIGEN_BUILD_PKGCONFIG=OFF
    EIGEN_BUILD_DOC=OFF
    EIGEN_DOC_USE_MATHJAX=OFF 
    EIGEN_BUILD_TESTING=OFF 
    EIGEN_TEST_NOQT=OFF 
    EIGEN_LEAVE_TEST_IN_ALL_TARGET=OFF 
    $<$<TARGET_EXISTS:BLAS::BLAS>:EIGEN_USE_BLAS>
    $<$<TARGET_EXISTS:LAPACK::LAPACK>:EIGEN_USE_LAPACKE>
    $<$<CXX_COMPILER_ID:MSVC>:EIGEN_Fortran_COMPILER_WORKS=OFF>
)
  
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "fetched")
