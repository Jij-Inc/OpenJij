include(FetchContent)

message(CHECK_START "Fetching GoogleTest")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

set(CMAKE_CXX_STANDARD 17)
set(FETCHCONTENT_QUIET OFF)

#### Google test ####
FetchContent_Declare(
    googletest
    GIT_REPOSITORY  https://github.com/google/googletest
    GIT_TAG         v1.17.0
    GIT_SHALLOW     TRUE
    FIND_PACKAGE_ARGS NAMES GTest
    EXCLUDE_FROM_ALL
)

if(WIN32)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

# FetchContent_MakeAvailableを使用（CMake 3.24+推奨）
# FETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYSにより、
# まずfind_package(GTest)を試み、見つからない場合のみダウンロード
FetchContent_MakeAvailable(googletest)

#FetchContent_GetProperties(googletest)

#message(STATUS "gtest_SOURCE_DIR = ${gtest_SOURCE_DIR}")
#message(STATUS "gmock_SOURCE_DIR = ${gmock_SOURCE_DIR}")


list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "fetched")
