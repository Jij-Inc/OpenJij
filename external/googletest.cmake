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
)

if(WIN32)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

# Prevent googletest from being installed by using FetchContent_Populate + add_subdirectory with EXCLUDE_FROM_ALL
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

find_package(GTest)

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "fetched")
