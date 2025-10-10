include(FetchContent)

message(CHECK_START "Fetching JSON")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

set(BUILD_TESTING OFF)

# Set policy version minimum to fix compatibility issue with CMake 4.0.0
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum CMake policy version" FORCE)

#### nlohmann_json ####
FetchContent_Declare(
     nlohmann_json
     GIT_REPOSITORY  https://github.com/nlohmann/json
     GIT_TAG         v3.12.0
     GIT_SHALLOW     TRUE
     )

FetchContent_GetProperties(nlohmann_json)

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "fetched")
