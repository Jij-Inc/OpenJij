# Copyright 2023 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(FetchContent)

message(CHECK_START "cimod")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

set(FETCHCONTENT_QUIET OFF)


#### Cimod ####
FetchContent_Declare(
    cimod
    GIT_REPOSITORY  https://github.com/Jij-Inc/cimod
    GIT_TAG         v1.7.2
    GIT_SHALLOW     TRUE
    )

# Prevent Cimod from being installed by using FetchContent_Populate + add_subdirectory with EXCLUDE_FROM_ALL
FetchContent_GetProperties(cimod)
if(NOT cimod_POPULATED)
    FetchContent_Populate(cimod)
    add_subdirectory(${cimod_SOURCE_DIR} ${cimod_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "fetched")
