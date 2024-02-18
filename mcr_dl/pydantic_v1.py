# Copyright 2023, The Ohio State University. All rights reserved.
# The MVAPICH software package is developed by the team members of
# The Ohio State University's Network-Based Computing Laboratory (NBCL),
# headed by Professor Dhabaleswar K. (DK) Panda.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pydantic v1 compatibility module.

Pydantic v2 introduced breaking changes that hinder its adoption:
https://docs.pydantic.dev/latest/migration/. To provide mcr-dl users the option to
migrate to pydantic v2 on their own timeline, mcr-dl uses this compatibility module
as a pydantic-version-agnostic alias for pydantic's v1 API.
"""

try:
    from pydantic.v1 import *  # noqa: F401
except ImportError:
    from pydantic import *  # noqa: F401
