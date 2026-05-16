// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_CONFIG_H
#define AOTRITON_V2_CONFIG_H

#define AOTRITON_ENABLE_SUFFIX 0

#if AOTRITON_ENABLE_SUFFIX
/* #undef AOTRITON_NAME_SUFFIX */
#define AOTRITON_NS aotriton
#else
#define AOTRITON_NAME_SUFFIX
#define AOTRITON_NS aotriton
#endif

#define AOTRITON_VERSION_MAJOR 0u
#define AOTRITON_VERSION_MINOR 11u
#define AOTRITON_VERSION_PATCH 2u
// Note the packaged SHA1 string may be different from the github release tag
// due to chicken-and-egg issue.
#define AOTRITON_GIT_SHA1 dd1b68b604b5258ee7a9f7b66ad95e7a82c18065

#ifdef _MSC_VER
#define AOTRITON_API __declspec(dllexport)
#else
#define AOTRITON_API __attribute__ ((visibility ("default")))
#endif

#endif
