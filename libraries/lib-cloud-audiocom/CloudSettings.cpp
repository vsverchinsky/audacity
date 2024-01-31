/*  SPDX-License-Identifier: GPL-2.0-or-later */
/*!********************************************************************

  Audacity: A Digital Audio Editor

  CloudSettings.cpp

  Dmitry Vedenko

**********************************************************************/
#pragma once

#include "CloudSettings.h"

#include "FileNames.h"
#include "wxFileNameWrapper.h"

namespace cloud::audiocom
{
StringSetting CloudProjectsSavePath {
   "/cloud/audiocom/CloudProjectsSavePath",
   []
   {
      wxFileNameWrapper path { FileNames::DataDir(), "" };
      path.AppendDir("CloudProjects");
      return path.GetPath();
   }
};

IntSetting MixdownGenerationFrequency {
   "/cloud/audiocom/MixdownGenerationFrequency", 1
};

IntSetting DaysToKeepFiles {
   "/cloud/audiocom/DaysToKeepFiles", 30
};
} // namespace cloud::audiocom
