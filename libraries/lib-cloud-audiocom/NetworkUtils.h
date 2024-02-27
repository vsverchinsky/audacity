/*  SPDX-License-Identifier: GPL-2.0-or-later */
/*!********************************************************************

  Audacity: A Digital Audio Editor

  NetworkUtils.h

  Dmitry Vedenko

**********************************************************************/

#pragma once

#include <chrono>
#include <string>

namespace audacity::network_manager
{
class IResponse;
class Request;
} // namespace audacity::network_manager

namespace audacity::cloud::audiocom
{
struct CLOUD_AUDIOCOM_API TransferStats final
{
   using Duration = std::chrono::milliseconds;

   int64_t BytesTransferred {};
   int64_t BlocksTransferred {};
   int64_t ProjectFilesTransferred {};

   Duration TransferDuration {};

   TransferStats& SetBytesTransferred(int64_t bytesTransferred);
   TransferStats& SetBlocksTransferred(int64_t blocksTransferred);
   TransferStats& SetProjectFilesTransferred(int64_t projectFilesTransferred);
   TransferStats& SetTransferDuration(Duration transferDuration);
}; // struct TransferStats


enum class ResponseResultCode
{
   Success,
   Cancelled,
   Expired,
   Conflict,
   ConnectionFailed,
   PaymentRequired,
   TooLarge,
   Unauthorized,
   Forbidden,
   NotFound,
   UnexpectedResponse,
   InternalClientError,
   InternalServerError,
   UnknownError,
}; // enum class ResponseResultCode

struct ResponseResult final
{
   ResponseResultCode Code { ResponseResultCode::Success };
   std::string Content;
}; // struct ResponseResult

[[nodiscard]] ResponseResult GetResponseResult(
   audacity::network_manager::IResponse& response, bool readBody);

void SetCommonHeaders(audacity::network_manager::Request& request);
} // namespace audacity::cloud::audiocom
