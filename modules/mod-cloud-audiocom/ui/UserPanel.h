/*  SPDX-License-Identifier: GPL-2.0-or-later */
/*!********************************************************************

  Audacity: A Digital Audio Editor

  UserPanel.h

  Dmitry Vedenko

**********************************************************************/
#pragma once

#include "wxPanelWrapper.h"
#include "Observer.h"

class wxStaticText;
class wxButton;

namespace cloud::audiocom
{
class ServiceConfig;
class OAuthService;
class UserService;
class UserImage;

struct UserPanelStateChangedMessage final
{
   bool IsAuthorized;
};

class UserPanel final
   : public wxPanelWrapper
   , public Observer::Publisher<UserPanelStateChangedMessage>
{
public:
   UserPanel(
      const ServiceConfig& serviceConfig,
      OAuthService& authService, UserService& userService,
      bool hasLinkButton, wxWindow* parent = nullptr,
      const wxPoint& pos = wxDefaultPosition,
      const wxSize& size = wxDefaultSize);

   ~UserPanel() override;

private:
   void UpdateUserData();
   void OnLinkButtonPressed();
   void SetAnonymousState();

   const ServiceConfig& mServiceConfig;
   OAuthService& mAuthService;
   UserService& mUserService;

   UserImage* mUserImage {};
   wxStaticText* mUserName {};
   wxButton* mLinkButton {};

   Observer::Subscription mUserDataChangedSubscription;
}; // class UserPanel

} // namespace cloud::audiocom
