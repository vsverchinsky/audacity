/*!********************************************************************
*
 Audacity: A Digital Audio Editor

 @file WaveTrackAffordanceHandle.cpp

 Vitaly Sverchinsky

 **********************************************************************/

#include "WaveTrackAffordanceHandle.h"
#include "WaveTrackAffordanceControls.h"
#include "WaveChannelView.h"
#include "ViewInfo.h"

#include "WaveClip.h"
#include "../../../../RefreshCode.h"
#include "../../../../TrackPanelMouseEvent.h"
#include "ProjectHistory.h"

#include <wx/event.h>

WaveTrackAffordanceHandle::WaveTrackAffordanceHandle(const std::shared_ptr<Track>& track,
                                                     const std::shared_ptr<WaveTrack::Interval>& target)
   : AffordanceHandle(track)
   , mTarget(target)
{ }

UIHandle::Result WaveTrackAffordanceHandle::Click(const TrackPanelMouseEvent& event, AudacityProject* project)
{
   Result result = RefreshCode::RefreshNone;
   if (WaveChannelView::ClipDetailsVisible(*mTarget->GetClip(0), ViewInfo::Get(*project), event.rect))
   {
      auto affordanceControl = std::dynamic_pointer_cast<WaveTrackAffordanceControls>(event.pCell);

      if (affordanceControl)
      {
         result |= affordanceControl->OnAffordanceClick(event, project);
         if (!event.event.GetSkipped())//event is "consumed"
            return result | RefreshCode::Cancelled;
         event.event.Skip(false);
      }
   }
   return result | AffordanceHandle::Click(event, project);
}

UIHandle::Result WaveTrackAffordanceHandle::SelectAt(const TrackPanelMouseEvent& event, AudacityProject* project)
{
   auto& viewInfo = ViewInfo::Get(*project);
   viewInfo.selectedRegion.setTimes(mTarget->Start(), mTarget->End());

   ProjectHistory::Get(*project).ModifyState(false);

   return RefreshCode::RefreshAll | RefreshCode::Cancelled;
}
