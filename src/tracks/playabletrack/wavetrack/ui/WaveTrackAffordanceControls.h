/*!********************************************************************
*
 Audacity: A Digital Audio Editor

 WaveTrackAffordanceControls.h

 Vitaly Sverchinsky

 **********************************************************************/

#pragma once

#include <wx/font.h>
#include <wx/event.h>

#include "../../../ui/CommonTrackPanelCell.h"
#include "../../../ui/TextEditHelper.h"


struct TrackListEvent;

class AffordanceHandle;
class WaveClip;
class TrackPanelResizeHandle;
class WaveClipTitleEditHandle;
class WaveTrackAffordanceHandle;
class TrackList;

//Handles clip movement, selection, navigation and
//allow name change
class AUDACITY_DLL_API WaveTrackAffordanceControls : 
    public CommonTrackCell,
    public TextEditDelegate,
    public wxEvtHandler,
    public std::enable_shared_from_this<WaveTrackAffordanceControls>
{
    std::weak_ptr<WaveClip> mFocusClip;
    std::weak_ptr<WaveTrackAffordanceHandle> mAffordanceHandle;
    std::weak_ptr<TrackPanelResizeHandle> mResizeHandle;
    std::weak_ptr<WaveClipTitleEditHandle> mTitleEditHandle;

    std::shared_ptr<TextEditHelper> mTextEditHelper;

    wxFont mClipNameFont;

public:
    WaveTrackAffordanceControls(const std::shared_ptr<Track>& pTrack);

    std::vector<UIHandlePtr> HitTest(const TrackPanelMouseState& state, const AudacityProject* pProject) override;

    void Draw(TrackPanelDrawingContext& context, const wxRect& rect, unsigned iPass) override;

    //Invokes name editing for a clip that currently is
    //in focus(as a result of hit testing), returns true on success
    //false if there is no focus
    bool StartEditClipName(AudacityProject* project);

    std::weak_ptr<WaveClip> GetSelectedClip() const;

    unsigned CaptureKey
    (wxKeyEvent& event, ViewInfo& viewInfo, wxWindow* pParent,
        AudacityProject* project) override;
    
    unsigned KeyDown (wxKeyEvent& event, ViewInfo& viewInfo, wxWindow* pParent,
        AudacityProject* project) override;

    unsigned Char
    (wxKeyEvent& event, ViewInfo& viewInfo, wxWindow* pParent,
        AudacityProject* project) override;

    void OnTextEditFinished(AudacityProject* project, const wxString& text) override;
    void OnTextEditCancelled(AudacityProject* project) override;
    void OnTextModified(AudacityProject* project, const wxString& text) override;
    void OnTextContextMenu(AudacityProject* project, const wxPoint& position) override;

private:
    void OnTrackChanged(TrackListEvent& evt);

    bool SelectNextClip(ViewInfo& viewInfo, AudacityProject* project, bool forward);
    bool StartEditSelectedClipName(ViewInfo& viewInfo, AudacityProject* project);

    std::shared_ptr<TextEditHelper> MakeTextEditHelper(const wxString& text);
};
