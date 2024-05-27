/*
* Audacity: A Digital Audio Editor
*/
#ifndef AU_PROJECTSCENE_PROJECTSCENEUIACTIONS_H
#define AU_PROJECTSCENE_PROJECTSCENEUIACTIONS_H

#include "ui/iuiactionsmodule.h"
#include "modularity/ioc.h"
#include "context/iuicontextresolver.h"
#include "async/asyncable.h"

#include "projectsceneactionscontroller.h"

namespace au::projectscene {
class ProjectSceneUiActions : public muse::ui::IUiActionsModule, public muse::async::Asyncable
{
    muse::Inject<mu::context::IUiContextResolver> uicontextResolver;

public:
    ProjectSceneUiActions(std::shared_ptr<ProjectSceneActionsController> controller);

    const muse::ui::UiActionList& actionsList() const override;

    bool actionEnabled(const muse::ui::UiAction& act) const override;
    muse::async::Channel<muse::actions::ActionCodeList> actionEnabledChanged() const override;

    bool actionChecked(const muse::ui::UiAction& act) const override;
    muse::async::Channel<muse::actions::ActionCodeList> actionCheckedChanged() const override;

private:
    std::shared_ptr<ProjectSceneActionsController> m_controller;
    static const muse::ui::UiActionList m_actions;
};
}

#endif // AU_PROJECTSCENE_PROJECTSCENEUIACTIONS_H
