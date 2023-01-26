/**********************************************************************

  Audacity: A Digital Audio Editor

  Effect.h

  Dominic Mazzoni
  Vaughan Johnson

**********************************************************************/

#ifndef __AUDACITY_EFFECT__
#define __AUDACITY_EFFECT__

#include "EffectBase.h"

#include "StatefulEffectBase.h"

#define BUILTIN_EFFECT_PREFIX wxT("Built-in Effect: ")

class EffectParameterMethods;
class WaveTrack;

class sampleCount;

//! Default implementation of EffectUIValidator invokes ValidateUI
//! method of an EffectUIServices
/*
 Also pops the even handler stack of a window, if given to the contructor

 This is a transitional class; it should be eliminated when all effect classes
 define their own associated subclasses of EffectUIValidator, which can hold
 state only for the lifetime of a dialog, so the effect object need not hold it
*/
class DefaultEffectUIValidator
   : public EffectUIValidator
   // Inherit wxEvtHandler so that Un-Bind()-ing is automatic in the destructor
   , protected wxEvtHandler
{
public:
   /*!
    @param pParent if not null, caller will push an event handler onto this
    window; then this object is responsible to pop it
    */
   DefaultEffectUIValidator(const EffectPlugin &plugin,
      EffectUIServices &services, EffectSettingsAccess &access,
      wxWindow *pParent = nullptr);
   //! Calls Disconnect
   ~DefaultEffectUIValidator() override;
   //! Calls mUIServices.ValidateUI()
   bool ValidateUI() override;
   void Disconnect() override;
protected:
   const EffectPlugin &mPlugin;
   wxWindow *mpParent{};
};

class AUDACITY_DLL_API Effect /* not final */
   : public wxEvtHandler
   , public EffectBase
{
 //
 // public methods
 //
 // Used by the outside program to determine properties of an effect and
 // apply the effect to one or more tracks.
 //
 public:
   static inline Effect *FetchParameters(Effect &e, EffectSettings &)
   { return &e; }

   // The constructor is called once by each subclass at the beginning of the program.
   // Avoid allocating memory or doing time-consuming processing here.
   Effect();
   virtual ~Effect();

   // ComponentInterface implementation

   PluginPath GetPath() const override;
   bool VisitSettings(
      SettingsVisitor &visitor, EffectSettings &settings) override;
   bool VisitSettings(
      ConstSettingsVisitor &visitor, const EffectSettings &settings)
      const override;

   ComponentInterfaceSymbol GetSymbol() const override;

   VendorSymbol GetVendor() const override;
   wxString GetVersion() const override;
   TranslatableString GetDescription() const override;

   // EffectDefinitionInterface implementation

   EffectType GetType() const override;
   EffectFamilySymbol GetFamily() const override;
   bool IsInteractive() const override;
   bool IsDefault() const override;
   RealtimeSince RealtimeSupport() const override;
   bool SupportsAutomation() const override;

   bool SaveSettings(
      const EffectSettings &settings, CommandParameters & parms) const override;
   bool LoadSettings(
      const CommandParameters & parms, EffectSettings &settings) const override;

   OptionalMessage LoadUserPreset(
      const RegistryPath & name, EffectSettings &settings) const override;
   bool SaveUserPreset(
      const RegistryPath & name, const EffectSettings &settings) const override;

   RegistryPaths GetFactoryPresets() const override;
   OptionalMessage LoadFactoryPreset(int id, EffectSettings &settings)
      const override;
   OptionalMessage LoadFactoryDefaults(EffectSettings &settings)
      const override;

   // VisitSettings(), SaveSettings(), and LoadSettings()
   // use the functions of EffectParameterMethods.  By default, this function
   // defines an empty list of parameters.
   virtual const EffectParameterMethods &Parameters() const;

   int ShowClientInterface(const EffectPlugin &plugin, wxWindow &parent,
      wxDialog &dialog, EffectUIValidator *pValidator, bool forceModal)
   const override;

   EffectUIServices* GetEffectUIServices() override;

   std::unique_ptr<EffectUIValidator> PopulateUI(const EffectPlugin &plugin,
      ShuttleGui &S, EffectInstance &instance, EffectSettingsAccess &access,
      const EffectOutputs *pOutputs) override;
   //! @return false
   bool ValidateUI(const EffectPlugin &context, EffectSettings &) override;
   bool CloseUI() const override;

   bool CanExportPresets() const override;
   void ExportPresets(
      const EffectPlugin &plugin, const EffectSettings &settings)
   const override;
   OptionalMessage ImportPresets(
      const EffectPlugin &plugin, EffectSettings &settings) const override;

   bool HasOptions() const override;
   void ShowOptions(const EffectPlugin &plugin) const override;

   // EffectPlugin implementation

   const EffectSettingsManager& GetDefinition() const override;
   virtual NumericFormatSymbol GetSelectionFormat() /* not override? */; // time format in Selection toolbar

   // EffectPlugin implementation

   int ShowHostInterface( wxWindow &parent,
      const EffectDialogFactory &factory,
      std::shared_ptr<EffectInstance> &pInstance, EffectSettingsAccess &access,
      bool forceModal = false) override;
   bool SaveSettingsAsString(
      const EffectSettings &settings, wxString & parms) const override;
   [[nodiscard]] OptionalMessage LoadSettingsFromString(
      const wxString & parms, EffectSettings &settings) const override;
   bool IsBatchProcessing() const override;
   void SetBatchProcessing() override;
   void UnsetBatchProcessing() override;
   bool TransferDataToWindow(const EffectSettings &settings) override;
   bool TransferDataFromWindow(EffectSettings &settings) override;

   // Effect implementation

   unsigned TestUIFlags(unsigned mask);

   //! Re-invoke DoEffect on another Effect object that implements the work
   bool Delegate(Effect &delegate, EffectSettings &settings);

   // Display a message box, using effect's (translated) name as the prefix
   // for the title.
   enum : long { DefaultMessageBoxStyle = wxOK | wxCENTRE };
   int MessageBox(const TranslatableString& message,
                  long style = DefaultMessageBoxStyle,
                  const TranslatableString& titleStr = {}) const;

   static void IncEffectCounter(){ nEffectsDone++;}

protected:

   //! Default implementation returns false
   bool CheckWhetherSkipEffect(const EffectSettings &settings) const override;

   //! Default implementation returns `previewLength`
   double CalcPreviewInputLength(
      const EffectSettings &settings, double previewLength) const override;

   //! Called only from PopulateUI, to add controls to effect panel
   /*!
    @return also returned from PopulateUI
    @post `result: result != nullptr`
    */
   virtual std::unique_ptr<EffectUIValidator> MakeEditor(
      ShuttleGui & S, EffectInstance &instance, EffectSettingsAccess &access,
      const EffectOutputs *pOutputs) = 0;

   // No more virtuals!

   // The Progress methods all return true if the user has cancelled;
   // you should exit immediately if this happens (cleaning up memory
   // is okay, but don't try to undo).

   // Pass a fraction between 0.0 and 1.0
   bool TotalProgress(double frac, const TranslatableString & = {}) const;

   // Pass a fraction between 0.0 and 1.0, for the current track
   // (when doing one track at a time)
   bool TrackProgress(
      int whichTrack, double frac, const TranslatableString & = {}) const;

   // Pass a fraction between 0.0 and 1.0, for the current track group
   // (when doing stereo groups at a time)
   bool TrackGroupProgress(
      int whichGroup, double frac, const TranslatableString & = {}) const;

   int GetNumWaveTracks() const { return mNumTracks; }
   int GetNumWaveGroups() const { return mNumGroups; }

   // Calculates the start time and length in samples for one or two channels
   void GetBounds(
      const WaveTrack &track, const WaveTrack *pRight,
      sampleCount *start, sampleCount *len);

   // Use this method to copy the input tracks to mOutputTracks, if
   // doing the processing on them, and replacing the originals only on success (and not cancel).
   // If not all sync-locked selected, then only selected wave tracks.
   void CopyInputTracks(bool allSyncLockSelected = false);

   // Use this to append a NEW output track.
   Track *AddToOutputTracks(const std::shared_ptr<Track> &t);

private:
   wxString GetSavedStateGroup();

   bool mIsBatch{ false };
};

//! Convenience for generating EffectDefinitionInterface overrides
//! and static down-casting functions
template<typename Settings, typename Base>
class EffectWithSettings : public Base {
public:
   EffectSettings MakeSettings() const override
   {
      return EffectSettings::Make<Settings>();
   }
   bool CopySettingsContents(
      const EffectSettings &src, EffectSettings &dst) const override
   {
      return EffectSettings::Copy<Settings>(src, dst);
   }
   //! Assume settings originated from MakeSettings() and copies thereof
   static inline Settings &GetSettings(EffectSettings &settings)
   {
      auto pSettings = settings.cast<Settings>();
      assert(pSettings);
      return *pSettings;
   }
   //! Assume settings originated from MakeSettings() and copies thereof
   static inline const Settings &GetSettings(const EffectSettings &settings)
   {
      return GetSettings(const_cast<EffectSettings &>(settings));
   }
   static inline Settings *
   FetchParameters(Base &, EffectSettings &s) {
      return &GetSettings(s);
   }
};

//! Subclass of Effect, to be eliminated after all of its subclasses
//! are rewritten to be stateless
class StatefulEffect
   : public StatefulEffectBase
   , public Effect
{
public:
   class AUDACITY_DLL_API Instance : public StatefulEffectBase::Instance {
   public:
      using StatefulEffectBase::Instance::Instance;
      bool Process(EffectSettings &settings) override;
      SampleCount GetLatency(
         const EffectSettings &settings, double sampleRate) const override;
      //! Default implementation fails (returns 0 always)
      size_t ProcessBlock(EffectSettings &settings,
         const float *const *inBlock, float *const *outBlock, size_t blockLen)
      override;
   };

   ~StatefulEffect() override;

   std::shared_ptr<EffectInstance> MakeInstance() const override;

   //! Allows PopulateOrExchange to return null
   std::unique_ptr<EffectUIValidator> PopulateUI(const EffectPlugin &plugin,
      ShuttleGui &S, EffectInstance &instance, EffectSettingsAccess &access,
      const EffectOutputs *pOutputs) override;

private:
   //! Needed to make subclasses concrete, but should never be called
   virtual std::unique_ptr<EffectUIValidator> MakeEditor(
      ShuttleGui & S, EffectInstance &instance, EffectSettingsAccess &access,
      const EffectOutputs *pOutputs) final;
};

// FIXME:
// FIXME:  Remove this once all effects are using the NEW dialog
// FIXME:

#define ID_EFFECT_PREVIEW ePreviewID

#endif
