#pragma once
#include <chrono>

using NSPACE::vec;
using NSPACE::lbool;

// A singleton which manages TorcOpenWbo's additional (w.r.t OpenWboInc) parameters and functionality
class Torc{
public:
   static Torc* Instance();
   
   void SetPolConservative(bool isConservative) { polIsConservative = isConservative; }
   void SetConservativeAllVars(bool isConservativeAllVars) { conservativeUseAllVars = isConservativeAllVars; }
   void SetPolOptimistic(bool isOptimistic) { polIsOptimistic = isOptimistic;}
   void SetTargetVarsBumpVal(int targetVarsBumpVal) { varTargetsBumpVal = targetVarsBumpVal; }   
   void SetBumpRelWeights(bool isBumpRelWeights) { bumpRelWeights = isBumpRelWeights; }   
   void SetTargetBumpMaxRandVal(int targetVarsBumpRandVal) { varTargetsBumpMaxRandVal = targetVarsBumpRandVal; }   
   void SetMsMutationClasses(int p) { msMutationClasses = p; }
   void SetMsConflictsPerSatCall(int p) { msConflictsPerSatCall = p; }
   void SetMsSatCallsPerEpoch(int p) { msSatCallsPerEpoch = p; }
   void SetMsMaxEpochs(int p) { msMaxEpochs = p; }
   void SetMsDisableTorc(bool p) { msDisableTorc = p; }
   void SetMsStayNearStartingModel(bool p) { msStayNearStartingModel = p; }   
   void SetMsConflictsLookNearMutation(int p) { msConflictsLookNearMutation = p; }
   void SetMsSortLitsStrat(int p) { msSortLitsStrat = p; }
   void SetMsSiftRemainingOnlyWhenOptimized(bool p) { msSiftRemainingOnlyWhenOptimized = p; }
   void SetMsSiftRemaining(bool p) { msSiftRemaining = p; }
   void SetMsAfterRepair(bool p) { msAfterRepair = p; }
   void SetMsModelPerSecThr(double p) { msModelPerSecThr = p; }
   void SetMsModelPerSecOnlyForFirstInv(bool p) { msModelPerSecOnlyForFirstInv = p; }
   void SetMsMutCombUseBestModel(bool p) { msMutCombUseBestModel = p; }
   void SetMsInitBadStrat(int p)  { msInitBadStrat = p; }
   void SetMsRndFreqForMublox(double p)  { msRndFreqForMublox = p; }
   void SetMsMinLBDFrozenClauseForMublox(int p)  { msMinLBDFrozenClauseForMublox = p; }
   void SetMsVerbosity(int p)  { msVerbosity = p; }   
   void SetConflToChrono(int p) { msConflToChrono = p; }
   void SetChrono(int p) { msChrono = p; }
   void SetPoContLocalImpr(int p) { poContLocalImpr = p; }
   void SetMrsBeaverFailWinSizeToSkipOutputs(int p) { mrsBeaverFailWinSizeToSkipOutputs = p; }
   void SetMrsBeaverFailSkipPercent(int p) { mrsBeaverFailSkipPercent = p; }
   void SetMrsBeaverSizeSwitchToComplete(int p) { mrsBeaverSizeSwitchToComplete = p; }
   void SetMrsBeaverEachIterStartBestModel(int p) { mrsBeaverEachIterStartBestModel = p; }
   void SetBlockBestModel(int p) { blockBestModel = p; }
   void SetPolosatTurnOffHighLevelConfThrForIters(int p) { polosatTurnOffHighLevelConfThrForIters = p; }
   void SetOptimisticMaxSoftFraction(double p) { optimisticMaxSoftFraction = p; }
   void SetConservativeMaxSoftFraction(double p) { conservativeMaxSoftFraction = p; }
   void SetWeightedObvBsFirstIterStrat(int p) { weightedObvBsFirstIterStrat = p; }
   void SetWobsAdaptiveStoppedStopMs(int p) { wobsAdaptiveStoppedStopMs = p; }
   void SetWobsAdaptiveNotStoppedMSThrMult(double p) { wobsAdaptiveNotStoppedMSThrMult = p; }
   void SetMsObvStrat(int p) { msObvStrat = p; }
   void SetMrsBeaverPolosatRegulateStrat(int p) { mrsBeaverPolosatRegulateStrat = p; }
   void SetMrsBeaverApplySizeThrDuringInitialPolosat(int p) { mrsBeaverApplySizeThrDuringInitialPolosat = p; }
   void SetPrintEveryModel(int p) { printEveryModel = p; }
   
   bool GetPolConservative() const { return polIsConservative; }
   bool GetConservativeAllVars() const { return conservativeUseAllVars; } 
   bool GetPolOptimistic() const { return polIsOptimistic; }
   int GetTargetVarsBumpVal() const { return varTargetsBumpVal; }   
   bool GetBumpRelWeights() const { return bumpRelWeights; }
   int GetTargetBumpMaxRandVal() const { return varTargetsBumpMaxRandVal; }   
   int GetMsMutationClasses() const { return msMutationClasses; }
   int GetMsConflictsPerSatCall() const { return msConflictsPerSatCall; }
   int GetMsSatCallsPerEpoch() const { return msSatCallsPerEpoch; }
   int GetMsMaxEpochs() const { return msMaxEpochs; }
   bool GetMsDisableTorc() const { return msDisableTorc; }
   bool GetMsStayNearStartingModel() const { return msStayNearStartingModel; }   
   bool GetMsConflictsLookNearMutation() const { return msConflictsLookNearMutation; }   
   int GetMsSortLitsStrat() const { return msSortLitsStrat; }
   bool GetMsSiftRemainingOnlyWhenOptimized() const { return msSiftRemainingOnlyWhenOptimized; }
   bool GetMsSiftRemaining() const { return msSiftRemaining; }
   bool GetMsAfterRepair() const { return msAfterRepair; }
   double GetMsModelPerSecThr() const { return msModelPerSecThr; }
   bool GetMsModelPerSecOnlyForFirstInv() const { return msModelPerSecOnlyForFirstInv; }
   bool GetMsMutCombUseBestModel() const { return msMutCombUseBestModel; }
   int GetMsInitBadStrat() const  { return msInitBadStrat; }
   double GetMsRndFreqForMublox() const  { return msRndFreqForMublox; }
   int GetMsMinLBDFrozenClauseForMublox() const  { return msMinLBDFrozenClauseForMublox; }
   int GetMsVerbosity() const { return msVerbosity; }
   int GetConflToChrono() const { return msConflToChrono; }
   int GetChrono() const { return msChrono; }
   int GetPoContLocalImpr() const { return poContLocalImpr; }
   int GetMrsBeaverFailWinSizeToSkipOutputs() const { return mrsBeaverFailWinSizeToSkipOutputs; }
   int GetMrsBeaverFailSkipPercent() const { return mrsBeaverFailSkipPercent; }
   int GetMrsBeaverSizeSwitchToComplete() const { return mrsBeaverSizeSwitchToComplete; }
   int GetMrsBeaverEachIterStartBestModel() const { return mrsBeaverEachIterStartBestModel; }
   int GetBlockBestModel() const { return blockBestModel; }
   int GetPolosatTurnOffHighLevelConfThrForIters() const { return polosatTurnOffHighLevelConfThrForIters; }
   double GetOptimisticMaxSoftFraction() const { return optimisticMaxSoftFraction; }
   double GetConservativeMaxSoftFraction() const { return conservativeMaxSoftFraction; }
   int GetWeightedObvBsFirstIterStrat() const { return weightedObvBsFirstIterStrat; }
   int GetWobsAdaptiveStoppedStopMs() const { return wobsAdaptiveStoppedStopMs; }
   double GetWobsAdaptiveNotStoppedMSThrMult() const { return wobsAdaptiveNotStoppedMSThrMult; }
   int GetMsObvStrat() const { return msObvStrat; }
   int GetMrsBeaverPolosatRegulateStrat() const { return mrsBeaverPolosatRegulateStrat; }
   int GetMrsBeaverApplySizeThrDuringInitialPolosat() const { return mrsBeaverApplySizeThrDuringInitialPolosat; }
   int GetPrintEveryModel() const { return printEveryModel; }
   int GetRandBump() const;
   
   vec<bool>& TargetIsVarTarget() { return isVarTarget; }
   bool& SkipFillingSolverPolarity() { return skipFillingSolverPolarity; }
   
   double WallTimePassed() const;
private:
   Torc() : timeStart(std::chrono::high_resolution_clock::now()), polIsConservative(true), conservativeUseAllVars(true), polIsOptimistic(true), varTargetsBumpVal(113), bumpRelWeights(false), varTargetsBumpMaxRandVal(0), 	
	msMutationClasses(6), msConflictsPerSatCall(0), msSatCallsPerEpoch(0), msMaxEpochs(0), msDisableTorc(false), msStayNearStartingModel(false), msConflictsLookNearMutation(0), msSortLitsStrat(0), msSiftRemainingOnlyWhenOptimized(false), msSiftRemaining(true), msAfterRepair(false),
	msModelPerSecThr(0.), msModelPerSecOnlyForFirstInv(true), msMutCombUseBestModel(false), msInitBadStrat(1), msRndFreqForMublox(0.), msMinLBDFrozenClauseForMublox(30), msConflToChrono(4000), msChrono(-1), poContLocalImpr(0), 
	mrsBeaverFailWinSizeToSkipOutputs(0), mrsBeaverFailSkipPercent(10), mrsBeaverSizeSwitchToComplete(1000000), mrsBeaverEachIterStartBestModel(0), blockBestModel(0), polosatTurnOffHighLevelConfThrForIters(0), optimisticMaxSoftFraction(1.), conservativeMaxSoftFraction(1.), 
	weightedObvBsFirstIterStrat(0), wobsAdaptiveStoppedStopMs(0), wobsAdaptiveNotStoppedMSThrMult(1.), msObvStrat(3), mrsBeaverPolosatRegulateStrat(0), mrsBeaverApplySizeThrDuringInitialPolosat(0),
	msVerbosity(1), skipFillingSolverPolarity(false), printEveryModel(1) {};  // Private so that it can  not be called
   static Torc* m_pInstance;
   
   std::chrono::high_resolution_clock::time_point timeStart;
   
   bool polIsConservative;
   bool conservativeUseAllVars;
   bool polIsOptimistic;
   int varTargetsBumpVal;  
   bool bumpRelWeights;    
   int varTargetsBumpMaxRandVal;  
   
   // Mutation-based optimizing-sampling after each model
   int msMutationClasses;
   int msConflictsPerSatCall;
   int msSatCallsPerEpoch;
   int msMaxEpochs;
   bool msDisableTorc;
   bool msStayNearStartingModel;
   int msConflictsLookNearMutation;
   int msSortLitsStrat; 
   bool msSiftRemainingOnlyWhenOptimized;
   bool msSiftRemaining;
   bool msAfterRepair;
   // Stop mutation-optimization forever when the improving models-per-second threshold is too low
   double msModelPerSecThr;
   bool msModelPerSecOnlyForFirstInv;
   bool msMutCombUseBestModel;
   int msInitBadStrat; 
   double msRndFreqForMublox;
   int msMinLBDFrozenClauseForMublox;
   int msConflToChrono;
   int msChrono;
   int poContLocalImpr;
   int mrsBeaverFailWinSizeToSkipOutputs;
   int mrsBeaverFailSkipPercent;
   int mrsBeaverSizeSwitchToComplete;
   int mrsBeaverEachIterStartBestModel;   
   int blockBestModel;
   int polosatTurnOffHighLevelConfThrForIters;
   double optimisticMaxSoftFraction;
   double conservativeMaxSoftFraction;
   int weightedObvBsFirstIterStrat;
   int wobsAdaptiveStoppedStopMs;
   double wobsAdaptiveNotStoppedMSThrMult;
   int msObvStrat;
   int mrsBeaverPolosatRegulateStrat;
   int mrsBeaverApplySizeThrDuringInitialPolosat;
   int printEveryModel;
   
   int msVerbosity;
   
   vec<bool> isVarTarget;
   bool skipFillingSolverPolarity;   
};


