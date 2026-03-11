
/*!
 * \author Ruben Martins - ruben@sat.inesc-id.pt
 *
 * @section LICENSE
 *
 * MiniSat,  Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
 *           Copyright (c) 2007-2010, Niklas Sorensson
 * Open-WBO, Copyright (c) 2013-2017, Ruben Martins, Vasco Manquinho, Ines Lynce
 * NuWLS -- Copyright (c) 2021-2022, Yi Chu, Xiang He
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include "utils/Options.h"
#include "utils/ParseUtils.h"
#include "utils/System.h"
#include <errno.h>
#include <signal.h>
#include <zlib.h>

#include <fstream>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <string>
#include <vector>

#ifdef SIMP
#include "simp/SimpSolver.h"
#else
#include "core/Solver.h"
#endif

#include "MaxSAT.h"
#include "MaxTypes.h"
#include "ParserMaxSAT.h"
#include "ParserPB.h"
#include "Torc.h"

// Algorithms
#include "algorithms/Alg_LinearSU.h"
#include "algorithms/Alg_LinearSU_Clustering.h"
#include "algorithms/Alg_LinearSU_Mod.h"
#include "algorithms/Alg_MSU3.h"
#include "algorithms/Alg_OLL.h"
#include "algorithms/Alg_OLL_Mod.h"
#include "algorithms/Alg_PartMSU3.h"
#include "algorithms/Alg_WBO.h"
#include "algorithms/Alg_OBV.h"
#include "algorithms/Alg_BLS.h"

#define VER1_(x) #x
#define VER_(x) VER1_(x)
#define SATVER VER_(SOLVERNAME)
#define VER VER_(VERSION)

using NSPACE::BoolOption;
using NSPACE::DoubleOption;
using NSPACE::IntOption;
using NSPACE::IntRange;
using NSPACE::DoubleRange;
using NSPACE::OutOfMemoryException;
using NSPACE::StringOption;
using NSPACE::cpuTime;
using NSPACE::parseOptions;
using namespace openwbo;

//=================================================================================================

static MaxSAT *mxsolver;

static void SIGINT_exit(int signum) {
  mxsolver->printAnswer(_UNKNOWN_);
  exit(_UNKNOWN_);
}

#include "Test.h"

// void test_encoding();

//=================================================================================================
// Main:

int main(int argc, char **argv) {
  printf(
      "c\nc NuWLS based on TT-Open-WBO-Inc:\t an Anytime MaxSAT Solver --- based on %s (%s version)\n",
      SATVER, VER);
  printf("c Version:\t MaxSAT Evaluation 2021\n");
  printf("c Author of NuWLS:\t Yi Chu, Xiang He\n");
  printf("c Author of TT-Open-WBO-Inc:\t Alexander Nadel\n");
  printf("c Authors of the baseline solver Open-WBO-Inc:\t Saurabh Joshi, Prateek Kumar, Ruben Martins, Sukrut Rao\n");
  try {
    NSPACE::setUsageHelp("c USAGE: %s [options] <input-file>\n\n");

#if defined(__linux__)
    fpu_control_t oldcw, newcw;
    _FPU_GETCW(oldcw);
    newcw = (oldcw & ~_FPU_EXTENDED) | _FPU_DOUBLE;
    _FPU_SETCW(newcw);
    printf(
        "c WARNING: for repeatability, setting FPU to use double precision\n");
#endif

    BoolOption printmodel("Open-WBO", "print-model", "Print model.\n", true);

    IntOption num_tests("Test", "num_tests", "Number of tests\n", 0,
                        IntRange(0, 10000000));

    IntOption test_rhs("Test", "test_rhs",
                       "RHS for a custom encoding test\n", 0,
                       IntRange(0, 10000000));

    IntOption test_rhs2("Test", "test_rhs2",
                        "RHS for a custom encoding test for the second tree\n", 0,
                        IntRange(0, 10000000));

    IntOption test_nsoft("Test", "test_nsoft",
                         "Nsoft for a custom encoding test\n", 0,
                         IntRange(0, 10000000));

    IntOption test_join("Test", "test_join",
                        "Join for a custom encoding test\n", 0, IntRange(0, 1));

    IntOption verbosity("Open-WBO", "verbosity",
                        "Verbosity level (0=minimal, 1=more).\n", 0,
                        IntRange(0, 1));

    IntOption algorithm("Open-WBO", "algorithm",
                        "Search algorithm "
                        "(0=wbo,1=linear-su,2=msu3,3=part-msu3,4=oll,5=best,6="
                        "bmo,7=obv,8=mcs)\n",
                        5, IntRange(0, 8));

    IntOption partition_strategy("PartMSU3", "partition-strategy",
                                 "Partition strategy (0=sequential, "
                                 "1=sequential-sorted, 2=binary)"
                                 "(only for unsat-based partition algorithms).",
                                 2, IntRange(0, 2));

    IntOption graph_type("PartMSU3", "graph-type",
                         "Graph type (0=vig, 1=cvig, 2=res) (only for unsat-"
                         "based partition algorithms).",
                         2, IntRange(0, 2));

    BoolOption bmo("Open-WBO", "bmo", "BMO search.\n", true);

    IntOption cardinality("Encodings", "cardinality",
                          "Cardinality encoding (0=cardinality networks, "
                          "1=totalizer, 2=modulo totalizer).\n",
                          1, IntRange(0, 2));

    IntOption amo("Encodings", "amo", "AMO encoding (0=Ladder).\n", 0,
                  IntRange(0, 0));

    IntOption pb("Encodings", "pb", "PB encoding (0=SWC,1=GTE,2=GTECluster).\n",
                 1, IntRange(0, 2));

    IntOption formula("Open-WBO", "formula",
                      "Type of formula (0=WCNF, 1=OPB).\n", 0, IntRange(0, 1));

    IntOption weight(
        "WBO", "weight-strategy",
        "Weight strategy (0=none, 1=weight-based, 2=diversity-based).\n", 2,
        IntRange(0, 2));

    BoolOption symmetry("WBO", "symmetry", "Symmetry breaking.\n", true);

    IntOption symmetry_lim(
        "WBO", "symmetry-limit",
        "Limit on the number of symmetry breaking clauses.\n", 500000,
        IntRange(0, INT32_MAX));

    IntOption cluster_algorithm("Clustering", "ca",
                                "Clustering algorithm "
                                "(0=none, 1=DivisiveMaxSeparate)",
                                1, IntRange(0, 1));
    IntOption num_clusters("Clustering", "c", "Number of agglomerated clusters",
                           100000, IntRange(1, INT_MAX));

    IntOption rounding_strategy(
        "Clustering", "rs",
        "Statistic used to select"
        " common weights in a cluster (0=Mean, 1=Median, 2=Min)",
        0, IntRange(0, 2));

    IntOption num_conflicts(
      "Incomplete","conflicts","Limit on the number of conflicts.\n", 1000,
      IntRange(0, INT32_MAX));

    IntOption num_iterations(
      "Incomplete","iterations","Limit on the number of iterations.\n", INT32_MAX,
      IntRange(0, INT32_MAX));

	// Changed to the more convinient IntOption
    IntOption local("Incomplete", "local", "Local limit on the number of conflicts.\n", 1);

	IntOption polConservative("TTOpenWboInc", "conservative", "Apply conservative polarity heuristic?\n", 1);
	BoolOption conservativeUseAllVars("TTOpenWboInc", "conservative_use_all_vars", "Re-use the polarity of all the variables within the conservative approach (or, otherwise, only the initial once)?\n", true);	
	IntOption polOptimistic("TTOpenWboInc", "optimistic", "Set target variables' polarity to the optimum?\n", 1);	
	IntOption targetVarsBumpVal("TTOpenWboInc", "target_vars_bump_val", "Bump factor of the activity of the targets at the beginning\n", 113);	
	// Bool
	IntOption targetVarsBumpRelWeights("TTOpenWboInc", "target_vars_bump_rel_weights", "Bump the variable scores, where the bump value is relative to the weights?\n", 1);	
	IntOption targetVarsBumpMaxRandVal("TTOpenWboInc", "target_vars_bump_max_rand_val", "Maximal random bump factor\n", 552);	
	IntOption msMaxEpochs("TTOpenWboInc", "ms_max_epochs", "Mutation sampling: the maximal number of epochs, even if there is an improvement. If there is no improvement or the maximal number of epochs is over, fall back to BMO\n", 10000000);	
	IntOption msMutationClasses("TTOpenWboInc", "ms_mutation_classes", "Mutation sampling: the number of mutation classes\n", 1);
	IntOption msConflictsPerSatCall("TTOpenWboInc", "ms_conflicts_per_sat_call", "Mutation sampling: the maximal number of conflicts per SAT call when testing a mutation\n", 1000);
	IntOption msSatCallsPerEpoch("TTOpenWboInc", "ms_sat_calls_per_iteration", "Mutation sampling: the maximal number of SAT calls per epoch\n", 150);
	BoolOption msDisableTorc("TTOpenWboInc", "ms_disable_torc", "Mutation sampling: disable TORC when looking for mutations\n", false);
    BoolOption msStayNearStartingModel("TTOpenWboInc", "ms_stay_near_starting_model", "Mutation sampling: stay near the starting model (rather the best model) when learning the mutations\n", false);
    IntOption msConflictsLookNearMutation("TTOpenWboInc", "ms_conflicts_look_near_mutation", "Mutation sampling: the maximal number of conflicts per SAT call when looking near a mutation\n", 1000);
    IntOption msSortLitsStrat("TTOpenWboInc", "ms_sort_lits_strategy", "Mutation sampling: 0: don't sort mutation literals; 1: shuffle mutation literals randomly, 2: sort mutation literals -- frequent first, 3: sort mutation literals -- frequent last \n", 1, IntRange(0, 3));
    BoolOption msSiftRemaining("TTOpenWboInc", "ms_sift_remaining", "Mutation sampling: sift the remaining mutation-candidate literals?\n", true);
    BoolOption msSiftRemainingOnlyWhenOptimized("TTOpenWboInc", "ms_sift_remaining_only_when_optimized", "Mutation sampling: sift the remaining mutation-candidate literals only when an optimizing model found?\n", false);
    BoolOption msAfterRepair("TTOpenWboInc", "ms_after_repair", "Mutation sampling: apply after repair?\n", true);
    DoubleOption msModelPerSecThr("TTOpenWboInc", "ms_model_per_sec_thr", "Stop mutation-optimization forever when the improving models-per-second threshold is too low. If non-0, ms_sat_calls_per_iteration is not taken into account for any iteration where ms_model_per_sec_thr is considered", 1.0, DoubleRange(0, true, HUGE_VAL, true));
    IntOption msModelPerSecOnlyForFirstInv("TTOpenWboInc", "ms_model_per_sec_only_for_first_inv", "Mutation sampling: apply model-per-second threshold only for the first mutation-optimization invocation?\n", 0, IntRange(0, 1));
    IntOption msMutCombUseBestModel("TTOpenWboInc", "ms_mut_comb_best_model", "Mutation sampling: apply mutation combination to the best model so far (or, otherwise, the best model)?\n", 0, IntRange(0, 1));
    IntOption msInitBadStrat("TTOpenWboInc", "ms_init_bad_strat", "Mutation sampling: how to initialize the bad observables at the beginning of each epoch: 0 : re-use remaining bad's from the previous epoch; 1: initialize with observables which hold in the best model so far; 2: initialize with observables which hold in the initial model\n", 2, IntRange(0, 2));
    DoubleOption msRndFreqForMublox("TTOpenWboInc", "ms_rnd_freq", "Mutation sampling: the frequency with which the decision heuristic tries to choose a random variable during MuBlox execution (corresponds to the rnd-freq parameter in Glucose", 0, DoubleRange(0, true, 1, true));
    IntOption msMinLBDFrozenClauseForMublox("TTOpenWboInc", "ms_min_lbd_frozen_clause", "Mutation sampling: the value of Glucose's parameter minLBDFrozenClause for the MuBlox phazes of the algorithm\n", 30);
    IntOption msConflToChrono("TTOpenWboInc", "ms_confl_to_chrono", "OBV-BS/Clustering: the value of Glucose's parameter confl-to-chrono\n", 0);
    IntOption msChrono("TTOpenWboInc", "ms_chrono", "OBV-BS/Clustering: the value of Glucose's parameter chrono\n", -1);
    IntOption poContLocalImpr("TTOpenWboInc", "po_cont_local_impr", "Polosat: continue Polosat's first epoch as long as there is a local improvement w.r.t the first model, even if it's not an improvement w.r.t globally best model. Relevant only if the adaptive strategy is on\n", 0);
    IntOption mrsBeaverSizeSwitchToComplete("TTOpenWboInc", "mrs_beaver_size_switch_to_complete", "Mrs. Beaver: the maximal size of the encoding to switch to complete part\n", 1000000);
    IntOption mrsBeaverEachIterStartBestModel("TTOpenWboInc", "mrs_beaver_each_iter_start_best_model", "Mrs. Beaver: make sure each iterations starts with the best (rather than the original) model\n", 0);
    IntOption blockBestModel("TTOpenWboInc", "block_best_model", "Block each best model\n", 0);
    IntOption polosatTurnOffHighLevelConfThrForIters("TTOpenWboInc", "ps_turn_off_high_lvl_confthr_for_iters", "Polosat: turn off higher-level coflict-threshold for iterations?\n", 0);    
    DoubleOption optimisticMaxSoftFraction("TTOpenWboInc", "optimistic_max_soft_percent", "The maximal fraction of soft clauses to leave the optimistic polarity selection enabled (if optimistic is on)\n", 1., DoubleRange(0., true, 1., true));
    DoubleOption conservativeMaxSoftFraction("TTOpenWboInc", "conservative_max_soft_percent", "The maximal fraction of soft clauses to leave the conservative polarity selection enabled (if conservative is on)\n", 1., DoubleRange(0., true, 1., true));
    IntOption mrsBeaverFailWinSizeToSkipOutputs("TTOpenWboInc", "mrs_beaver_fail_win_size_to_skip_outputs", "Mrs. Beaver: the window size of failing obv-bs/ums-obv-bs iterations to start randomly skipping mrs_beaver_fail_skip_percent\% of the outputs\n", 0);
    IntOption mrsBeaverFailSkipPercent("TTOpenWboInc", "mrs_beaver_fail_skip_percent", "Mrs. Beaver: the initial percent of outputs to skip, if mrs_beaver_fail_win_size_to_skip_outputs subsequent Mrs. Beaver iterations failed\n", 10);
    IntOption weightedObvBsFirstIterStrat("TTOpenWboInc", "weighted_obvbs_firstiter_strat", "weighted: 0: don't use OBV-BS; 1: use OBV-BS immediately after the first SAT invocation (replace MS, if MS is on)\n", 0, IntRange(0, 1));
    IntOption wobsAdaptiveStoppedStopMs("TTOpenWboInc", "wobs_adaptive_stopped_stop_ms", "Weighted OBV-BS: 0: if adaptive policy stopped OBV-BS execution, don't stop MS; 1: if adaptive policy stopped OBV-BS execution, stop MS\n", 0, IntRange(0, 1));
    DoubleOption wobsAdaptiveNotStoppedMSThrMult("TTOpenWboInc", "wobs_adaptive_not_stopped_ms_thr_mult", "Weighted OBV-BS: if adaptive policy didn't stop OBV-BS execution, multiply MS's adaptive threshold by this value\n", 1.);
    IntOption msObvStrat("TTOpenWboInc", "ms_obv_strat", "MS and Polosat OBV-BS integration strategy: 0: don't use OBV-BS in MS; 1: fully switch to OBV-BS-like alg. in MS; 2: switch to OBV-BS in MS, but try standard MS call too, if OBV failed to generate a model; 3: switch to OBV-BS in MS, but try standard MS call too, if OBV failed to generate a *better* model\n", 3, IntRange(0, 3));
    IntOption mrsBeaverPolosatRegulateStrat("TTOpenWboInc", "mrs_beaver_polosat_regulate_strat", "Mrs. Beaver: 0: no change to default; 1 (relevant only if ms_obv_strat>0): switch off ms_obv_strat after the initial SAT invocation in Mrs. Beaver; 2 (relevant only if ms_obv_strat>0): switch off ms_obv_strat before the complete part; 3: switch off polosat before the complete part (if not already off)\n", 3, IntRange(0, 3));
    IntOption mrsBeaverApplySizeThrDuringInitialPolosat("TTOpenWboInc", "mrs_beaver_apply_size_thr_during_initial_polosat", "Mrs. Beaver: apply the size threshold *during* the initial polosat\n", 1);
    IntOption printEveryModel("TTOpenWboInc", "print_every_model", "Print every new improving model explicitly\n", 0);
    IntOption msVerbosity("TTOpenWboInc", "ms_verbosity", "Mutation sampling: verbosity (0: not verbose at all; only print when interrupted or finished; 1: print o and timeo messages; 2: print some info about the alg. progress; 3: debugging)\n", 0, IntRange(0, 3));
    
    parseOptions(argc, argv, true);

    if ((int)num_tests) {
      if ((int)test_join) {
        for (int i = 0; i < (int)num_tests; i++) {
          test_encoding_join();
        }
      } else {
        for (int i = 0; i < (int)num_tests; i++) {
          test_encoding();
        }
      }

      return 0;
    }

	Torc::Instance()->SetPolConservative((bool)polConservative);
	Torc::Instance()->SetConservativeAllVars(conservativeUseAllVars);
	Torc::Instance()->SetPolOptimistic((bool)polOptimistic);
	Torc::Instance()->SetTargetVarsBumpVal(targetVarsBumpVal);	
	Torc::Instance()->SetBumpRelWeights((bool)targetVarsBumpRelWeights);	
	Torc::Instance()->SetTargetBumpMaxRandVal(targetVarsBumpMaxRandVal);	
	Torc::Instance()->SetMsMutationClasses(msMutationClasses);
	Torc::Instance()->SetMsConflictsPerSatCall(msConflictsPerSatCall);
	Torc::Instance()->SetMsSatCallsPerEpoch(msSatCallsPerEpoch);
	Torc::Instance()->SetMsMaxEpochs(msMaxEpochs);
	Torc::Instance()->SetMsDisableTorc(msDisableTorc);
	Torc::Instance()->SetMsStayNearStartingModel(msStayNearStartingModel);
	Torc::Instance()->SetMsConflictsLookNearMutation(msConflictsLookNearMutation);
	Torc::Instance()->SetMsSortLitsStrat(msSortLitsStrat);
	Torc::Instance()->SetMsSiftRemaining(msSiftRemaining);
	Torc::Instance()->SetMsSiftRemainingOnlyWhenOptimized(msSiftRemainingOnlyWhenOptimized);
	Torc::Instance()->SetMsAfterRepair(msAfterRepair);
	Torc::Instance()->SetMsModelPerSecThr(msModelPerSecThr);
	Torc::Instance()->SetMsModelPerSecOnlyForFirstInv((bool)msModelPerSecOnlyForFirstInv);
	Torc::Instance()->SetMsMutCombUseBestModel((bool)msMutCombUseBestModel);
	Torc::Instance()->SetMsInitBadStrat(msInitBadStrat);
	Torc::Instance()->SetMsRndFreqForMublox(msRndFreqForMublox);
	Torc::Instance()->SetMsMinLBDFrozenClauseForMublox(msMinLBDFrozenClauseForMublox);
	Torc::Instance()->SetConflToChrono(msConflToChrono);
    Torc::Instance()->SetChrono(msChrono);
    Torc::Instance()->SetPoContLocalImpr(poContLocalImpr);
    Torc::Instance()->SetMrsBeaverSizeSwitchToComplete(mrsBeaverSizeSwitchToComplete);
    Torc::Instance()->SetMrsBeaverEachIterStartBestModel(mrsBeaverEachIterStartBestModel);    
    Torc::Instance()->SetBlockBestModel(blockBestModel);    
	Torc::Instance()->SetPolosatTurnOffHighLevelConfThrForIters(polosatTurnOffHighLevelConfThrForIters);
	Torc::Instance()->SetMrsBeaverFailWinSizeToSkipOutputs(mrsBeaverFailWinSizeToSkipOutputs);
	Torc::Instance()->SetMrsBeaverFailSkipPercent(mrsBeaverFailSkipPercent);
	Torc::Instance()->SetWeightedObvBsFirstIterStrat(weightedObvBsFirstIterStrat);
	Torc::Instance()->SetWobsAdaptiveStoppedStopMs(wobsAdaptiveStoppedStopMs);
    Torc::Instance()->SetWobsAdaptiveNotStoppedMSThrMult(wobsAdaptiveNotStoppedMSThrMult);
    Torc::Instance()->SetMsObvStrat(msObvStrat);
    Torc::Instance()->SetMrsBeaverPolosatRegulateStrat(mrsBeaverPolosatRegulateStrat);
    Torc::Instance()->SetMrsBeaverApplySizeThrDuringInitialPolosat(mrsBeaverApplySizeThrDuringInitialPolosat);
    Torc::Instance()->SetPrintEveryModel(printEveryModel);
	Torc::Instance()->SetMsVerbosity(msVerbosity);
    double initial_time = cpuTime();
    MaxSAT *S = NULL;

    Statistics rounding_statistic =
        static_cast<Statistics>((int)rounding_strategy);

    switch ((int)algorithm) {
    case _ALGORITHM_WBO_:
      S = new WBO(verbosity, weight, symmetry, symmetry_lim);
      break;

    case _ALGORITHM_LINEAR_SU_:
      if ((int)(cluster_algorithm) == 1) {
        S = new LinearSUMod(verbosity, bmo, cardinality, pb,
                            ClusterAlg::_DIVISIVE_, rounding_statistic,
                            (int)(num_clusters));
      } else {
        S = new LinearSU(verbosity, bmo, cardinality, pb);
      }
      break;

    case _ALGORITHM_PART_MSU3_:
      S = new PartMSU3(verbosity, partition_strategy, graph_type, cardinality);
      break;

    case _ALGORITHM_MSU3_:
      S = new MSU3(verbosity);
      break;

    case _ALGORITHM_LSU_CLUSTER_:
      S = new LinearSUClustering(verbosity, bmo, cardinality, pb,
                                 ClusterAlg::_DIVISIVE_, rounding_statistic,
                                 (int)(num_clusters));
      break;

    case _ALGORITHM_LSU_MRSBEAVER_:
      S = new OBV(verbosity, cardinality, num_conflicts, num_iterations, (bool)local); 
      break;

    case _ALGORITHM_LSU_MCS_:
      // The default num_iterations changed from 100000, so the default MCS call has changed
      S = new BLS(verbosity, cardinality, num_conflicts, num_iterations, (bool)local);
      break;

    case _ALGORITHM_OLL_:
      if ((int)(cluster_algorithm) == 1) {
        S = new OLLMod(verbosity, cardinality, ClusterAlg::_DIVISIVE_,
                       rounding_statistic, (int)(num_clusters));
      } else {
        S = new OLL(verbosity, cardinality);
      }
      break;

    case _ALGORITHM_BEST_:
      break;

    default:
      printf("c Error: Invalid MaxSAT algorithm.\n");
      printf("s UNKNOWN\n");
      exit(_ERROR_);
    }

    signal(SIGXCPU, SIGINT_exit);
    signal(SIGTERM, SIGINT_exit);

    if (argc == 1) {
      printf("c Error: no filename.\n");
      printf("s UNKNOWN\n");
      exit(_ERROR_);
    }

    gzFile in = (argc == 1) ? gzdopen(0, "rb") : gzopen(argv[1], "rb");
    if (in == NULL)
      printf("c ERROR! Could not open file: %s\n",
             argc == 1 ? "<stdin>" : argv[1]),
          printf("s UNKNOWN\n"), exit(_ERROR_);

    MaxSATFormula *maxsat_formula = new MaxSATFormula();

    if ((int)formula == _FORMAT_MAXSAT_) {
      parseMaxSATFormula(in, maxsat_formula);
      maxsat_formula->setFormat(_FORMAT_MAXSAT_);
    } else {
      ParserPB *parser_pb = new ParserPB();
      parser_pb->parsePBFormula(argv[1], maxsat_formula);
      maxsat_formula->setFormat(_FORMAT_PB_);
    }
    gzclose(in);

    if ((int)test_rhs) {
      if ((int)test_rhs2) {
        test_encoding(maxsat_formula, (uint64_t)test_rhs, (uint64_t)test_rhs2,
                      (uint64_t)test_nsoft);
      } else {
        test_encoding(maxsat_formula, (uint64_t)test_rhs);
      }
      return 0;
    }

    printf("c |                                                                "
           "                                       |\n");
    printf("c ========================================[ Problem Statistics "
           "]===========================================\n");
    printf("c |                                                                "
           "                                       |\n");

    if (maxsat_formula->getFormat() == _FORMAT_MAXSAT_)
      printf(
          "c |  Problem Format:  %17s                                         "
          "                          |\n",
          "MaxSAT");
    else
      printf(
          "c |  Problem Format:  %17s                                         "
          "                          |\n",
          "PB");

    if (maxsat_formula->getProblemType() == _UNWEIGHTED_)
      printf("c |  Problem Type:  %19s                                         "
             "                          |\n",
             "Unweighted");
    else
      printf("c |  Problem Type:  %19s                                         "
             "                          |\n",
             "Weighted");

    printf("c |  Number of variables:  %12d                                    "
           "                               |\n",
           maxsat_formula->nVars());
    printf("c |  Number of hard clauses:    %7d                                "
           "                                   |\n",
           maxsat_formula->nHard());
    printf("c |  Number of soft clauses:    %7d                                "
           "                                   |\n",
           maxsat_formula->nSoft());
    printf("c |  Number of cardinality:     %7d                                "
           "                                   |\n",
           maxsat_formula->nCard());
    printf("c |  Number of PB :             %7d                                "
           "                                   |\n",
           maxsat_formula->nPB());
    double parsed_time = cpuTime();

    printf("c |  Parse time:           %12.2f s                                "
           "                                 |\n",
           parsed_time - initial_time);
    printf("c |                                                                "
           "                                       |\n");
	
	const auto clss = maxsat_formula->nSoft() + maxsat_formula->nHard();
	const double softsFract = (double)maxsat_formula->nSoft()/(double)clss;
	
	if (polOptimistic && clss != 0 && optimisticMaxSoftFraction < 1. && softsFract > optimisticMaxSoftFraction) 
	{
		Torc::Instance()->SetPolOptimistic((bool)(polOptimistic = 0));
		printf("c | optimistic polarity turned off, since the softs comprise %f of the clauses > %f |\n", softsFract, (double)optimisticMaxSoftFraction);	
	}
	
	if (polConservative && clss != 0 && conservativeMaxSoftFraction < 1. && softsFract > conservativeMaxSoftFraction) 
	{
		Torc::Instance()->SetPolConservative((bool)(polConservative = 0));
		printf("c | conservative polarity turned off, since the softs comprise %f of the clauses > %f |\n", softsFract, (double)conservativeMaxSoftFraction);	
	}
	
    if (algorithm == _ALGORITHM_BEST_) {
	  assert(S == NULL);
	  
      if (maxsat_formula->getProblemType() == _UNWEIGHTED_) {
        // Unweighted
        /*S = new PartMSU3(_VERBOSITY_MINIMAL_, _PART_BINARY_, RES_GRAPH,
                         cardinality);
        S->loadFormula(maxsat_formula);

        if (((PartMSU3 *)S)->chooseAlgorithm() == _ALGORITHM_MSU3_) {
          // FIXME: possible memory leak
          S = new MSU3(_VERBOSITY_MINIMAL_);
        }*/
        
        if (!targetVarsBumpVal.is_set_by_user())
        {
			Torc::Instance()->SetTargetVarsBumpVal(0);
			printf("c | best-alg-selector overriden target_vars_bump_val to 0, since it hadn't been set by the user |\n");	
		}
		
		if (!msModelPerSecThr.is_set_by_user())
        {
			Torc::Instance()->SetMsModelPerSecThr(2.);
			printf("c | best-alg-selector overriden ms_model_per_sec_thr to 2, since it hadn't been set by the user |\n");	
		}				
		
		if (!msMutationClasses.is_set_by_user())
		{
			Torc::Instance()->SetMsMutationClasses(0);
			printf("c | best-alg-selector overriden ms_mutation_classes to 0, since it hadn't been set by the user |\n");	
		}
		
		if (!msChrono.is_set_by_user())
		{
			Torc::Instance()->SetChrono(100);
			printf("c | best-alg-selector overriden ms_chrono to 100, since it hadn't been set by the user |\n");	
		}
		
        
        // -cardinality=2 -conflicts=10000 -iterations=100 -algorithm=7
		S = new OBV(verbosity, 2, num_conflicts, num_iterations, (bool)local); 
		algorithm = _ALGORITHM_LSU_MRSBEAVER_;  
      } else {
		  
        // Weighted
        // S = new OLL(_VERBOSITY_MINIMAL_, cardinality);
        S = new LinearSUClustering(verbosity, bmo, cardinality, pb,
                                 ClusterAlg::_DIVISIVE_, rounding_statistic,
                                 (int)(num_clusters));   
        algorithm = _ALGORITHM_LSU_CLUSTER_;  
    
      }
    }

    if (S->getMaxSATFormula() == NULL) {
      S->loadFormula(maxsat_formula);
      if ((int)(cluster_algorithm) == 1) {
        switch ((int)algorithm) {
        case _ALGORITHM_LINEAR_SU_:
          static_cast<LinearSUMod *>(S)->initializeCluster();
          break;
        case _ALGORITHM_OLL_:
          static_cast<OLLMod *>(S)->initializeCluster();
          break;
        case _ALGORITHM_LSU_CLUSTER_:
          static_cast<LinearSUClustering *>(S)->initializeCluster();
          break;
        }
      }
    }
    S->setPrintModel(printmodel);
    S->setInitialTime(initial_time);
    mxsolver = S;
    mxsolver->search();

  } catch (OutOfMemoryException &) {
    sleep(1);
    printf("c Error: Out of memory.\n");
    mxsolver->printAnswer(_UNKNOWN_);
    exit(_UNKNOWN_);
    //printf("s UNKNOWN\n");
    //exit(_ERROR_);
  }
}
