/*!
 * \author Ruben Martins - ruben@sat.inesc-id.pt
 *
 * @section LICENSE
 *
 * Open-WBO, Copyright (c) 2013-2017, Ruben Martins, Vasco Manquinho, Ines Lynce
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

 */

#include <signal.h>
#include <limits>
#include "MaxSAT.h"
#include "Encoder.h"
#include "algorithms/Alg_nuwls.h"

namespace
{
  clauselit **nuwls_clause_lit;
  int *nuwls_clause_lit_count;
  int nuwls_nvars;
  int nuwls_nclauses;
  uint64_t nuwls_topclauseweight;
  long long *nuwls_clause_weight;
  bool using_nuwls = false;
}

using namespace openwbo;

/************************************************************************************************
 //
 // Public methods
 //
 ************************************************************************************************/

void MaxSAT::search()
{
  printf("Error: Invalid MaxSAT algoritm.\n");
  exit(_ERROR_);
}

void MaxSAT::setInitialTime(double initial)
{
  initialTime = initial;
} // Sets the initial time.

/************************************************************************************************
 //
 // SAT solver interface
 //
 ************************************************************************************************/

// Creates an empty SAT Solver.
Solver *MaxSAT::newSATSolver()
{

#ifdef SIMP
  NSPACE::SimpSolver *S = new NSPACE::SimpSolver();
#else
  Solver *S = new Solver();
#endif

  return (Solver *)S;
}

// Creates a new variable in the SAT solver.
void MaxSAT::newSATVariable(Solver *S)
{

  static int count = 0;

#ifdef SIMP
  ((NSPACE::SimpSolver *)S)->newVar();
#else
  count++;
  //  printf("Created new variable, count is : %d\n",count);
  S->newVar();
#endif
}

bool MaxSAT::isRealModel(vec<lbool> &currentModel)
{
  assert(currentModel.size() != 0);

  for (int i = 0; i < maxsat_formula->nHard(); i++)
  {
    bool unsatisfied = true;
    for (int j = 0; j < maxsat_formula->getHardClause(i).clause.size(); j++)
    {

      assert(var(maxsat_formula->getHardClause(i).clause[j]) <
             currentModel.size());
      if ((sign(maxsat_formula->getHardClause(i).clause[j]) &&
           currentModel[var(maxsat_formula->getHardClause(i).clause[j])] ==
               l_False) ||
          (!sign(maxsat_formula->getHardClause(i).clause[j]) &&
           currentModel[var(maxsat_formula->getHardClause(i).clause[j])] ==
               l_True))
      {
        unsatisfied = false;
        break;
      }
    }

    if (unsatisfied)
    {
      return false;
    }
  }
  return true;
}

/*
build clause structure used in nuwls
*/

void MaxSAT::build_nuwls_clause_structure()
{

  nuwls_nvars = maxsat_formula->nVars() - maxsat_formula->nSoft();
  nuwls_nclauses = maxsat_formula->nHard() + maxsat_formula->nSoft();
  nuwls_topclauseweight = maxsat_formula->getHardWeight();

  int nuwls_num_hclauses = maxsat_formula->nHard();
  nuwls_clause_lit = new clauselit *[nuwls_nclauses + 10];
  nuwls_clause_lit_count = new int[nuwls_nclauses + 10];
  nuwls_clause_weight = new long long[nuwls_nclauses + 10];

  int *redunt_test = new int[nuwls_nvars + 10];
  memset(redunt_test, 0, sizeof(int) * (nuwls_nvars + 10));

  int tem_v, tem_sense, tem_lit_count;
  bool clause_reduent;
  int c = 0;
  for (int i = 0; i < maxsat_formula->nHard(); ++i)
  {
    nuwls_clause_lit_count[c] = maxsat_formula->getHardClause(i).clause.size();
    nuwls_clause_lit[c] = new clauselit[nuwls_clause_lit_count[c] + 1];
    clause_reduent = false;
    tem_lit_count = 0;
    for (int j = 0; j < nuwls_clause_lit_count[c]; ++j)
    {
      tem_v = var(maxsat_formula->getHardClause(i).clause[j]) + 1;
      tem_sense = 1 - sign(maxsat_formula->getHardClause(i).clause[j]);
      if (redunt_test[tem_v] == 0)
      {
        redunt_test[tem_v] = tem_sense - 2;

        // nuwls_clause_lit[c][tem_lit_count].clause_num = c;
        nuwls_clause_lit[c][tem_lit_count].var_num = tem_v;
        nuwls_clause_lit[c][tem_lit_count].sense = tem_sense;

        tem_lit_count++;
      }
      else if (redunt_test[tem_v] == tem_sense - 2)
      {
        continue;
      }
      else
      {
        clause_reduent = true;
        break;
      }
    }
    for (int j = 0; j < nuwls_clause_lit_count[c]; ++j)
      redunt_test[var(maxsat_formula->getHardClause(i).clause[j]) + 1] = 0;
    if (clause_reduent == false)
    {
      nuwls_clause_weight[c] = nuwls_topclauseweight;
      nuwls_clause_lit[c][tem_lit_count].var_num = 0;
      // nuwls_clause_lit[c][tem_lit_count].clause_num = -1;
      nuwls_clause_lit_count[c] = tem_lit_count;
      c++;
    }
    else
    {
      delete nuwls_clause_lit[c];
    }
  }
  for (int i = nuwls_num_hclauses; i < nuwls_nclauses; ++i)
  {
    nuwls_clause_lit_count[c] = maxsat_formula->getSoftClause(i - nuwls_num_hclauses).clause.size();
    nuwls_clause_lit[c] = new clauselit[nuwls_clause_lit_count[c] + 1];
    clause_reduent = false;
    tem_lit_count = 0;
    for (int j = 0; j < nuwls_clause_lit_count[c]; ++j)
    {
      tem_v = var(maxsat_formula->getSoftClause(i - nuwls_num_hclauses).clause[j]) + 1;
      tem_sense = 1 - sign(maxsat_formula->getSoftClause(i - nuwls_num_hclauses).clause[j]);
      if (redunt_test[tem_v] == 0)
      {
        redunt_test[tem_v] = tem_sense - 2;

        // nuwls_clause_lit[c][tem_lit_count].clause_num = c;
        nuwls_clause_lit[c][tem_lit_count].var_num = tem_v;
        nuwls_clause_lit[c][tem_lit_count].sense = tem_sense;

        tem_lit_count++;
      }
      else if (redunt_test[tem_v] == tem_sense - 2)
      {
        continue;
      }
      else
      {
        clause_reduent = true;
        break;
      }
    }
    for (int j = 0; j < nuwls_clause_lit_count[c]; ++j)
      redunt_test[var(maxsat_formula->getSoftClause(i - nuwls_num_hclauses).clause[j]) + 1] = 0;
    if (clause_reduent == false)
    {
      nuwls_clause_weight[c] = maxsat_formula->getSoftClause(i - nuwls_num_hclauses).weight;
      nuwls_clause_lit[c][tem_lit_count].var_num = 0;
      // nuwls_clause_lit[c][tem_lit_count].clause_num = -1;
      nuwls_clause_lit_count[c] = tem_lit_count;
      c++;
    }
    else
    {
      delete nuwls_clause_lit[c];
    }
  }
  nuwls_nclauses = c;
}

lbool MaxSAT::polosat(Solver *solver, vec<Lit> &assumptions, vec<Lit> &obsVecLit, std::function<bool()> StopAfterNewImprovingModel)
{
  using namespace std;

  if (Torc::Instance()->GetMsMaxEpochs() == 0)
  {
    return searchSATSolver(solver, assumptions);
  }

  std::vector<Lit> observables;
  observables.reserve(obsVecLit.size());

  for (int i = 0; i < obsVecLit.size(); i++)
    observables.push_back(obsVecLit[i]);

  const int verbosity = Torc::Instance()->GetMsVerbosity();
  const bool contLocalImpr = Torc::Instance()->GetPoContLocalImpr() != 0;

  const auto prevCost = model.size() == 0 ? (uint64_t)-1 : computeCostModel(model);

  auto res = searchSATSolver(solver, assumptions);

  if (res == l_True && using_nuwls == false)
  {
    auto currCost = computeCostModel(solver->model);
    saveModel(solver->model, currCost);
    cout << "o " << currCost << endl;
    using_nuwls = true;
    NUWLS nuwls_solver;
    build_nuwls_clause_structure();

    nuwls_solver.build_instance(nuwls_nvars, nuwls_nclauses, nuwls_topclauseweight,
                                nuwls_clause_lit, nuwls_clause_lit_count, nuwls_clause_weight);

    cout << "c build NuWLS instance done!" << endl;
    cout << "c changing to NuWLS solver!!!" << endl;
    nuwls_solver.settings();

    vector<int> init_solu(nuwls_nvars + 1);
    for (int i = 0; i < nuwls_nvars; ++i)
    {
      if (solver->model[i] == l_False)
        init_solu[i + 1] = 0;
      else
        init_solu[i + 1] = 1;
    }

    nuwls_solver.init(init_solu);

    start_timing();
    int time_limit_for_ls = NUWLS_TIME_LIMIT;
    if (nuwls_solver.if_using_neighbor)
    {
      for (int step = 1; step < nuwls_solver.max_flips; ++step)
      {
        if (nuwls_solver.hard_unsat_nb == 0)
        {
          nuwls_solver.local_soln_feasible = 1;
          if (nuwls_solver.soft_unsat_weight < nuwls_solver.opt_unsat_weight)
          {
            nuwls_solver.max_flips = step + nuwls_solver.max_non_improve_flip;
            time_limit_for_ls = get_runtime() + NUWLS_TIME_LIMIT;

            nuwls_solver.best_soln_feasible = 1;
            nuwls_solver.opt_unsat_weight = nuwls_solver.soft_unsat_weight;
            cout << "o " << nuwls_solver.opt_unsat_weight << endl;
            for (int v = 1; v <= nuwls_solver.num_vars; ++v)
            {
              if (nuwls_solver.cur_soln[v] == 0)
                solver->model[v - 1] = l_False;
              else
                solver->model[v - 1] = l_True;
            }
            // modify by ychu
            auto oriCost = computeCostModel(solver->model);
            saveModel(solver->model, oriCost);
            // solver->model.copyTo(best_model);

            if (nuwls_solver.opt_unsat_weight == 0)
              break;
          }
        }
        int flipvar = nuwls_solver.pick_var();
        nuwls_solver.flip2(flipvar);
        nuwls_solver.time_stamp[flipvar] = step;

        if (step % 1000 == 0)
        {
          if (get_runtime() > time_limit_for_ls)
            break;
        }
      }
    }
    else
    {
      for (int step = 1; step < nuwls_solver.max_flips; ++step)
      {
        if (nuwls_solver.hard_unsat_nb == 0)
        {
          nuwls_solver.local_soln_feasible = 1;
          if (nuwls_solver.soft_unsat_weight < nuwls_solver.opt_unsat_weight)
          {
            nuwls_solver.max_flips = step + nuwls_solver.max_non_improve_flip;
            time_limit_for_ls = get_runtime() + NUWLS_TIME_LIMIT;

            nuwls_solver.best_soln_feasible = 1;
            nuwls_solver.opt_unsat_weight = nuwls_solver.soft_unsat_weight;
            cout << "o " << nuwls_solver.opt_unsat_weight << endl;
            for (int v = 1; v <= nuwls_solver.num_vars; ++v)
            {
              if (nuwls_solver.cur_soln[v] == 0)
                solver->model[v - 1] = l_False;
              else
                solver->model[v - 1] = l_True;
            }
            //modify by ychu
            auto oriCost = computeCostModel(solver->model);
            saveModel(solver->model, oriCost);
            // solver->model.copyTo(best_model);

            if (nuwls_solver.opt_unsat_weight == 0)
              break;
          }
        }
        int flipvar = nuwls_solver.pick_var();
        nuwls_solver.flip(flipvar);
        nuwls_solver.time_stamp[flipvar] = step;

        if (step % 1000 == 0)
        {
          if (get_runtime() > time_limit_for_ls)
            break;
        }
      }
    }
    nuwls_solver.free_memory();
    cout << "c nuwls search done!" << endl;
  }

  if (res != l_True)
  {
    return res;
  }

  if (Torc::Instance()->GetPolosatTurnOffHighLevelConfThrForIters())
  {
    solver->budgetOffConflict2();
  }

  const bool isConservativeOn = Torc::Instance()->GetPolConservative();
  if (!isConservativeOn)
  {
    Torc::Instance()->SetPolConservative(true);
  }

  auto currOriginalCost = computeCostModel(solver->model);
  if (currOriginalCost < prevCost)
  {
    saveModel(solver->model, currOriginalCost, Torc::Instance()->GetBlockBestModel() ? solver : NULL);
    if (verbosity > 0 && !Torc::Instance()->GetPrintEveryModel())
      printf("o %" PRId64 " \n", currOriginalCost);
    if (verbosity > 0)
      printf("c timeo %u %" PRId64 " \n", (unsigned)ceil(Torc::Instance()->WallTimePassed()), currOriginalCost);

    cout << "o " << currOriginalCost << endl;
  }

  std::chrono::high_resolution_clock::time_point poloStart = std::chrono::high_resolution_clock::now();
  unsigned improvingModelsSoFar = 1;
  const bool isModelPerSecThrApplied = Torc::Instance()->GetMsModelPerSecThr() != 0.;

  auto StopDueToModelPerSecThr = [&]()
  {
    return isModelPerSecThrApplied &&
           improvingModelsSoFar / std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - poloStart).count() < Torc::Instance()->GetMsModelPerSecThr();
  };

  if (verbosity > 2)
    printf("c Entering polosat\n");

  vec<lbool> startingModel;
  model.copyTo(startingModel);
  uint64_t best_cost = computeCostModel(startingModel);
  const int msObvStrat = Torc::Instance()->GetMsObvStrat();
  const int initAssumpsNum = assumptions.size();

  bool goodEpoch = true;
  int litsNumToSwitch = 0;

  for (int iEpoch = 0; iEpoch < Torc::Instance()->GetMsMaxEpochs() && goodEpoch && !StopDueToModelPerSecThr() && (StopAfterNewImprovingModel == nullptr || !StopAfterNewImprovingModel()); ++iEpoch)
  {
    goodEpoch = false;
    int satInvocationsPerEpoch = 0;
    // The vector of optimizing mutations in this epoch
    vector<vector<bool>> optMutations;

    vector<Lit> remainingRelLits;
    remainingRelLits.reserve(maxsat_formula->nSoft());

    for (int i = 0; i < observables.size(); ++i)
    {
      auto relLit = observables[i];
      // Assert the relaxation literal is positive
      assert(sign(relLit) == false);
      // With TORC, the soft clause is falsified iff the relaxation literal is true
      bool isSoftClsFalsified = startingModel[var(relLit)] == l_True;
      if (isSoftClsFalsified)
      {
        remainingRelLits.push_back(relLit);
      }
    }

    if (verbosity > 2 || (verbosity > 1 && iEpoch > 0))
      printf("c Polosat: epoch %d < %d starting with %u remaining literals\n", iEpoch, Torc::Instance()->GetMsMaxEpochs(), remainingRelLits.size());

    int statSat = 0;
    int statUnsat = 0;
    int statConfThr = 0;

    unsigned moUncheckedObsInd = observables.size() - 1;

    while (!remainingRelLits.empty() && !StopDueToModelPerSecThr())
    {
      Lit relLit = remainingRelLits.back();
      remainingRelLits.pop_back();
      // Assert the relaxation literal is positive and its clause is falsified
      assert((sign(relLit) == false && startingModel[var(relLit)] == l_True));

      solver->setConfBudget(Torc::Instance()->GetMsConflictsPerSatCall());
      if (verbosity > 2)
        printf("c Calling SAT for %d time with %u relaxation variables remaining and %d assumptions\n", satInvocationsPerEpoch, remainingRelLits.size(), assumptions.size());

      if (msObvStrat > 0)
      {
        while (observables[moUncheckedObsInd] != relLit)
        {
          const Lit &obvLit = observables[moUncheckedObsInd];
          // The literal must be positive
          assert(sign(obvLit) == false);
          assumptions.push(model[var(obvLit)] == l_True ? obvLit : ~obvLit);
          assert(moUncheckedObsInd != 0);
          --moUncheckedObsInd;
        }

        if (litsNumToSwitch > 0)
        {
          auto assumpsRange = assumptions.size() - initAssumpsNum;
          int litsSwitched = 0;
          while (assumpsRange > 0 && litsSwitched < litsNumToSwitch)
          {
            const auto randInd = rand() % assumpsRange;
            const auto switchedAssump = ~assumptions[initAssumpsNum + randInd];
            assumptions[initAssumpsNum + randInd] = assumptions[assumptions.size() - 1];
            assumptions[assumptions.size() - 1] = switchedAssump;
            --assumpsRange;
            ++litsSwitched;
          }
        }
      }

      assumptions.push(~relLit);
      auto currRes = searchSATSolver(solver, assumptions);
      assumptions.pop();

      ++satInvocationsPerEpoch;

      if ((msObvStrat == 2 && currRes != l_True) ||
          (msObvStrat == 3 && (currRes != l_True || computeCostModel(solver->model) >= best_cost)))
      {
        solver->setConfBudget(Torc::Instance()->GetMsConflictsPerSatCall());
        vec<Lit> savedAssumps;
        assumptions.copyTo(savedAssumps);
        assumptions.resize(initAssumpsNum);

        assumptions.push(~relLit);
        if (verbosity > 2)
          printf("c OBV strategy calling SAT for %d time with %u relaxation variables remaining and %d assumptions\n", satInvocationsPerEpoch, remainingRelLits.size(), assumptions.size());
        currRes = searchSATSolver(solver, assumptions);

        savedAssumps.moveTo(assumptions);

        ++satInvocationsPerEpoch;
      }

      if (currRes == l_True)
      {
        ++statSat;
        uint64_t currOriginalCost = computeCostModel(solver->model);
        if (verbosity > 2)
          printf("c SAT; cost % " PRId64 "\n", currOriginalCost);

        auto Sift = [&](vector<Lit> &toSift)
        {
          toSift.erase(std::remove_if(toSift.begin(),
                                      toSift.end(),
                                      [&](Lit &l)
                                      { return solver->model[var(l)] != l_True; }),
                       toSift.end());
        };

        if (currOriginalCost < best_cost)
        {

          best_cost = currOriginalCost;
          if (currOriginalCost < prevCost)
          {
            saveModel(solver->model, currOriginalCost, Torc::Instance()->GetBlockBestModel() ? solver : NULL);

            if (verbosity > 0 && !Torc::Instance()->GetPrintEveryModel())
              printf("o %" PRId64 " \n", currOriginalCost);
            if (verbosity > 0)
              printf("c timeo %u %" PRId64 " \n", (unsigned)ceil(Torc::Instance()->WallTimePassed()), currOriginalCost);

            cout << "o " << currOriginalCost << endl;
            if (StopAfterNewImprovingModel != nullptr && StopAfterNewImprovingModel())
            {
              break;
            }

            goodEpoch = true;
            ++improvingModelsSoFar;
            litsNumToSwitch = 0;

            if (Torc::Instance()->GetMsMutationClasses() > 0)
            {
              vector<bool> currMutation(solver->model.size(), false);
              for (int i = 0; i < solver->model.size(); ++i)
              {
                if (solver->model[i] != startingModel[i])
                {
                  currMutation[i] = true;
                }
              }
              optMutations.emplace_back(move(currMutation));
            }
          }
          else if (contLocalImpr)
          {
            ++improvingModelsSoFar;
          }
        }

        Sift(remainingRelLits);
      }
      else if (currRes == l_False)
      {
        ++statUnsat;
        if (verbosity > 2)
          printf("c UNSAT\n");
      }
      else
      {
        ++statConfThr;
        if (verbosity > 2)
          printf("c Conflict threshold reached\n");
      }
    }

    if (msObvStrat > 0)
    {
      assumptions.resize(initAssumpsNum);
    }

    bool mutCombClassImproved = true;
    for (int iMutClass = 0; iMutClass < Torc::Instance()->GetMsMutationClasses() && mutCombClassImproved && !StopDueToModelPerSecThr() && !optMutations.empty() && (StopAfterNewImprovingModel == nullptr || !StopAfterNewImprovingModel()); ++iMutClass)
    {
      if (iMutClass == 0 && verbosity > 1)
        printf("c Created close solutions: (sat,unsat,confthr)=(%d,%d,%d). Moving on to mutation combination\n", statSat, statUnsat, statConfThr);
      if (verbosity > 1)
        printf("c Class %d < %d starting. %d mutations.\n", iMutClass, Torc::Instance()->GetMsMutationClasses(), (int)optMutations.size());

      mutCombClassImproved = false;
      vector<vector<bool>> mutCombs;

      for (int i = 0; i < (int)optMutations.size(); ++i)
      {
        for (int j = i + 1; j < (int)optMutations.size(); ++j)
        {
          // Combine mutations i and j
          vector<bool> combMut(model.size(), false);
          vec<lbool> newModel;
          for (int iVar = 0; iVar < model.size(); ++iVar)
          {
            combMut[iVar] = optMutations[i][iVar] | optMutations[j][iVar];
            auto &mu0 = Torc::Instance()->GetMsMutCombUseBestModel() ? model : startingModel;
            newModel.push(combMut[iVar] ? (mu0[iVar] == l_True ? l_False : l_True) : mu0[iVar]);
          }

          uint64_t newModelOriginalCost = computeCostModel(newModel);

          if (newModelOriginalCost < best_cost)
          {
            mutCombs.emplace_back(move(combMut));

            bool betterModelFound = false;
            if (Torc::Instance()->GetMsConflictsLookNearMutation() > 0)
            {
              vec<lbool> saved_user_phase_saving;
              solver->_user_phase_saving.copyTo(saved_user_phase_saving);

              solver->_user_phase_saving.clear();
              for (int i = 0; i < newModel.size(); i++)
              {
                solver->_user_phase_saving.push(newModel[i]);
              }

              solver->setConfBudget(Torc::Instance()->GetMsConflictsLookNearMutation());

              auto currRes = searchSATSolver(solver);

              if (currRes == l_True)
              {
                uint64_t currResCost = computeCostModel(solver->model);
                if (currResCost < best_cost)
                {
                  solver->model.copyTo(newModel);
                  newModelOriginalCost = computeCostModel(newModel);
                  betterModelFound = true;
                }
              }

              if (!betterModelFound)
              {
                solver->_user_phase_saving.clear();
                saved_user_phase_saving.moveTo(solver->_user_phase_saving);
              }
            }
            else
            {
              betterModelFound = isRealModel(newModel);
            }

            if (betterModelFound)
            {
              ++improvingModelsSoFar;
              saveModel(newModel, newModelOriginalCost);
              best_cost = newModelOriginalCost;
              if (verbosity > 1)
                printf("c Real model (j,j)=(%d,%d) out of %d\n", i, j, (int)optMutations.size());
              if (verbosity > 0 && !Torc::Instance()->GetPrintEveryModel())
                printf("o %" PRId64 " \n", newModelOriginalCost);
              if (verbosity > 0)
                printf("c timeo %u %" PRId64 " \n", (unsigned)ceil(Torc::Instance()->WallTimePassed()), newModelOriginalCost);

              cout << "o " << newModelOriginalCost << endl;
              if (StopAfterNewImprovingModel != nullptr && StopAfterNewImprovingModel())
              {
                break;
              }
              goodEpoch = true;
              mutCombClassImproved = true;
            }
          }
        }
      }

      optMutations = move(mutCombs);
      mutCombs.clear();
    }

    if (goodEpoch)
    {
      startingModel.clear();
      model.copyTo(startingModel);
    }
  }

  if (StopAfterNewImprovingModel != nullptr && StopAfterNewImprovingModel())
  {
    if (verbosity > 1)
      printf("c Stopped Polosat per user request\n");
  }
  else if (StopDueToModelPerSecThr())
  {
    if (verbosity > 1)
      printf("c Stopping Polosat forever due to low model-per-second threshold. Models = %u; Time passed = %f; Model-per-second = %f < user-threshold = %f\n",
             improvingModelsSoFar, std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - poloStart).count(), improvingModelsSoFar / std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - poloStart).count(), Torc::Instance()->GetMsModelPerSecThr());
    Torc::Instance()->SetMsMaxEpochs(0);
  }

  solver->budgetOffConflict();

  if (!isConservativeOn)
  {
    Torc::Instance()->SetPolConservative(false);
    solver->_user_phase_saving.clear();
  }

  if (verbosity > 2)
    printf("c Polosat done\n");

  return l_True;
}

// Solve the formula that is currently loaded in the SAT solver with a set of
// assumptions and with the option to use preprocessing for 'simp'.
lbool MaxSAT::searchSATSolver(Solver *S, vec<Lit> &assumptions, bool pre)
{
  const int verbosity = Torc::Instance()->GetMsVerbosity();
  if (save_model_calls_last_polarity_update != save_model_calls)
  {
    if (Torc::Instance()->GetPolConservative() && model.size() > 0 && !Torc::Instance()->SkipFillingSolverPolarity())
    {
      S->_user_phase_saving.clear();
      model.copyTo(S->_user_phase_saving);
      /*
      for (int i = 0; i < model.size(); i++){
        S->_user_phase_saving.push(model[i]);
      }*/
    }
    save_model_calls_last_polarity_update = save_model_calls;
  }

  // Currently preprocessing is disabled by default.
  // Variable elimination cannot be done on relaxation variables nor on variables
  // that belong to soft clauses. To preprocessing to be used those variables
  // should be frozen.

  if (verbosity > 3)
  {
    printf("c Calling SAT with the following %d assumptions:", assumptions.size());
    for (int i = 0; i < assumptions.size(); ++i)
      printf(" %d", assumptions[i]);
    printf("\n");
  }

#ifdef SIMP
  lbool res = ((NSPACE::SimpSolver *)S)->solveLimited(assumptions, pre);
#else
  lbool res = S->solveLimited(assumptions);
#endif

  return res;
}

// Solve the formula without assumptions.
lbool MaxSAT::searchSATSolver(Solver *S, bool pre)
{
  vec<Lit> dummy; // Empty set of assumptions.
  return searchSATSolver(S, dummy, pre);
}

/************************************************************************************************
 //
 // Utils for model management
 //
 ************************************************************************************************/

/*_________________________________________________________________________________________________
  |
  |  saveModel : (currentModel : vec<lbool>&)  ->  [void]
  |
  |  Description:
  |
  |    Saves the current model found by the SAT solver and generates an output string if optionalWeightForPrintOuts!=-1
  |
  |  Pre-conditions:
  |    * Assumes that 'nbInitialVariables' has been initialized.
  |    * Assumes that 'currentModel' is not empty.
  |
  |  Post-conditions:
  |    * 'model' is updated to the current model.
  |
  |________________________________________________________________________________________________@*/
void MaxSAT::saveModel(vec<lbool> &currentModel, uint64_t optionalWeightForPrintOuts, Solver *sToBlockModel)
{
  ++save_model_calls;
  assert(maxsat_formula->nInitialVars() != 0);
  assert(currentModel.size() != 0);

  // Block the SIGTERM signal
  sigset_t newMask, oldMask;
  sigaddset(&newMask, SIGTERM);
  sigprocmask(SIG_BLOCK, &newMask, &oldMask);

  oInLatestModel = optionalWeightForPrintOuts;
  model.clear();
  for (int i = 0; i < maxsat_formula->nInitialVars(); i++)
  {
    model.push(currentModel[i]);
  }

  if (Torc::Instance()->GetPrintEveryModel())
  {
    printModel();
  }

  // Restore the old mask, including unblocking SIGTERM
  sigprocmask(SIG_SETMASK, &oldMask, NULL);

  if (Torc::Instance()->GetConservativeAllVars())
  {
    // Store the value of all the variables
    for (int i = maxsat_formula->nInitialVars(); i < currentModel.size(); i++)
    {
      model.push(currentModel[i]);
    }
  }

  if (sToBlockModel != NULL)
  {
    vec<Lit> blocking;

    for (int i = 0; i < maxsat_formula->nSoft(); i++)
    {
      auto l = maxsat_formula->getSoftClause(i).relaxation_vars[0];
      if (model[var(l)] == l_False)
      {
        blocking.push(l);
      }
    }

    sToBlockModel->addClause(blocking);
  }
}

/*_________________________________________________________________________________________________
  |
  |  computeCostModel : (currentModel : vec<lbool>&) (weight : int) ->
  |                     [uint64_t]
  |
  |  Description:
  |
  |    Computes the cost of 'currentModel'. The cost of a model is the sum of
  |    the weights of the unsatisfied soft clauses.
  |    If a weight is specified, then it only considers the sum of the weights
  |    of the unsatisfied soft clauses with the specified weight.
  |
  |  Pre-conditions:
  |    * Assumes that 'currentModel' is not empty.
  |
  |________________________________________________________________________________________________@*/
uint64_t MaxSAT::computeCostModel(vec<lbool> &currentModel, uint64_t weight)
{

  assert(currentModel.size() != 0);
  uint64_t currentCost = 0;

  // printf("c computeCostModel : ");

  for (int i = 0; i < maxsat_formula->nSoft(); i++)
  {
    bool unsatisfied = true;
    // printf("%d ", maxsat_formula->getSoftClause(i).relaxation_vars[0]);
    for (int j = 0; j < maxsat_formula->getSoftClause(i).clause.size(); j++)
    {

      if (weight != UINT64_MAX &&
          maxsat_formula->getSoftClause(i).weight != weight)
      {
        unsatisfied = false;
        continue;
      }

      assert(var(maxsat_formula->getSoftClause(i).clause[j]) <
             currentModel.size());
      if ((sign(maxsat_formula->getSoftClause(i).clause[j]) &&
           currentModel[var(maxsat_formula->getSoftClause(i).clause[j])] ==
               l_False) ||
          (!sign(maxsat_formula->getSoftClause(i).clause[j]) &&
           currentModel[var(maxsat_formula->getSoftClause(i).clause[j])] ==
               l_True))
      {
        unsatisfied = false;
        break;
      }
    }

    if (unsatisfied)
    {
      currentCost += maxsat_formula->getSoftClause(i).weight;
    }
  }

  // printf("\n");

  return currentCost;
}

/*_________________________________________________________________________________________________
  |
  |  isBMO : (cache : bool)  ->  [void]
  |
  |  Description:
  |
  |    Tests if the MaxSAT formula has lexicographical optimization criterion.
  |
  |  For further details see:
  |    * Joao Marques-Silva, Josep Argelich, Ana Graça, Inês Lynce: Boolean
  |      lexicographic optimization: algorithms & applications. Ann. Math.
  |      Artif. Intell. 62(3-4): 317-343 (2011)
  |
  |  Post-conditions:
  |    * 'orderWeights' is updated with the weights in lexicographical order if
  |      'cache' is true.
  |
  |________________________________________________________________________________________________@*/
bool MaxSAT::isBMO(bool cache)
{
  assert(orderWeights.size() == 0);
  bool bmo = true;
  std::set<uint64_t> partitionWeights;
  std::map<uint64_t, uint64_t> nbPartitionWeights;

  for (int i = 0; i < maxsat_formula->nSoft(); i++)
  {
    partitionWeights.insert(maxsat_formula->getSoftClause(i).weight);
    nbPartitionWeights[maxsat_formula->getSoftClause(i).weight]++;
  }

  for (auto iter = partitionWeights.rbegin(); iter != partitionWeights.rend();
       ++iter)
  {
    orderWeights.push_back(*iter);
  }

  uint64_t totalWeights = 0;
  for (int i = 0; i < (int)orderWeights.size(); i++)
    totalWeights += orderWeights[i] * nbPartitionWeights[orderWeights[i]];

  for (int i = 0; i < (int)orderWeights.size(); i++)
  {
    totalWeights -= orderWeights[i] * nbPartitionWeights[orderWeights[i]];
    if (orderWeights[i] < totalWeights)
    {
      bmo = false;
      break;
    }
  }

  if (!cache)
    orderWeights.clear();

  return bmo;
}

/************************************************************************************************
 //
 // Utils for printing
 //
 ************************************************************************************************/

// Prints information regarding the AMO encoding.
void MaxSAT::print_AMO_configuration(int encoding)
{
  switch (encoding)
  {
  case _AMO_LADDER_:
    printf("c |  AMO Encoding:         %12s                      "
           "                                             |\n",
           "Ladder");
    break;

  default:
    printf("c Error: Invalid AMO encoding.\n");
    printf("s UNKNOWN\n");
    break;
  }
}

// Prints information regarding the PB encoding.
void MaxSAT::print_PB_configuration(int encoding)
{
  switch (encoding)
  {
  case _PB_SWC_:
    printf("c |  PB Encoding:         %13s                        "
           "                                           |\n",
           "SWC");
    break;

  case _PB_GTE_:
    printf("c |  PB Encoding:         %13s                        "
           "                                           |\n",
           "GTE");
    break;

  case _PB_GTECLUSTER_:
    printf("c |  PB Encoding:         %13s                        "
           "                                           |\n",
           "GTECLUSTER");
    break;

  default:
    printf("c Error: Invalid PB encoding.\n");
    printf("s UNKNOWN\n");
    break;
  }
}

// Prints information regarding the cardinality encoding.
void MaxSAT::print_Card_configuration(int encoding)
{
  switch (encoding)
  {
  case _CARD_CNETWORKS_:
    printf("c |  Cardinality Encoding: %12s                                "
           "                                   |\n",
           "CNetworks");
    break;

  case _CARD_TOTALIZER_:
    printf("c |  Cardinality Encoding: %12s                                "
           "                                   |\n",
           "Totalizer");
    break;

  case _CARD_MTOTALIZER_:
    printf("c |  Cardinality Encoding:    %19s                             "
           "                            |\n",
           "Modulo Totalizer");
    break;

  default:
    printf("c Error: Invalid cardinality encoding.\n");
    printf("s UNKNOWN\n");
    break;
  }
}

void MaxSAT::blockModel(Solver *solver)
{
  assert(model.size() != 0);

  vec<Lit> blocking;

  printf("v ");
  for (int i = 0; i < model.size(); i++)
  {
    indexMap::const_iterator iter = maxsat_formula->getIndexToName().find(i);
    if (iter != maxsat_formula->getIndexToName().end())
    {
      if (model[i] == l_False)
        printf("-");
      printf("%s ", iter->second.c_str());
    }
  }
  printf("\n");

  for (int i = 0; i < model.size(); i++)
  {
    blocking.push((model[i] == l_True) ? ~mkLit(i) : mkLit(i));
  }

  solver->addClause(blocking);
}

// Prints the best satisfying model. Assumes that 'model' is not empty.
void MaxSAT::printModel()
{
  static std::string mdl(maxsat_formula->nInitialVars(), '0');
  static bool statusPrinted = false;
  if (!statusPrinted && Torc::Instance()->GetPrintEveryModel())
  {
    printf("s SATISFIABLE\n");
  }
  statusPrinted = true;

  if (oInLatestModel != (uint64_t)-1)
  {
    printf("o %" PRId64 " \n", oInLatestModel);
  }

  assert(model.size() != 0);

  if (maxsat_formula->getFormat() == _FORMAT_PB_)
  {

    printf("v ");
    for (int i = 0; i < model.size(); i++)
    {
      indexMap::const_iterator iter = maxsat_formula->getIndexToName().find(i);
      if (iter != maxsat_formula->getIndexToName().end())
      {
        if (model[i] == l_False)
          printf("-");
        printf("%s ", iter->second.c_str());
      }
    }
    printf("\n");
  }
  else
  {

    // It's sufficient to print out the model for the initial variables only
    for (int i = 0; i < maxsat_formula->nInitialVars(); i++)
    {
      mdl[i] = model[i] == l_True ? '1' : '0';
    }
    printf("v %s\n", mdl.c_str());
  }
  fflush(stdout);
}

// Prints search statistics.
void MaxSAT::printStats()
{
  double totalTime = cpuTime();
  float avgCoreSize = 0;
  if (nbCores != 0)
    avgCoreSize = (float)sumSizeCores / nbCores;

  printf("c\n");
  if (model.size() == 0)
    printf("c  Best solution:          %12s\n", "-");
  else
    printf("c  Best solution:          %12" PRIu64 "\n", ubCost);
  printf("c  Total time:             %12.2f s\n", totalTime - initialTime);
  printf("c  Nb SAT calls:           %12d\n", nbSatisfiable);
  printf("c  Nb UNSAT calls:         %12d\n", nbCores);
  printf("c  Average core size:      %12.2f\n", avgCoreSize);
  printf("c  Nb symmetry clauses:    %12d\n", nbSymmetryClauses);
  printf("c\n");
}

// Prints the corresponding answer.
void MaxSAT::printAnswer(int type)
{
  if (verbosity > 0)
    printStats();

  if (type == _UNKNOWN_ && model.size() > 0)
    type = _SATISFIABLE_;

  switch (type)
  {
  case _SATISFIABLE_:
    if (!Torc::Instance()->GetPrintEveryModel())
      printf("s SATISFIABLE\n");
    if (print_model && !Torc::Instance()->GetPrintEveryModel())
      printModel();
    break;
  case _OPTIMUM_:
    if (!Torc::Instance()->GetPrintEveryModel())
      printf("s OPTIMUM FOUND\n");
    if (print_model && !Torc::Instance()->GetPrintEveryModel())
      printModel();
    break;
  case _UNSATISFIABLE_:
    printf("s UNSATISFIABLE\n");
    break;
  case _UNKNOWN_:
    printf("s UNKNOWN\n");
    break;
  default:
    printf("c Error: Invalid answer type.\n");
  }
}

void MaxSAT::printFormulaStats(Solver *S)
{
  // printf("c nVars: %d\n", S->nVars());
  // printf("c nClauses: %d\n", S->nClauses());
}

uint64_t MaxSAT::getUB()
{
  // only works for partial MaxSAT currently
  Solver *solver = newSATSolver();

  vec<Lit> relaxation_vars;
  for (int i = 0; i < maxsat_formula->nSoft(); i++)
  {
    Lit p = mkLit(maxsat_formula->nVars() + i, false);
    relaxation_vars.push(p);
  }

  for (int i = 0; i < maxsat_formula->nVars() + maxsat_formula->nSoft(); i++)
    newSATVariable(solver);

  for (int i = 0; i < maxsat_formula->nHard(); i++)
    solver->addClause(maxsat_formula->getHardClause(i).clause);

  vec<Lit> clause;
  for (int i = 0; i < maxsat_formula->nSoft(); i++)
  {
    clause.clear();
    maxsat_formula->getSoftClause(i).clause.copyTo(clause);

    for (int j = 0; j < maxsat_formula->getSoftClause(i).relaxation_vars.size();
         j++)
      clause.push(maxsat_formula->getSoftClause(i).relaxation_vars[j]);

    clause.push(relaxation_vars[i]);

    solver->addClause(clause);
  }

  int limit = 1000;
  solver->setConfBudget(limit);

  vec<Lit> dummy;
  lbool res = searchSATSolver(solver, dummy);
  if (res == l_True)
  {
    uint64_t ub = computeCostModel(solver->model);
    return ub;
  }
  else if (res == l_False)
  {
    printAnswer(_UNSATISFIABLE_);
    exit(_UNSATISFIABLE_);
  }

  return maxsat_formula->nSoft();
}

std::pair<uint64_t, int> MaxSAT::getLB()
{
  // only works for partial MaxSAT currently
  Solver *solver = newSATSolver();

  vec<Lit> relaxation_vars;
  for (int i = 0; i < maxsat_formula->nSoft(); i++)
  {
    Lit p = mkLit(maxsat_formula->nVars() + i, false);
    relaxation_vars.push(p);
  }

  for (int i = 0; i < maxsat_formula->nVars() + maxsat_formula->nSoft(); i++)
    newSATVariable(solver);

  for (int i = 0; i < maxsat_formula->nHard(); i++)
    solver->addClause(maxsat_formula->getHardClause(i).clause);

  vec<Lit> clause;
  for (int i = 0; i < maxsat_formula->nSoft(); i++)
  {
    clause.clear();
    maxsat_formula->getSoftClause(i).clause.copyTo(clause);

    clause.push(relaxation_vars[i]);

    solver->addClause(clause);
  }

  std::map<Lit, int> core; // Mapping between the assumption literal and
                           // the respective soft clause.

  for (int i = 0; i < maxsat_formula->nSoft(); i++)
    core[relaxation_vars[i]] = i;

  int limit = 1000;
  lbool res = l_False;
  uint64_t lb = 0;

  vec<bool> active;
  active.growTo(relaxation_vars.size(), false);
  vec<Lit> assumptions;
  for (int i = 0; i < relaxation_vars.size(); i++)
  {
    if (!active[i])
    {
      assumptions.push(~relaxation_vars[i]);
    }
  }

  while (res == l_False)
  {
    solver->setConfBudget(limit);
    res = searchSATSolver(solver, assumptions);
    if (res == l_False)
    {

      for (int i = 0; i < solver->conflict.size(); i++)
      {
        Lit p = solver->conflict[i];
        if (core.find(p) != core.end())
        {
          assert(!active[core[p]]);
          active[core[p]] = true;
        }
      }

      assumptions.clear();
      for (int i = 0; i < relaxation_vars.size(); i++)
      {
        if (!active[i])
        {
          assumptions.push(~relaxation_vars[i]);
        }
      }
      lb++;
    }
  }

  int nb_relaxed = 0;
  for (int i = 0; i < relaxation_vars.size(); i++)
  {
    if (active[i])
      nb_relaxed++;
  }

  return std::make_pair(lb, nb_relaxed);
}

void MaxSAT::BumpTargets(const vec<Lit> &objFunction, const vec<uint64_t> &coeffs, Solver *solver)
{
  if (Torc::Instance()->GetBumpRelWeights() == false)
  {
    for (int i = 0; i < objFunction.size(); i++)
    {
      auto v = var(objFunction[i]);
      solver->varBumpActivity(v, (double)Torc::Instance()->GetTargetVarsBumpVal() + (double)Torc::Instance()->GetRandBump());
    }
  }
  else
  {
    double minWeight = std::numeric_limits<double>::max();
    double maxWeight = 0.;

    for (int i = 0; i < coeffs.size(); i++)
    {
      if ((double)coeffs[i] < minWeight)
      {
        minWeight = (double)coeffs[i];
      }

      if ((double)coeffs[i] > maxWeight)
      {
        maxWeight = (double)coeffs[i];
      }
    }

    const double maxBumpVal = (double)Torc::Instance()->GetTargetVarsBumpVal();
    const double weightDomain = maxWeight - minWeight;

    for (int i = 0; i < objFunction.size(); i++)
    {
      auto v = var(objFunction[i]);
      const double currWeight = (double)coeffs[i];

      const double bumpVal = weightDomain == 0 ? maxBumpVal + (double)Torc::Instance()->GetRandBump() : (((currWeight - minWeight) / weightDomain) * maxBumpVal + (double)Torc::Instance()->GetRandBump());

      solver->varBumpActivity(v, bumpVal);
      // printf("Bumped %u of weight %f by %f (minWeight = %f ; maxWeight = %f)\n", v, currWeight, bumpVal, minWeight, maxWeight);
    }
  }
}
