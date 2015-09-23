#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include "caffe/net.hpp"

namespace caffe {

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * a client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
 
// -----------------------------modification part -------------------------------  
// Modify Solver Construction Function 
template <typename Dtype>
class Solver {
 public:
  //explicit Solver(const SolverParameter& param, const Solver* root_solver = NULL);
  //explicit Solver(const string& param_file, const Solver* root_solver = NULL);
  explicit Solver(const SolverParameter& param, const Solver* root_solver = NULL, 
      const map<string, vector<int> >* layer_blobs_global_idx_ptr = NULL,
      const int thread_id = 0);
  explicit Solver(const string& param_file, const Solver* root_solver = NULL, 
      const map<string, vector<int> >* layer_blobs_global_idx_ptr = NULL,
      const int thread_id = 0);
// -----------------------------modification part end------------------------------- 
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  void Restore(const char* resume_file);
  virtual ~Solver() {}
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() { return iter_; }
// -----------------------------modification part ------------------------------- 
  void PrintNetOutputs(const string& filename); 
// -----------------------------modification part end-------------------------------  

  // Invoked at specific points during an iteration
  // 内部类
  class Callback {
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

  void CheckSnapshotWritePermissions();

 protected:
  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;
// -----------------------------modification part -------------------------------
// ************************************************************************
// Currently add all these functions in, they has all been realized in inherit class or lower network  
// ************************************************************************
// PreSolve realized in SGDSolver, and called in construction function
  // PreSolve is run before any solving iteration starts, allowing one to
  // put up some scaffold.
  virtual void PreSolve() {}
  virtual void InitSVB();
  // ForwardBackward realized in Net, the backward operation is different
  virtual Dtype ForwardBackward(const vector<Blob<Dtype>* >& bottom);
  // Get the update value for the current iteration.
  virtual void ComputeUpdateValue(const int param_id) = 0;
  virtual void ComputeUpdateValue() = 0;
  virtual void ThreadSyncWithPS(const shared_ptr<Blob<Dtype> >& param,
      const int param_id, const int param_owner, const int clock);
  virtual void ThreadSyncWithSVB(
    const shared_ptr<Blob<Dtype> >& param, const int param_id, 
    const shared_ptr<Layer<Dtype> >& layer, const int layer_id,
    const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void JoinSyncThreads();
// -----------------------------modification part end-------------------------------  
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
// -----------------------------modification part ------------------------------- 
  virtual void SnapshotSolverState(SolverState* state) = 0;
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  //void Restore(const char* resume_file);
  virtual void RestoreSolverState(const SolverState& state) = 0;
// -----------------------------modification part end------------------------------- 
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void DisplayOutputBlobs(const int net_id);
  void PrintOutputBlobs(shared_ptr<Net<Dtype> >& net, const bool trian, 
      std::ofstream& outfile);
// -----------------------------modification part end------------------------------- 

  SolverParameter param_;
  int iter_;
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  vector<Callback*> callbacks_;
// -----------------------------modification part ------------------------------- 
  int display_counter_;
  int test_counter_;
  int clock_counter_;
  int param_table_staleness_;
  // layer/net_name => vector of blobs' global indexes 
  const map<string, vector<int> >* layer_blobs_global_idx_ptr_; 

  vector<std::thread*> sync_threads_;
  int max_local_sv_updates_;
  int max_remote_sv_updates_;

  const int thread_id_;
  int client_id_;
  int num_threads_;
  int num_clients_;

  petuum::HighResolutionTimer total_timer_;
// -----------------------------modification part end------------------------------- 
  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};

/**
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
 */
template <typename Dtype>
class WorkerSolver : public Solver<Dtype> {
 public:
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}

 protected:
  void ApplyUpdate() {}
  void SnapshotSolverState(const string& model_filename) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromBinaryProto(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromHDF5(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  // -----------------------------modification part ------------------------------- 
  // Just Add these virtual functions in bosen to avoid memory allocation failure
  virtual void ComputeUpdateValue();
  virtual void ComputeUpdateValue(const int param_id);
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // -----------------------------modification part end------------------------------- 
};

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
// -----------------------------modification part ------------------------------- 
// Modify SGDSolver construction function
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  //explicit SGDSolver(const SolverParameter& param) : Solver<Dtype>(param) { PreSolve(); }
  //explicit SGDSolver(const string& param_file) : Solver<Dtype>(param_file) { PreSolve(); }
  explicit SGDSolver(const SolverParameter& param, 
      const map<string, vector<int> >* layer_blobs_global_idx_ptr,
      const int thread_id) : Solver<Dtype>(
      param, layer_blobs_global_idx_ptr, thread_id) {}
  explicit SGDSolver(const string& param_file,
      const map<string, vector<int> >* layer_blobs_global_idx_ptr,
      const int thread_id) : Solver<Dtype>(
      param_file, layer_blobs_global_idx_ptr, thread_id) {}
// -----------------------------modification part end------------------------------- 
  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
// -----------------------------modification part -------------------------------  
// Modify PreSolve definition type, use virtual
  //void PreSolve();
  virtual void PreSolve();
// -----------------------------modification part end------------------------------- 
  Dtype GetLearningRate();
// -----------------------------modification part ------------------------------- 
// Add functions in bosen
  virtual void ComputeUpdateValue();
  virtual void ComputeUpdateValue(const int param_id);
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
// -----------------------------modification part end------------------------------- 
  virtual void ApplyUpdate();
  virtual void Normalize(int param_id);
  virtual void Regularize(int param_id);
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  virtual void ClipGradients();
  virtual void SnapshotSolverState(const string& model_filename);
  virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
  virtual void SnapshotSolverStateToHDF5(const string& model_filename);
  virtual void RestoreSolverStateFromHDF5(const string& state_file);
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};
// -----------------------------modification part end------------------------------- 
template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  //explicit NesterovSolver(const SolverParameter& param) : SGDSolver<Dtype>(param) {}
  //explicit NesterovSolver(const string& param_file) : SGDSolver<Dtype>(param_file) {}
  explicit NesterovSolver(const SolverParameter& param, 
      const map<string, vector<int> >* layer_blobs_global_idx_ptr, 
      const int thread_id) : SGDSolver<Dtype>(
      param, layer_blobs_global_idx_ptr, thread_id) {}
  explicit NesterovSolver(const string& param_file,
      const map<string, vector<int> >* layer_blobs_global_idx_ptr, 
      const int thread_id) : SGDSolver<Dtype>(
      param_file, layer_blobs_global_idx_ptr, thread_id) {}
// -----------------------------modification part end------------------------------- 

 protected:
// -----------------------------modification part ------------------------------- 
  virtual void ComputeUpdateValue(const int param_id);
  virtual void ComputeUpdateValue();
// -----------------------------modification part end------------------------------- 
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};
// -----------------------------modification part ------------------------------- 
template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  //explicit AdaGradSolver(const SolverParameter& param) : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  //explicit AdaGradSolver(const string& param_file) : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  explicit AdaGradSolver(const SolverParameter& param, 
      const map<string, vector<int> >* layer_blobs_global_idx_ptr, 
      const int thread_id) 
      : SGDSolver<Dtype>(param, layer_blobs_global_idx_ptr, thread_id) {
    constructor_sanity_check(); 
  }
  explicit AdaGradSolver(const string& param_file,
      const map<string, vector<int> >* layer_blobs_global_idx_ptr, 
      const int thread_id) : SGDSolver<Dtype>(
      param_file, layer_blobs_global_idx_ptr, thread_id) {
    constructor_sanity_check(); 
  }
// -----------------------------modification part end------------------------------- 

 protected:
// -----------------------------modification part ------------------------------- 
  virtual void ComputeUpdateValue(const int param_id);
  virtual void ComputeUpdateValue();
// -----------------------------modification part end------------------------------- 
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};


template <typename Dtype>
class RMSPropSolver : public SGDSolver<Dtype> {
 public:
  explicit RMSPropSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit RMSPropSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with RMSProp.";
    CHECK_GE(this->param_.rms_decay(), 0)
        << "rms_decay should lie between 0 and 1.";
    CHECK_LT(this->param_.rms_decay(), 1)
        << "rms_decay should lie between 0 and 1.";
  }

  DISABLE_COPY_AND_ASSIGN(RMSPropSolver);
};

template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdaDeltaPreSolve(); }
  explicit AdaDeltaSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdaDeltaPreSolve(); }

 protected:
  void AdaDeltaPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
};

/**
 * @brief AdamSolver, an algorithm for first-order gradient-based optimization
 *        of stochastic objective functions, based on adaptive estimates of
 *        lower-order moments. Described in [1].
 *
 * [1] D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization."
 *     arXiv preprint arXiv:1412.6980v8 (2014).
 */
template <typename Dtype>
class AdamSolver : public SGDSolver<Dtype> {
 public:
  explicit AdamSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdamPreSolve();}
  explicit AdamSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdamPreSolve(); }

 protected:
  void AdamPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdamSolver);
};
// -----------------------------modification part ------------------------------- 
// Modify GetSolver construction function, input variable "layer_blobs_global_idx_ptr & thread_id"
template <typename Dtype>
Solver<Dtype>* GetSolver(const SolverParameter& param, 
    const map<string, vector<int> >* layer_blobs_global_idx_ptr,
    const int thread_id) {
  SolverParameter_SolverType type = param.solver_type();

  switch (type) {
  case SolverParameter_SolverType_SGD:
      //return new SGDSolver<Dtype>(param);
      return new SGDSolver<Dtype>(param, layer_blobs_global_idx_ptr, 
          thread_id);
  case SolverParameter_SolverType_NESTEROV:
      //return new NesterovSolver<Dtype>(param);
      return new NesterovSolver<Dtype>(param, layer_blobs_global_idx_ptr,
          thread_id);
  case SolverParameter_SolverType_ADAGRAD:
      //return new AdaGradSolver<Dtype>(param);
      return new AdaGradSolver<Dtype>(param, layer_blobs_global_idx_ptr, 
          thread_id);	  
  case SolverParameter_SolverType_RMSPROP:
      return new RMSPropSolver<Dtype>(param);
  case SolverParameter_SolverType_ADADELTA:
      return new AdaDeltaSolver<Dtype>(param);
  case SolverParameter_SolverType_ADAM:
      return new AdamSolver<Dtype>(param);
  default:
      LOG(FATAL) << "Unknown SolverType: " << type;
  }
  return (Solver<Dtype>*) NULL;
}
// -----------------------------modification part end------------------------------- 
}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
