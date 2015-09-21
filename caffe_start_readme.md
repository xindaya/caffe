# Caffe in bosen VS Multi-GPU Caffe
Show how caffe works in these TWO versions

## Caffe in bosen的启动路线

\# 1. tools/caffe-main.cpp
 - 解析参数
 - caffe_engine::InitPS (Create Table)
 -  caffe_engin::Start (num worker threads)

\# 2. caffe_engine::InitPS
- Init() -> InitPS()
- InitPS() -> InitPSForTrainNet()
- InitPS() -> InitPSForTestNet()

\# 3. caffe_engine::Start
- solver: new Solver by GetSolver
- solver -> Solve()

\# 4. solver -> Solver()
 - Solver::Init -> InitTrainNet() & InitTestNets()
 - Solver::Solve()
     - PreSolve()
     - Register net output tables (net_ -> RegisterNetOutputPSTable)
     - Synchronize (net_ -> SyncWithPS)
     - For each iter,
         - JoinSyncThreads & snapshot & TestAll
         - Get loss by Solver::ForwardBackward(bottom_vec)
         - Display & net_->table()->Inc

\# 5. Solver::ForwardBackward()
- net_->Forward(bottom, &loss)
- for each layer, layer::Backward
- Sync (Thread) (For **Inner Product** &**Convolution**  layer)
    - new thread (Solver::ThreadSyncWithPS or ThreadSyncWithSVB)
    - net_->params() 

\# 6. Solver::ThreadSyncWithPS()
- ComputerUpdateValue (for each **blob(param_id)** in the layer of net)
- param->UpdatePSTable()
- param->SyncWithPSTable(clock+1) -> Blob::set\_cpu\_ps\_data

\# 7. SGDSolver::ComputerUpdateValue(param_id)
- GetLearningRate
- Get momentum & weight_decay
- Regularization: L1 or L2, caffe_axpy for data & diff
- Compute and Update value : caffe\_cpu\_axpby & caffe_copy 

## Multi-GPU Caffe的启动路线
\# 1. tools/caffe.cpp
- get_gpus & set device and mode
- solver: new Solver by caffe::GetSolver
- solver->SetActionFunction
- if GPU size > 1
    - **caffe::P2PSync** sync(solver, NULL, solver->param()) **--> Step 2**
    - sync.run(gpus) **--> Step 2**
- else 
    - solver->Solve()

\# 2. \src\caffe\parallel.cpp
- **caffe::P2PSync** sync(solver, NULL, solver->param())
    - get & set device id
    - set root solver (input parent is NULL)
    - solver\_->add\_callback(this)
- sync.run(gpus)
    - DevicePair::compute(gpus, &pairs)
    - syncs : new vector P2PSync(gpus.size())
    - Build GPU tree: for each pair size, set parent, device id and its children push_back
    - for each sync(not include root solver), P2PSync::StartInternalThread() **--> Step 3**
    - solver_->Solve() for root solver **--> Step 7**
    - for each sync, P2PSync::StopInternalThread()

\# 3. P2PSync::StartInternalThread()
- SetDevice
- check root solver
- set random seed for each device ID
- solver\_->Step(max\_iter - initial\_iter\_) **--> Step 4**

\# 4. Solver::Step(iteration)
For each iteration
- for each callback (Callback是Solver的内部类，包含on_start和on\_gradients\_ready)
    - P2PSync::on_start()
    - compute and smooth loss: net\_- >ForwardBackward(bottom\_vec)
- for each callback, P2PSync::on\_gradients\_ready()
- Solver::ApplyUpdate() **--> Step 5**
- SnapShot()

\# 5. SGDSolver::ApplyUpdate()
- check root solver
- GetLearningRate()
- ClipGradients()
- for each param id in net learnable params
    - Normalize(param_id) : caffe\_scal
    - Regularize(param_id) : L1 or L2, caffe\_axpy
    - ComputeUpdateValue(param_id, rate) : caffe\_cpu\_axpby, copy
- this->net_->Update()  **--> Step 6**

\# 6. Net::Update
For each learnable param, leanrable\_params\_[i]->Update()
- Blob::Update() : caffe\_axpy for data

\# 7. Solver::Solve()
- Restore (if has resume file)
- Step(iteration)
- net_->ForwardPrefilled(&loss) for param\_.display() == 0
