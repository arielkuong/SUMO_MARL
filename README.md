# SUMO_MARL
MARL for TSC with different learning methods and network structures, using SUMO time-variant random traffic flows


**None-temporal-based methods training examples**

Independent DQN using MLP (no cooperation, baseline):  
`python3 train_dqn_mlp.py --grid-n 3 --episodes 200 --device cuda`

Independent DQN using GNN:  
`python3 train_dqn_gnn.py --grid-n 3 --episodes 200 --device cuda`

Centralized Training, Decentralized Execution(CTDE) using MLP:  
`python3 train_dqn_ctde_vdn_mlp.py --grid-n 3 --episodes 200 --device cuda`

Centralized Training, Decentralized Execution(CTDE) using GNN:  
`python3 train_dqn_ctde_vdn_gnn.py --grid-n 3 --episodes 200 --device cuda`


**Temporal-based methods training examples**

DQN using LSTM:  
`python3 train_drqn_lstm.py --grid-n 3 --batch-size-seq 16 --seq-len 8 --burn-in 4 --episodes 200 --device cuda`

DQN using GNN+LSTM:  
`python3 train_drqn_gnn_lstm.py --grid-n 3 --batch-size-seq 16 --seq-len 8 --burn-in 4 --episodes 200 --device cuda`

Centralized Training, Decentralized Execution(CTDE) using LSTM:  
`python3 train_drqn_ctde_vdn_lstm.py --grid-n 3 --batch-size-seq 16 --seq-len 8 --burn-in 4 --episodes 200 --device cuda`

Centralized Training, Decentralized Execution(CTDE) using GNN+LSTM:  
`python3 train_drqn_ctde_vdn_gnn_lstm.py --grid-n 3 --batch-size-seq 16 --seq-len 8 --burn-in 4 --episodes 200 --device cuda`


