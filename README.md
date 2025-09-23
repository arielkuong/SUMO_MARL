# SUMO_MARL
MARL for TSC with different learning methods and network structures, using SUMO time-variant random traffic flows


**None-temporal-based methods training examples**

Independent DQN uding MLP (no cooperation, baseline):  
`python3 train_idqn_mlp_shared.py --grid-n 3 --episodes 200 --cpu`

Independent DQN using MLP, neighbors' observations are included in the agent's own observation:  
`python3 train_idqn_mlp_shared_nbobs.py --grid-n 3 --episodes 200 --cpu`

Independent DQN using GNN:  
`python3 train_idqn_gnn_shared.py --grid-n 3 --episodes 200 --cpu`

Centralized Training, Decentralized Execution(CTDE) using MLP:  
`python3 train_vdn_ctde_mlp_shared.py --grid-n 3 --episodes 200 --cpu`


**Temporal-based methods training examples**

Independent DQN using LSTM (no cooperation, baseline):  
`python3 train_idqn_lstm_shared.py --grid-n 3 --batch-size 16 --seq-len 8 --burn-in 4 --episodes 200 --cpu`

Independent DQN using LSTM, neighbors' observations are included in the agent's own observation:  
`python3 train_idqn_lstm_shared_nbobs.py --grid-n 3 --batch-size 16 --seq-len 8 --burn-in 4 --episodes 200 --cpu`

Independent DQN using GNN+LSTM:  
`python3 train_idqn_gnn_lstm_shared.py --grid-n 3 --batch-size 16 --seq-len 8 --burn-in 4 --episodes 200 --cpu`

Centralized Training, Decentralized Execution(CTDE) using LSTM strucutre:  
`python3 train_vdn_ctde_lstm_shared.py --grid-n 3 --batch-size 16 --seq-len 8 --burn-in 4 --episodes 200 --cpu`
