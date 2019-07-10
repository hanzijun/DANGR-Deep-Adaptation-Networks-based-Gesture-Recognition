# Deep-Adaptation-Networks-based-Gesture-Recognition
The project can be devided into two parts:
1) Data augment based on conditional GAN, related modules: 

   Earth_move_GAN.py and ops_modify.py
2) Domain adaptation based multi-kernel Maximum Mean Discrepancy, related modules: 

   s_model.py for training deep neural networks in source domain;
   
   t_model.py for transfering source model to targets;
   
   mmd.py for computing domain discrepancy for adaptation.
