import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distrib
import numpy as np

class DRPG_agent(nn.module):
  def __init__(self,layer_sizes,learning_rate=0.001,critic=True):
    super(DRPG_agent,self).__init__()
    self.memory = []
    self.layer_sizes = layer_sizes
    self.gamma = 1.0
    self.critic = critic
    
    if critic:
      self.critic_network = self.iniitialise_network(layer_sizes,critic_nw=True)
      self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=learning_rate)
      self.actor_network = self.initialise_network(layer_sizes)
      self.actor_optimizer = optim.Adam(self.actor_network.parameters(),lr=learning_rate)
      
      self.values=[]
      self.actions=[]
   
      
      // In Progress
      
      
      
   def get_actions(self,inputs):
    states,sequences = inputs
    sequences = torch.nn.utils.rnn.pad_sequence(sequences,batch_first=True,padding_value=0.0)
    mu,log_std = self.actor_network(states,sequences)
    actions = mu + torch.randn_like(mu) * torch.exp(log_std)
    return actions
  def loss(self,inputs,actions,returns):
    mu,log_std = self.actor_network(inputs)
    log_probability = self.log_probability(actions,mu,log_std)
    loss_actor = torch.mean(returns * log_probability)
    return loss_actor
  def log_probability(self,actions,mu,log_std):
    EPS = 1e-8
    pre_sum = -0.5 * (((actions - mu)/(torch.exp(log_std)+EPS))**2+2*log_std+np.log(2*np.pi))
    return torch.sum(pre_sum,dim=1)
