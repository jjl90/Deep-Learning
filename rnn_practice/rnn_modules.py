import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCell(nn.Module):
  """Implementation of GRU cell from https://arxiv.org/pdf/1406.1078.pdf."""

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    # Learnable weights and bias for `update gate`
    self.W_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_z = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_z', None)

    # Learnable weights and bias for `reset gate`
    self.W_r = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_r = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_r', None)

    # Learnable weights and bias for `output gate`
    self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b', None)

    self.reset_parameters()

  def forward(self, x, prev_state):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state

    concat_hx = torch.cat((prev_h, x), dim=1)
    z = torch.sigmoid(F.linear(concat_hx, self.W_z, self.b_z))
    r = torch.sigmoid(F.linear(concat_hx, self.W_r, self.b_r))
    h_tilde = torch.tanh(
        F.linear(torch.cat((r * prev_h, x), dim=1), self.W, self.b))
    next_h = (1 - z) * prev_h + z * h_tilde
    return next_h

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return


class LSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    # Learnable weights and bias for `forget gate`
    self.W_fg = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_fg_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_fg_bi', None)
    # Learnable weights and bias for `input gate`
    self.W_ig= nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_ig_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ig_bi', None)    
    # Learnable weights and bias for `other gate`
    self.W_og = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_og_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_og_bi', None) 
    # Learnable weights and bias for `output layer`
    self.W_ol = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_ol_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ol_bi', None) 

    self.reset_parameters()
    self.count_parameters()

    return

  def forward(self, x, prev_state, c_state=None):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state
    if c_state is None:
      batch = x.shape[0]
      c_st = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      c_st = c_state

    concat_hx = torch.cat((prev_h, x), dim=1)
    ft = torch.sigmoid(F.linear(concat_hx, self.W_fg, self.W_fg_bi))
    it = torch.sigmoid(F.linear(concat_hx, self.W_ig ,self.W_ig_bi))
    ct_t = torch.sigmoid(F.linear(concat_hx, self.W_og ,self.W_og_bi))
    ct = ft * c_st + it * ct_t
    ot = torch.sigmoid(F.linear(concat_hx, self.W_ol, self.W_ol_bi))
    ht = ot * torch.tanh(ct)
    return ht

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return


class PeepholedLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

   # Learnable weights and bias for `forget gate`
    self.W_fg = nn.Parameter(torch.Tensor(hidden_size, hidden_size*2 + input_size))
    if bias:
      self.W_fg_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_fg_bi', None)
    # Learnable weights and bias for `input gate`
    self.W_ig= nn.Parameter(torch.Tensor(hidden_size, hidden_size*2 + input_size))
    if bias:
      self.W_ig_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ig_bi', None)    
    # Learnable weights and bias for `other gate`
    self.W_og = nn.Parameter(torch.Tensor(hidden_size, hidden_size*2 + input_size))
    if bias:
      self.W_og_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_og_bi', None) 
    # Learnable weights and bias for `output layer`
    self.W_ol = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
    if bias:
      self.W_ol_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ol_bi', None) 

    self.reset_parameters()
    self.count_parameters()
    return

  def forward(self, x, prev_state, c_state=None):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state
    if c_state is None:
      batch = x.shape[0]
      c_st = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      c_st = c_state

    concat_hx = torch.cat((prev_h, x), dim=1)
    concat_hxc = torch.cat((concat_hx, c_st), dim=1)
    ft = torch.sigmoid(F.linear(concat_hxc, self.W_fg, self.W_fg_bi))
    it = torch.sigmoid(F.linear(concat_hxc, self.W_ig ,self.W_ig_bi))
    ct_t = torch.sigmoid(F.linear(concat_hxc, self.W_og ,self.W_og_bi))
    ct = ft * prev_h + it * ct_t
    ot = torch.sigmoid(F.linear(ct, self.W_ol, self.W_ol_bi))
    ht = ot * torch.tanh(ct)
    return ht

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return


class CoupledLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

   # Learnable weights and bias for `forget gate`
    self.W_fg = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_fg_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_fg_bi', None)
    # Learnable weights and bias for `input gate`
    self.W_ig= nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_ig_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ig_bi', None)    
    # Learnable weights and bias for `other gate`
    self.W_og = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_og_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_og_bi', None) 
    # Learnable weights and bias for `output layer`
    self.W_ol = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_ol_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ol_bi', None) 

    self.reset_parameters()
    self.count_parameters()

    return

  def forward(self, x, prev_state, c_state=None):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state
    if c_state is None:
      batch = x.shape[0]
      c_st = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      c_st = c_state

    concat_hx = torch.cat((prev_h, x), dim=1)
    ft = torch.sigmoid(F.linear(concat_hx, self.W_fg, self.W_fg_bi))
    it = torch.sigmoid(F.linear(concat_hx, self.W_ig ,self.W_ig_bi))
    ct_t = torch.sigmoid(F.linear(concat_hx, self.W_og ,self.W_og_bi))
    ct = ft * c_st + (1 - ft) * ct_t
    ot = torch.sigmoid(F.linear(concat_hx, self.W_ol, self.W_ol_bi))
    ht = ot * torch.tanh(ct)
    return ht

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return