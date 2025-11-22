"""
Temporal Attention Mechanism for HAR
Source: HAR-CNN-LSTM-ATT-pyTorch/GitFYP_experiment/supervised/UCI/Attention/
"""

import torch
import torch.nn as nn


class TemporalAttn(nn.Module):
    """Temporal Attention Layer for sequence data"""
    
    def __init__(self, hidden_size):
        super(TemporalAttn, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            attn_output: (batch_size, hidden_size) - attended context vector
            attn_weights: (batch_size, seq_len) - attention weights
        """
        # Calculate attention scores
        score_first_part = self.fc1(hidden_states)
        h_t = hidden_states[:,-1,:]  # Last hidden state
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = torch.nn.functional.softmax(score, dim=1).unsqueeze(2)
        
        # Apply attention weights
        scored_x = hidden_states * attention_weights
        
        # Combine with last hidden state
        condensed_x = torch.sum(scored_x, dim=1)
        final_x = torch.cat((condensed_x, h_t), 1)
        
        # Final transformation
        attn_output = torch.nn.functional.relu(self.fc2(final_x))
        
        return attn_output, attention_weights
