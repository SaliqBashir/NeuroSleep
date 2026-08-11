import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out

class DualBranchCNN(nn.Module):
    def __init__(self, in_channels=3, emb_dim=256):
        super(DualBranchCNN, self).__init__()
        
        # Branch 1 (Small Filter - High Frequency)
        # We aim for 128 dimensions from this branch
        self.branch1_conv = nn.Conv1d(in_channels, out_channels=16, kernel_size=50, stride=5, padding=25)
        self.branch1_bn = nn.BatchNorm1d(16)
        self.branch1_res = ResidualBlock1D(16)
        self.branch1_pool = nn.AdaptiveMaxPool1d(8)
        self.branch1_drop = nn.Dropout(p=0.5)
        
        # Branch 2 (Large Filter - Low Frequency)
        # We aim for 128 dimensions from this branch
        self.branch2_conv = nn.Conv1d(in_channels, out_channels=16, kernel_size=400, stride=20, padding=200)
        self.branch2_bn = nn.BatchNorm1d(16)
        self.branch2_res = ResidualBlock1D(16)
        self.branch2_pool = nn.AdaptiveMaxPool1d(8)
        self.branch2_drop = nn.Dropout(p=0.5)
        
        # Output after concat will be 16*8 + 16*8 = 128 + 128 = 256
        
    def forward(self, x):
        # x shape: (B, 3, 3000)
        
        # Branch 1
        x1 = self.branch1_conv(x)
        x1 = self.branch1_bn(x1)
        x1 = F.relu(x1)
        x1 = self.branch1_res(x1)
        x1 = self.branch1_pool(x1)
        x1 = self.branch1_drop(x1)
        x1 = x1.view(x1.size(0), -1) # Flatten -> (B, 128)
        
        # Branch 2
        x2 = self.branch2_conv(x)
        x2 = self.branch2_bn(x2)
        x2 = F.relu(x2)
        x2 = self.branch2_res(x2)
        x2 = self.branch2_pool(x2)
        x2 = self.branch2_drop(x2)
        x2 = x2.view(x2.size(0), -1) # Flatten -> (B, 128)
        
        # Merge
        out = torch.cat([x1, x2], dim=1) # (B, 256)
        return out


class SequenceLSTM(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, num_layers=2, num_classes=5):
        super(SequenceLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        
        # 512 * 2 (bidirectional) = 1024
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: (B, SeqLen, 256)
        lstm_out, _ = self.lstm(x) # (B, SeqLen, 1024)
        
        # Self-Attention (query, key, value)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        out = lstm_out + attn_out
        
        # Project to logits
        logits = self.classifier(out) # (B, SeqLen, 5)
        return logits


class NeuroSleepModel(nn.Module):
    def __init__(self, in_channels=3, cnn_emb_dim=256, lstm_hidden=512, num_classes=5):
        super(NeuroSleepModel, self).__init__()
        
        self.feature_extractor = DualBranchCNN(in_channels=in_channels, emb_dim=cnn_emb_dim)
        self.sequence_model = SequenceLSTM(input_dim=cnn_emb_dim, hidden_dim=lstm_hidden, num_classes=num_classes)
        
    def forward(self, x):
        # x shape: (B, SeqLen, Channels, Time) -> e.g., (32, 20, 3, 3000)
        B, L, C, T = x.shape
        
        # Reshape to process all epochs through the CNN independently
        x_flat = x.view(B * L, C, T) # (B*20, 3, 3000)
        
        features = self.feature_extractor(x_flat) # (B*20, 256)
        
        # Reshape back to sequence
        features = features.view(B, L, -1) # (B, 20, 256)
        
        # Process through LSTM
        logits = self.sequence_model(features) # (B, 20, 5)
        
        return logits
