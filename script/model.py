# import torch
# import torch.nn as nn

# class CRNN(nn.Module):
#     def __init__(self, num_classes, input_channels=1):
#         super(CRNN, self).__init__()
#         # Convolutional layers
#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2)
#         )

#         # Recurrent layers (LSTM)
#         self.rnn = nn.LSTM(128, 256, bidirectional=True, num_layers=2, dropout=0.5)

#         # Final fully connected layer
#         self.fc = nn.Linear(512, num_classes)  # 512 because of bidirectional LSTM

#     def forward(self, x):
#         # CNN forward
#         conv_output = self.cnn(x)  # (batch_size, channels, height, width)
#         conv_output = conv_output.permute(3, 0, 1, 2)  # Convert to (width, batch_size, channels, height)

#         # Flatten the height
#         batch_size, channels, height, width = conv_output.size()
#         conv_output = conv_output.view(width, batch_size, -1)  # (width, batch_size, height * channels)

#         # RNN forward
#         rnn_output, _ = self.rnn(conv_output)

#         # Fully connected layer
#         output = self.fc(rnn_output)

#         return output




import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(CRNN, self).__init__()
        # Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        # Recurrent layers (LSTM)
        self.rnn = nn.LSTM(128 * 56, 256, bidirectional=True, num_layers=2, dropout=0.5)

        # Final fully connected layer
        self.fc = nn.Linear(512, num_classes)  # 512 because of bidirectional LSTM

    def forward(self, x):
        # CNN forward
        conv_output = self.cnn(x)  # (batch_size, channels, height, width)
        
        # Reshape for RNN input
        batch_size, channels, height, width = conv_output.size()
        conv_output = conv_output.permute(0, 3, 1, 2)  # (batch_size, width, channels, height)
        conv_output = conv_output.contiguous().view(batch_size, width, -1)  # (batch_size, width, channels * height)

        # RNN forward
        rnn_output, _ = self.rnn(conv_output)

        # Fully connected layer
        output = self.fc(rnn_output)

        return output