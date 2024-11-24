import torch
import torch.nn as nn


# Encoder and Decoder for shape 224x224
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(args['n_channel'], args['dim_h'], 4, 2, 1, bias=False),  # 224 -> 112
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args['dim_h'], args['dim_h'] * 2, 4, 2, 1, bias=False),  # 112 -> 56
            nn.BatchNorm2d(args['dim_h'] * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args['dim_h'] * 2, args['dim_h'] * 4, 4, 2, 1, bias=False),  # 56 -> 28
            nn.BatchNorm2d(args['dim_h'] * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args['dim_h'] * 4, args['dim_h'] * 8, 4, 2, 1, bias=False),  # 28 -> 14
            nn.BatchNorm2d(args['dim_h'] * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args['dim_h'] * 8, args['dim_h'] * 16, 4, 2, 1, bias=False),  # 14 -> 7
            nn.BatchNorm2d(args['dim_h'] * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args['dim_h'] * 16, args['dim_h'] * 32, 7, 1, 0, bias=False),  # 7 -> 1
            nn.BatchNorm2d(args['dim_h'] * 32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(args['dim_h'] * 32, args['n_z'])

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)   # Shape: (batch_size, dim_h * 32)
        x = self.fc(x)  # Shape: (batch_size, n_z)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args['n_z'], args['dim_h'] * 32),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(args['dim_h'] * 32, args['dim_h'] * 16, 7, 1, 0),  # 1 -> 7
            nn.BatchNorm2d(args['dim_h'] * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(args['dim_h'] * 16, args['dim_h'] * 8, 4, 2, 1),  # 7 -> 14
            nn.BatchNorm2d(args['dim_h'] * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(args['dim_h'] * 8, args['dim_h'] * 4, 4, 2, 1),  # 14 -> 28
            nn.BatchNorm2d(args['dim_h'] * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(args['dim_h'] * 4, args['dim_h'] * 2, 4, 2, 1),  # 28 -> 56
            nn.BatchNorm2d(args['dim_h'] * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(args['dim_h'] * 2, args['dim_h'], 4, 2, 1),  # 56 -> 112
            nn.BatchNorm2d(args['dim_h']),
            nn.ReLU(True),
            nn.ConvTranspose2d(args['dim_h'], args['n_channel'], 4, 2, 1),  # 112 -> 224
            nn.BatchNorm2d(args['n_channel']),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x).view(-1, args['dim_h'] * 32, 1, 1)  # Shape: (batch_size, dim_h * 32, 1, 1)
        x = self.deconv(x)  # Shape: (batch_size, 1, 224, 224)
        return x

class DeepSMOTE(nn.Module):
    def __init__(self, args):
        super(DeepSMOTE, self).__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    

# Example arguments
args = {
    'dim_h': 64,
    'n_channel': 3,
    'n_z': 256
}

if __name__ == '__main__':
    # Initialize models
    encoder = Encoder(args)
    decoder = Decoder(args)
    encoder.eval()
    decoder.eval()
    input = torch.randn(1, 3, 224, 224)
    z = encoder(input)
    print(z.shape)  # torch.Size([1, 256])
    output = decoder(z)
    print(output.shape)  # torch.Size([1, 1, 224, 224])
