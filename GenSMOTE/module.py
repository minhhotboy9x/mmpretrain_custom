import torch
import torch.nn as nn


# Encoder and Decoder for shape 224x224
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.embed = nn.Embedding(args['num_class'], args['dim_h'])

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
        self.fc = nn.Linear(args['dim_h'] * 33, args['n_z'])

    def forward(self, x, labels):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)   # Shape: (batch_size, dim_h * 32)
        x = torch.cat((x, self.embed(labels)), dim=1)  # Shape: (batch_size, dim_h * 33)
        x = self.fc(x)  # Shape: (batch_size, n_z)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.embed = nn.Embedding(args['num_class'], args['dim_h'])
        self.fc = nn.Sequential(
            nn.Linear(args['n_z'] + args['dim_h'], args['dim_h'] * 32),
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

    def forward(self, x, labels):
        x = torch.cat((x, self.embed(labels)), dim=1)  # Shape: (batch_size, n_z + dim_h)
        x = self.fc(x)  # Shape: (batch_size, dim_h * 32)
        x = x.view(-1, args['dim_h'] * 32, 1, 1)  # Shape: (batch_size, dim_h * 32, 1, 1)
        x = self.deconv(x)  # Shape: (batch_size, 1, 224, 224)
        return x

class GenSMOTE(nn.Module):
    def __init__(self, args):
        super(GenSMOTE, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def encode(self, x, labels):
        return self.encoder(x, labels)
    
    def decode(self, z, labels):
        return self.decoder(z, labels)

    def forward(self, x, labels):
        z = self.encoder(x, labels)
        x_recon = self.decoder(z, labels)
        return x_recon
    

# Example arguments
args = {
    'num_class': 7,
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
    labels = torch.tensor([1])
    z = encoder(input, labels)
    print(z.shape)  # torch.Size([1, 256])
    output = decoder(z, labels)
    print(output.shape)  # torch.Size([1, 1, 224, 224])
