# model.py
class StyleAwareHandwritingModel(nn.Module):
    def __init__(self, latent_dim=256, style_dim=64):
        super().__init__()
        self.style_dim = style_dim
        
        # Style encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(len(HandwritingAnalyzer().feature_names), 128),
            nn.ReLU(),
            nn.Linear(128, style_dim),
            nn.Tanh()
        )
        
        # Content encoder (similar to previous implementation)
        self.content_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, latent_dim - style_dim)
        )
        
        # Decoder with style conditioning
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 16)),
            StyleAwareConvTranspose(64, 32, style_dim),
            nn.ReLU(),
            StyleAwareConvTranspose(32, 1, style_dim),
            nn.Tanh()
        )
        
    def encode_style(self, style_params):
        return self.style_encoder(style_params)
    
    def encode_content(self, image):
        return self.content_encoder(image)
    
    def forward(self, image, style_params):
        style_encoding = self.encode_style(style_params)
        content_encoding = self.encode_content(image)
        latent = torch.cat([content_encoding, style_encoding], dim=1)
        return self.decoder(latent)

# Custom layer for style-aware generation
class StyleAwareConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 
                                     kernel_size=4, stride=2, padding=1)
        self.style_mod = nn.Linear(style_dim, out_channels * 2)  # Scale and bias
        
    def forward(self, x, style):
        x = self.conv(x)
        style_params = self.style_mod(style).unsqueeze(2).unsqueeze(3)
        scale, bias = style_params.chunk(2, dim=1)
        return x * scale + bias