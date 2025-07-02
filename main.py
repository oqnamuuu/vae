import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------
# モデルのクラス定義
# --------------------------------

class Encoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(28*28 + 16, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, image, label):
        flattened = image.view(image.size(0), -1)
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embed = F.relu(self.label_embedding(label_one_hot))
        concat = torch.cat([flattened, label_embed], dim=1)
        hidden = F.relu(self.fc_hidden(concat))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        self.fc_out = nn.Linear(128, 28*28)

    def forward(self, latent_vector, label):
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embed = F.relu(self.label_embedding(label_one_hot))
        concat = torch.cat([latent_vector, label_embed], dim=1)
        hidden = F.relu(self.fc_hidden(concat))
        out = torch.sigmoid(self.fc_out(hidden))
        return out.view(-1, 1, 28, 28)

class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, label):
        mu, logvar = self.encoder(x, label)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, label)
        return recon_x, mu, logvar

# --------------------------------
# Streamlit アプリ
# --------------------------------
st.title("CVAE Digit Generator")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの準備
model = CVAE(latent_dim=3, num_classes=10).to(device)
model.load_state_dict(torch.load("cvae.pth", map_location=device))
model.eval()

# ユーザー入力
selected_digit = st.selectbox("生成したい数字を選んでください (0〜9)", list(range(10)))
generate_button = st.button("画像を生成する")

if generate_button:
    with torch.no_grad():
        # 潜在変数を標準正規分布からサンプリング
        z = torch.randn(1, 3).to(device)
        label = torch.tensor([selected_digit], dtype=torch.long, device=device)
        generated = model.decoder(z, label)
        generated_img = generated.cpu().squeeze().numpy()
        
        fig, ax = plt.subplots()
        ax.imshow(generated_img, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
