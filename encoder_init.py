import sys
import torch
from autoencoder.encoder import VariationalEncoder

class EncodeState():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.conv_encoder = VariationalEncoder(self.latent_dim).to(self.device)
            self.conv_encoder.load()
            self.conv_encoder.eval()

            for params in self.conv_encoder.parameters():
                params.requires_grad = False
        except Exception as e:
            import traceback
            print('Encoder could not be initialized.')
            traceback.print_exc()
            print('Actual error:', e)
            sys.exit()
    
    def process(self, observation):
        image_obs = torch.tensor(observation[0], dtype=torch.float).to(self.device)
        image_obs = image_obs.unsqueeze(0)
        image_obs = image_obs.permute(0,3,2,1)
        image_obs = self.conv_encoder(image_obs)
        navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
        image_obs = torch.nan_to_num(image_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        navigation_obs = torch.nan_to_num(navigation_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        observation = torch.cat((image_obs.view(-1), navigation_obs), -1)
        observation = torch.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return observation
