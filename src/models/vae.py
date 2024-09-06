"""VAE model definitions."""

import math
from flax import linen as nn
from jax import random
import jax.numpy as jnp

class Encoder(nn.Module):
    c_hid : int
    latents : int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 28x28 => 14x14
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x) 
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 14x14 => 7x7
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 7x7 => 4x4
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector  # 4x4x40 => 640
        # x = nn.Dense(features=self.latents)(x)
        mean_x = nn.Dense(self.latents, name='fc2_mean')(x) # 640 => 20
        logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
        return mean_x, logvar_x
    
class Decoder(nn.Module):
    c_out : int
    c_hid : int
    latents : int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*16*self.c_hid)(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], 4, 4, -1)
        x = nn.ConvTranspose(features=2*self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(2, 2), padding=0)(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_out, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.tanh(x)
        return x
    
class Autoencoder(nn.Module):
    c_hid: int
    latents : int
        
    def setup(self):
        # Alternative to @nn.compact -> explicitly define modules
        # Better for later when we want to access the encoder and decoder explicitly
        self.encoder = Encoder(c_hid=self.c_hid, latents=self.latents)
        self.decoder = Decoder(c_hid=self.c_hid, latents=self.latents, c_out=1)
        
    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
  


# class Encoder(nn.Module):
#   """VAE Encoder."""

#   latents: int

#   @nn.compact
#   def __call__(self, x):
#     x = nn.Dense(500, name='fc1')(x)
#     x = nn.relu(x)
#     mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
#     logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
#     return mean_x, logvar_x


# class Decoder(nn.Module):
#   """VAE Decoder."""

#   @nn.compact
#   def __call__(self, z):
#     z = nn.Dense(500, name='fc1')(z)
#     z = nn.relu(z)
#     z = nn.Dense(784, name='fc2')(z)
#     return z


class VAE(nn.Module):
  """Full VAE model."""
  c_hid: int = 32
  latents: int = 20

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder(c_out=1, c_hid=self.c_hid, latents=self.latents)

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))
  
  def encode(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    return z
  
  def reconstruct(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x

def reparameterize(encoder_out, rng):
  mean, logvar = encoder_out
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std

def model(latents):
  return VAE(latents=latents)
