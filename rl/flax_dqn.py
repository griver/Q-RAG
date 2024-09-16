from flax.training import train_state
from optax import adam
import jax
from jax import numpy as jnp
from transformers import AutoConfig, FlaxAutoModel, AutoTokenizer
from .text_env import TextMemory, TextMemoryItem
from flax import linen as nn


class Embed(nn.Module):
    bert: nn.Module

    @nn.compact
    def __call__(self, input_ids, attention_mask, dropout_rng, train=True):
        out = self.bert(input_ids, attention_mask, dropout_rng=dropout_rng, train=train).last_hidden_state
        mask = attention_mask[:, :, jnp.newaxis]
        embeds = (out * mask).sum(1) / mask.sum(1)
        return embeds


class Qfn(nn.Module):
    state_embed: Embed
    action_embed: Embed

    @nn.compact
    def __call__(self, s, a, dropout_rng, train=True):
        s_embed = self.state_embed(s.input_ids, s.attention_mask, dropout_rng, train)
        a_embed = self.action_embed(a.input_ids, a.attention_mask, dropout_rng, train)
        
        logits = (s_embed * a_embed) 

        return logits


class FlaxDQN:

    def __init__(self,
        embed_model: FlaxAutoModel):

        key = jax.random.seed(0)
        key, init_key = jax.random.split(key, 2)
        q_model = Qfn(Embed(embed_model), Embed(embed_model))

        self.state = train_state.TrainState.create(
            apply_fn=q_model.apply,
            params={"state_embed": {"bert": embed_model.params}, "action_embed": {"bert": embed_model.params}},
            tx=adam(learning_rate=1e-4)
        )

    
            


