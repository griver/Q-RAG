from transformers import FlaxBertPreTrainedModel
from transformers.models.bert.modeling_flax_bert import FlaxBertModule
import jax
from jax import numpy as jnp
import flax
from flax import nnx
from flax.nnx import bridge
from .text_env import TextMemory, TextMemoryItem


@nnx.jit
def soft_update(target: nnx.Module, source: nnx.State, tau):
    new_params = jax.tree.map(lambda p1, p2: p1 * (1.0 - tau) + p2 * tau, nnx.state(target, nnx.Param), source)
    nnx.update(target, new_params)

@nnx.jit
def hard_update(target: nnx.Module, source: nnx.State):
    new_params = jax.tree.map(lambda p: jnp.copy(p), source)
    nnx.update(target, new_params)


class Embed(nnx.Module):

    def __init__(self, flax_model: FlaxBertPreTrainedModel):
        input_ids, attention_mask = jnp.zeros((1, 100), dtype=jnp.int32), jnp.ones((1, 100), dtype=jnp.int32)
        bert = bridge.ToNNX(FlaxBertModule(flax_model.config), rngs=nnx.Rngs(0, dropout=0)).lazy_init(input_ids, attention_mask, deterministic=False)

        graph_def, state = nnx.split(bert)
        state = state.flat_state()
        for path, val in flax.traverse_util.flatten_dict(flax_model.params).items():
            mapped_path = path[:-1] if path[-1] == 'raw_value' else path
            if mapped_path not in state:
                raise ValueError(f"{mapped_path} doesn't exist in {state.keys()}")
            state[mapped_path].value = jnp.copy(val)
        state = nnx.State.from_flat_path(state)
        
        self.bert = nnx.merge(graph_def, state)
    
    def __call__(self, input_ids, attention_mask, *args, deterministic=False, **kw):
        out = self.bert(input_ids, attention_mask, deterministic=deterministic).last_hidden_state
        mask = attention_mask[:, :, jnp.newaxis]
        embeds = (out * mask).sum(1) / mask.sum(1)
        return embeds / 10.0


class TextQNet(nnx.Module):
    
    def __init__(self, flax_model: FlaxBertPreTrainedModel):
        self.s_embedder = Embed(flax_model)
        self.a_embedder = Embed(flax_model)

    def __call__(self, s: TextMemory, a: TextMemoryItem, deterministic=False):
        s_embed =  self.s_embedder(s.input_ids, s.attention_mask, deterministic)
        a_embed =  self.a_embedder(a.input_ids, a.attention_mask, deterministic)
        
        D = s_embed.shape[-1] // 2
        logits_1 = (s_embed[:, :D] * a_embed[:, :D]).sum(-1) 
        logits_2 = (s_embed[:, D:] * a_embed[:, D:]).sum(-1) 

        return logits_1, logits_2


class ActionEmbedTarget(nnx.Module):
    
    def __init__(self, flax_model: FlaxBertPreTrainedModel, q_net: TextQNet) -> None:
        self.action_embed = Embed(flax_model)
        hard_update(self.action_embed, nnx.state(q_net.a_embedder))

    def update(self, q_net: TextQNet, decay: float = 0.01):
        params = nnx.state(q_net.a_embedder, nnx.Param)
        soft_update(self.action_embed, params, decay)

    def __call__(self, *args, **kw):
        return self.action_embed(*args, **kw)
    

class TextMaxQNet(nnx.Module):

    def __init__(self, flax_model: FlaxBertPreTrainedModel, q_net: TextQNet) -> None:
        super().__init__()
        self.state_embed = Embed(flax_model)
        hard_update(self.state_embed, nnx.state(q_net.s_embedder))

    def update(self, q_net: TextQNet, decay: float = 0.01):
        params = nnx.state(q_net.s_embedder, nnx.Param)
        soft_update(self.state_embed, params, decay)

    def __call__(self, s: TextMemory, deterministic=False):

        s_embed = self.state_embed(s.input_ids, s.attention_mask, deterministic)
        s_embed = s_embed[:, jnp.newaxis, :]
        a_embeds: jnp.ndarray = s.embeds
        
        D = s_embed.shape[-1] // 2
        logits_1 = (s_embed[:, :, :D] * a_embeds[:, :, :D]).sum(-1) 
        logits_2 = (s_embed[:, :, D:] * a_embeds[:, :, D:]).sum(-1) 

        logits_1 = jnp.where(s.available_mask, logits_1, jnp.min(logits_1))
        logits_2 = jnp.where(s.available_mask, logits_2, jnp.min(logits_2))

        return (logits_1.max(axis=-1), 
                logits_2.max(axis=-1))


class TextQNetPolicy(nnx.Module):
    
    def __init__(self, flax_model: FlaxBertPreTrainedModel, q_net: TextQNet) -> None:
        self.state_embed = Embed(flax_model)
        hard_update(self.state_embed, nnx.state(q_net.s_embedder))

    def update(self, q_net: TextQNet):
        hard_update(self.state_embed, nnx.state(q_net.s_embedder))

    def __call__(self, s: TextMemory, a_embeds: jnp.ndarray, deterministic=True):
        
        input_ids = s.input_ids[jnp.newaxis,]
        attention_mask = s.attention_mask[jnp.newaxis,]
        mask = s.available_mask[jnp.newaxis,]
        a_embeds = a_embeds[jnp.newaxis, ]

        s_embed = self.state_embed(input_ids, attention_mask, deterministic)
        s_embed = s_embed[:, jnp.newaxis, :]
        
        logits = (s_embed * a_embeds).sum(-1) / 2
        logits = jnp.where(mask, logits, logits.min() - 1)
       
        return jnp.argmax(logits, -1)
    

class TextRandomPolicy(nnx.Module):

    def __init__(self, rngs=nnx.Rngs(0)):
        self.rngs = rngs

    def __call__(self, s: TextMemory):

        mask = jnp.asarray(s.available_mask)
        
        probs = jnp.ones(mask.shape[0])
        probs = jnp.where(mask, probs, -100)
        action = jax.random.categorical(self.rngs(), logits=probs)
        
        return action

        