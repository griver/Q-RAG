from functools import partial
from flax.training import train_state
from optax import adam
import jax
from jax import numpy as jnp
from transformers import AutoConfig, FlaxAutoModel, AutoTokenizer
from .text_env import TextMemory, TextMemoryItem
from transformers import FlaxBertPreTrainedModel
from transformers.models.bert.modeling_flax_bert import FlaxBertModule
import jax
from jax import numpy as jnp
import flax
from flax import nnx
from flax.nnx import bridge
from collections import namedtuple
import optax
import numpy as np
from rl.flax_q_module import TextMaxQNet, TextQNet, TextRandomPolicy, TextQNetPolicy, ActionEmbedTarget


DQNArgs = namedtuple("DQNArgs", ["gamma", "tau", "lr", "max_steps"])
FilteredTextMemory = namedtuple("FilteredTextMemory", ["available_mask", "input_ids", "attention_mask", "embeds"]) 
FilteredTextMemoryItem = namedtuple("FilteredTextMemoryItem", ["input_ids", "attention_mask"]) 


class TrainState(nnx.TrainState):
    other_variables: nnx.State


@partial(jax.jit, static_argnames=("gamma", "tau"))
def train_step(
    q_state: TrainState, 
    v_state: TrainState,
    state_batch: TextMemory, 
    action_batch: TextMemoryItem, 
    next_state_batch: TextMemory, 
    reward_batch: jnp.asarray, 
    mask_batch: jnp.asarray,
    gamma: float,
    tau: float):

  q_net = nnx.merge(q_state.graphdef, q_state.params, q_state.other_variables)
  v_net_target = nnx.merge(v_state.graphdef, v_state.params, v_state.other_variables)

  q_net.train()
  v_net_target.train()

  v_next_target_1, v_next_target_2 = v_net_target(next_state_batch)
  v_next_target = jnp.minimum(v_next_target_1, v_next_target_2)
  next_q_value = reward_batch + mask_batch * gamma * v_next_target

  def loss_fn(m):
    qf_1, qf_2 = m(state_batch, action_batch)  
    qf_loss = 0.5 * optax.l2_loss(qf_1, next_q_value).mean() + 0.5 * optax.l2_loss(qf_2, next_q_value).mean()    
    return qf_loss
  
  loss, grads = nnx.value_and_grad(loss_fn)(q_net)
  _, _, model_stats = nnx.split(q_net, nnx.Param, ...)

  new_q_state = q_state.apply_gradients(grads=grads, other_variables=model_stats)

  v_net_target.update(q_net, tau)
  _, v_params, v_model_stats = nnx.split(v_net_target, nnx.Param, ...)
  new_v_state = v_state.replace(params=v_params, other_variables=v_model_stats)

  return new_q_state, new_v_state, loss


@partial(nnx.jit, static_argnames="deterministic")
def fast_policy(model, state, a_embeds, deterministic):
    return model(state, a_embeds, deterministic)


class FlaxDQN:

    def __init__(self, flax_model: FlaxBertPreTrainedModel, args: DQNArgs):

        self.gamma = args.gamma
        self.tau = args.tau

        critic = TextQNet(flax_model)
        sheduler = optax.cosine_decay_schedule(args.lr, args.max_steps, alpha=args.lr * 1e-2)
        critic_optim = optax.adamw(learning_rate=sheduler, b1=0.9, b2=0.97, weight_decay=0.001, eps=1e-6)

        graph, model_params, model_stats = nnx.split(critic, nnx.Param, ...)
        self.q_state = TrainState.create(graph, params=model_params, other_variables=model_stats, tx=critic_optim)
        v_net_target = TextMaxQNet(flax_model, critic)
        v_graph, v_model_params, v_model_stats = nnx.split(v_net_target, nnx.Param, ...)
        self.v_state = TrainState.create(v_graph, params=v_model_params, other_variables=v_model_stats, tx=optax.identity())

        self.policy = TextQNetPolicy(flax_model, critic)
        self.random_policy = TextRandomPolicy()
        self.action_embed_target = ActionEmbedTarget(flax_model, critic)
        self.action_embed = critic.a_embedder


    def select_action(self, state: TextMemory, a_embeds: jnp.ndarray, evaluate=False, random=False):
        state = FilteredTextMemory(state.available_mask, state.input_ids, state.attention_mask, state.embeds)
        
        if random and not evaluate:
            action = self.random_policy(state)
        else:
            action = fast_policy(self.policy, state, a_embeds, evaluate)
        
        return int(np.asarray(action))
    
    def get_q_model(self):
        return nnx.merge(self.q_state.graphdef, self.q_state.params, self.q_state.other_variables)

    def update(self, 
                state_batch: TextMemory, 
                action_batch: TextMemoryItem, 
                next_state_batch: TextMemory, 
                reward_batch: jnp.asarray, 
                mask_batch: jnp.asarray):
        
        reward_batch = reward_batch.squeeze()
        mask_batch = mask_batch.squeeze()

        state_batch = FilteredTextMemory(state_batch.available_mask, state_batch.input_ids, state_batch.attention_mask, state_batch.embeds)
        next_state_batch = FilteredTextMemory(next_state_batch.available_mask, next_state_batch.input_ids, next_state_batch.attention_mask, next_state_batch.embeds)
        action_batch = FilteredTextMemoryItem(action_batch.input_ids, action_batch.attention_mask)

        self.q_state, self.v_state, qf_loss = train_step(
            self.q_state, 
            self.v_state, 
            state_batch, 
            action_batch, 
            next_state_batch, 
            reward_batch, 
            mask_batch, 
            self.gamma,
            self.tau
        )

        q_net = nnx.merge(self.q_state.graphdef, self.q_state.params, self.q_state.other_variables)
        self.action_embed_target.update(q_net, self.tau)
        nnx.update(self.action_embed, nnx.state(q_net.a_embedder))

        return qf_loss
    
            


