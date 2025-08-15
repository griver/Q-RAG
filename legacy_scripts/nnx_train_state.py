from typing import Any, TypeVar, Generic
from typing_extensions import Self
from collections.abc import Callable

import optax
from jax import numpy as jnp
import jax
from flax import core, struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.nnx.graph import GraphDef, GraphState
from flax import nnx

M = TypeVar("M", bound=nnx.Module)


class TS(Generic[M], struct.PyTreeNode):
 
  step: int | jax.Array
  graphdef: GraphDef[M] = struct.field(pytree_node=False)
  params: nnx.Param = struct.field(pytree_node=True)
  other_variables: nnx.State = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, model: M, *, grads, **kwargs):
    
    grads_with_opt = grads
    _, params_with_opt, new_other_variables = nnx.split(model, nnx.Param, ...)

    updates, new_opt_state = self.tx.update(
      grads_with_opt, self.opt_state, params_with_opt
    )
    new_params = optax.apply_updates(params_with_opt, updates)

    return self.replace(
      step=self.step + 1,
      params=new_params,
      other_variables=new_other_variables,
      opt_state=new_opt_state,
      **kwargs,
    )
  
  def get_model(self) -> M:
    return nnx.merge(self.graphdef, self.params, self.other_variables)

  @classmethod
  def create(cls, *, model: M, tx, **kwargs):
    graphdef, params, other_variables = nnx.split(model, nnx.Param, ...)
    opt_state = tx.init(params)

    return cls(
      step=jnp.array(0),
      graphdef=graphdef,
      params=params,
      other_variables=other_variables,
      tx=tx,
      opt_state=opt_state,
      **kwargs,
    )