import json
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import aiofiles
import captum.attr as C
from ..clients.client import Client, Response
from ..latents.latents import ActivatingExample, LatentRecord
from ..logger import logger
from typing import Callable, Optional
from transformers import PreTrainedModel

#for debug purpose
from lxt.utils import pdf_heatmap, clean_tokens
from transformers import AutoTokenizer
path = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(path)


#########################################################
class ExplainerResult(NamedTuple):
    record: LatentRecord
    """Latent record passed through to scorer."""

    explanation: str
    """Generated explanation for latent."""

def show_activation_for_debug(
    act_examples: List[ActivatingExample],latentnumber: int):
    for ex in act_examples:
        max_act = ex.activations.max()
        os.makedirs(f"heatmap/feature_{latentnumber}", exist_ok=True)
        tokens = clean_tokens(tokenizer.convert_ids_to_tokens(ex.tokens))
        pdf_heatmap(tokens, ex.normalized_activations/10, path=f"heatmap/feature_{latentnumber}/feature_{latentnumber}_contribution_{max_act}.pdf")
        print(f"Feature {latentnumber} - Max Activation: {max_act}")
    

def _to_heat(relevance: torch.Tensor, heat_method: str = 'sum', normalize_method: str = 'abs_max') -> torch.Tensor:

    if heat_method == 'sum':
        relevance = relevance.float().sum(-1).detach().cpu().squeeze()
    elif heat_method == 'l2':
        relevance = relevance.float().pow(2).sum(-1).sqrt().detach().cpu().squeeze()
    else:
        raise ValueError(f"Unknown heat method: {heat_method}")
    h = relevance
    if normalize_method == 'minmax':
        h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    elif normalize_method == 'abs_max':
        h = h / (h.abs().max() + 1e-16) # default in attnlrp
    else:
        raise ValueError(f"Unknown normalization method: {normalize_method}")
    return h

@dataclass
class Explainer(ABC):
    """
    Abstract base class for explainers.
    """

    client: Client
    """Client to use for explanation generation. """
    verbose: bool = False
    """Whether to print verbose output."""
    threshold: float = 0.3
    """The activation threshold to select tokens to highlight."""
    temperature: float = 0.0
    """The temperature for explanation generation."""
    generation_kwargs: dict = field(default_factory=dict)
    """Additional keyword arguments for the generation client."""
    model: Optional[PreTrainedModel] = None
    """Optional model to use for explanation."""
    hookpoint_to_sparse_encode: Optional[dict[str, Callable]] = None
    """Optional sparse encoders to use for explanation."""
    apply_attnlrp: bool = True
    """Whether to apply the AttnLRP patch to the model."""

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        if self.apply_attnlrp:
            self.update_examples_with_relevance(record)
        
        
        messages = self._build_prompt(record.train)

        response = await self.client.generate(
            messages, temperature=self.temperature, **self.generation_kwargs
        )
        assert isinstance(response, Response)

        try:
            explanation = self.parse_explanation(response.text)
            if self.verbose:
                logger.info(f"Explanation: {explanation}")
                logger.info(f"Messages: {messages[-1]['content']}")
                logger.info(f"Response: {response}")

            return ExplainerResult(record=record, explanation=explanation)
        except Exception as e:
            logger.error(f"Explanation parsing failed: {repr(e)}")
            return ExplainerResult(
                record=record, explanation="Explanation could not be parsed."
            )
#######################################################################
# attnlrp saliency for one SAE latent:sangyu
#######################################################################

    def update_examples_with_relevance(self, record: LatentRecord) -> None:
        """
        For every split in `record`, replace
          • .activations  → top-1-only tensor
          • .normalized_activations → relevance-based ints in [-10, 10]
        """
        hp   = record.latent.module_name
        idx  = record.latent.latent_index
        
        record.train      = self._apply_relevance(record.train,      hp, idx)
        # if self.verbose:
        #     show_activation_for_debug(record.train, idx) # 이 코드 사용하면 heatmap/ 폴더에 activation pdf 저장됨
        record.examples   = self._apply_relevance(record.examples,   hp, idx)
        record.test       = self._apply_relevance(record.test,       hp, idx)
        record.neighbours = self._apply_relevance(record.neighbours, hp, idx)

# ────────────────────────────────────────────────────────────────────────────
# 3)  내부 helper : 실제로 relevance 계산하고 예시 수정
# ────────────────────────────────────────────────────────────────────────────
    def _apply_relevance(
        self,
        act_examples : List[ActivatingExample],
        hookpoint    : str,
        latentnumber : int,
    ) -> List[ActivatingExample]:
        """
        Return a *new* list of ActivatingExample with:
          • activations           –> top-1 only
          • normalized_activations –> int relevance in [-10, 10]
        """
        if not act_examples:
            return act_examples

        # ------------------------------------------------------------------
        # (1)  prepare batch tensors
        # ------------------------------------------------------------------
        toks, acts, _ = zip(*[(ex.tokens,
                               ex.activations,
                               ex.normalized_activations)
                              for ex in act_examples])

        batch_ids  = torch.stack(toks).to("cuda")           # (B,S)
        batch_acts = torch.stack(acts).to("cuda")           # (B,S)

        model = self.model.to("cuda").eval()
        sae   = self.hookpoint_to_sparse_encode[hookpoint].keywords["sae"]

        # ------------------------------------------------------------------
        # (2)  relevance (Grad × Input)  — micro-batched
        # ------------------------------------------------------------------
        relevance = self._compute_relevance(                 # (B,S)  in [-1,1]
            batch_ids, batch_acts, model,
            hookpoint, latentnumber, sae, micro_bs=4
        ).cpu()

        # ------------------------------------------------------------------
        # (3)  build NEW ActivatingExample list
        # ------------------------------------------------------------------
        
        for i, (ex, orig_act, rel) in enumerate(zip(act_examples, acts, relevance)):
            # 3-a) top-1 activation tensor
            top1 = orig_act.clone()
            top1.zero_()
            max_i = torch.argmax(orig_act)
            top1[max_i] = orig_act[max_i]

            # 3-b) relevance → -10~10 ints
            norm_int = torch.clamp((rel * 10.0).round(), -10, 10).to(torch.int)
            #Sangyu:monkey patch to salience first token #TODO: remove this
            norm_int[0] = 0
            # 직접 원본 객체의 필드 수정
            act_examples[i].activations = top1
            act_examples[i].normalized_activations = norm_int

        # 수정된 원본 리스트 반환
        return act_examples

# ────────────────────────────────────────────────────────────────────────────
# 4)  relevance 계산 함수 (기존 relevance_fn 을 메서드로 뺌)
#     — 내용은 이전과 동일, 이름만 바꿈
# ────────────────────────────────────────────────────────────────────────────
    def _compute_relevance(
        self,
        batch_ids    : torch.Tensor,
        batch_acts   : torch.Tensor,
        model        : nn.Module,
        layername    : str,
        latent_idx   : int,
        sae,
        micro_bs     : int = 4,
    ) -> torch.Tensor:                       # → (B,S) relevance in [-1,1]
        device        = batch_ids.device
        embed_layer   = model.get_input_embeddings()
        relevance_all = []

        b_dec = sae.b_dec.to(model.dtype)
        W_enc = sae.encoder.weight[latent_idx].to(model.dtype)
        b_enc = sae.encoder.bias[latent_idx].to(model.dtype)

        for s in range(0, batch_ids.size(0), micro_bs):
            e      = s + micro_bs
            ids_mb = batch_ids[s:e]
            act_mb = batch_acts[s:e]

            model.zero_grad(set_to_none=True)
            embeds = embed_layer(ids_mb).detach().clone().requires_grad_(True)

            captured = {}
            def _hook(_, __, out):
                captured["out"] = out[0] if isinstance(out, tuple) else out
            handle = model.get_submodule(layername).register_forward_hook(_hook)

            attn_mask = ids_mb.ne(tokenizer.pad_token_id).long() \
                        if tokenizer.pad_token_id is not None else None
            _ = model(inputs_embeds=embeds, attention_mask=attn_mask)
            handle.remove()

            outs = captured["out"]                         # (mb,S,H)
            tok_idx = act_mb.argmax(dim=1)                 # (mb,)
            vecs    = outs[torch.arange(outs.size(0), device=device),
                           tok_idx, :]

            score = F.linear(vecs - b_dec, W_enc.unsqueeze(0), b_enc).sum()
            score.backward()

            # Grad × Input → relevance
            rel_mb = embeds.grad * embeds                  # (mb,S,H)
            relevance_all.append(_to_heat(rel_mb))        # (mb,S)

            del embeds, outs, captured
            torch.cuda.empty_cache()

        return torch.cat(relevance_all, dim=0)             # (B,S)

   


    
    def parse_explanation(self, text: str) -> str:
        try:
            match = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return "Explanation could not be parsed."
        except Exception as e:
            logger.error(f"Explanation parsing regex failed: {repr(e)}")
            raise

    def _highlight(self, str_toks: list[str], activations: list[float]) -> str:
        result = ""
        threshold = max(activations) * self.threshold

        def check(i):
            return activations[i] > threshold

        i = 0
        while i < len(str_toks):
            if check(i):
                result += "<<"

                while i < len(str_toks) and check(i):
                    result += str_toks[i]
                    i += 1
                result += ">>"
            else:
                result += str_toks[i]
                i += 1

        return "".join(result)

    def _join_activations(
        self,
        str_toks: list[str],
        token_activations: list[float],
        normalized_activations: list[float],
    ) -> str:
        acts = ""
        activation_count = 0
        for str_tok, token_activation, normalized_activation in zip(
            str_toks, token_activations, normalized_activations
        ):
            if token_activation > max(token_activations) * self.threshold:
                # TODO: for each example, we only show the first 10 activations
                # decide on the best way to do this
                if activation_count > 10:
                    break
                acts += f'("{str_tok}" : {int(normalized_activation)}), '
                activation_count += 1

        return "Activations: " + acts

    @abstractmethod
    def _build_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        pass


async def explanation_loader(
    record: LatentRecord, explanation_dir: str
) -> ExplainerResult:
    try:
        async with aiofiles.open(f"{explanation_dir}/{record.latent}.txt", "r") as f:
            explanation = json.loads(await f.read())
        return ExplainerResult(record=record, explanation=explanation)
    except FileNotFoundError:
        print(f"No explanation found for {record.latent}")
        return ExplainerResult(record=record, explanation="No explanation found")


async def random_explanation_loader(
    record: LatentRecord, explanation_dir: str
) -> ExplainerResult:
    explanations = [f for f in os.listdir(explanation_dir) if f.endswith(".txt")]
    if str(record.latent) in explanations:
        explanations.remove(str(record.latent))
    random_explanation = random.choice(explanations)
    async with aiofiles.open(f"{explanation_dir}/{random_explanation}", "r") as f:
        explanation = json.loads(await f.read())

    return ExplainerResult(record=record, explanation=explanation)
