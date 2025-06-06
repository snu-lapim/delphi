import asyncio
from dataclasses import dataclass
from typing import NamedTuple, List

from ..explainer import ActivatingExample, Explainer, ExplainerResult, show_activation_for_debug
from .prompt_builder import build_prompt
from delphi.latents import ActivatingExample, LatentRecord
import torch
import torch.nn as nn
import torch.nn.functional as F
import aiofiles
import captum.attr as C
from delphi.clients.client import Client, Response
from delphi.logger import logger
from typing import Callable, Optional
from transformers import PreTrainedModel

#for debug purpose
from lxt.utils import pdf_heatmap, clean_tokens
from transformers import AutoTokenizer
path = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(path)



def _to_heat(relevance: torch.Tensor, heat_method: str = 'sum', normalize_method: str = 'abs_max') -> torch.Tensor:

    if heat_method == 'sum':
        relevance = relevance.float().sum(-1).detach().cpu()
    elif heat_method == 'l2':
        relevance = relevance.float().pow(2).sum(-1).sqrt().detach().cpu()
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
class AttnLRPExplainer(Explainer):
    activations: bool = True
    """Whether to show activations to the explainer."""
    cot: bool = False
    """Whether to use chain of thought reasoning."""

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        
        # show_activation_for_debug(record.train, record.latent.latent_index, postfix="org")
        self.update_examples_with_relevance(record)
        # show_activation_for_debug(record.train, record.latent.latent_index, postfix="attnlrp")
        
        messages = self._build_prompt(record.train)

        response = await self.client.generate(
            messages, temperature=self.temperature, **self.generation_kwargs
        )
        # import pdb;pdb.set_trace()
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


    def _build_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        highlighted_examples = []

        for i, example in enumerate(examples):
            str_toks = example.str_tokens
            activations = example.activations.tolist()
            normalized_activations = example.normalized_activations.tolist()
            highlighted_examples.append(f"Example {i+1}: ")
            highlighted_examples.append(self._highlight(str_toks, activations, normalized_activations))

            if self.activations:
                assert (
                    example.normalized_activations is not None
                ), "Normalized activations are required for activations in explainer"
                normalized_activations = example.normalized_activations.tolist()
                highlighted_examples.append(
                    self._join_activations(
                        str_toks, activations, normalized_activations
                    )
                )

        highlighted_examples = "\n".join(highlighted_examples)

        return build_prompt(
            examples=highlighted_examples,
            activations=self.activations,
            cot=self.cot,
        )

    def _highlight(self, str_toks: list[str], activations: list[float], normalized_activations: list[int]) -> str:
        result = ""
        # threshold = max(activations) * self.threshold
        threshold = 0.1  # Sangyu:monkey patch to use fixed threshold for now

        def check(i):
            return normalized_activations[i] > threshold
        def act_check(i):
            return activations[i] > threshold
        i = 0
        while i < len(str_toks):
            
            if check(i):
                result += "<<"

                while i < len(str_toks) and check(i):
                    if act_check(i):
                        result += "{{"
                    result += str_toks[i]
                    i += 1
                if act_check(i-1):
                    result += "}}"
                result += ">>"
            else:
                result += str_toks[i]
                i += 1
        ### Sangyu:actviated focused token : *word*

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
            if token_activation > 0.1: # only one activated token
                acts += f'("Contribution to token {{{{{str_tok}}}}} whose feature activation is {token_activation} : '
        
        acts += '['
        for str_tok, token_activation, normalized_activation in zip(
            str_toks, token_activations, normalized_activations
        ):
            if normalized_activation > 0.1:
                # TODO: for each example, we only show the first 10 activations
                # decide on the best way to do this
                if activation_count > 100:
                    break
                acts += f'("{str_tok}" : {int(normalized_activation)}),'
                activation_count += 1
        acts = acts[:-1] + ']'  # remove last comma and close the list
        return "Activations: " + acts
        
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

            # attn_mask = ids_mb.ne(tokenizer.pad_token_id).long() \
            #             if tokenizer.pad_token_id is not None else None # tokenizer는 이 class에 전달 안됨
            attn_mask = None
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

    def call_sync(self, record):
        return asyncio.run(self.__call__(record))
