# -*- coding: utf-8 -*-

import torch
from torch import Tensor

import numpy as np

from tqdm import tqdm

from typing import List, Tuple, Callable


def flatten_structure(query_structure):
    if type(query_structure) == str:
        return [query_structure]

    flat_structure = []
    for element in query_structure:
        flat_structure.extend(flatten_structure(element))

    return flat_structure


def query_to_atoms(query_structure, flat_ids):
    flat_structure = flatten_structure(query_structure)
    batch_size, query_length = flat_ids.shape
    assert len(flat_structure) == query_length

    query_triples = []
    variable = 0
    previous = flat_ids[:, 0]
    conjunction_mask = []
    negation_mask = []

    for i in range(1, query_length):
        if flat_structure[i] == 'r':
            variable -= 1
            triples = torch.empty(batch_size, 3,
                                  device=flat_ids.device,
                                  dtype=torch.long)
            triples[:, 0] = previous
            triples[:, 1] = flat_ids[:, i]
            triples[:, 2] = variable

            query_triples.append(triples)
            previous = variable
            conjunction_mask.append(True)
            negation_mask.append(False)
        elif flat_structure[i] == 'e':
            previous = flat_ids[:, i]
            variable += 1
        elif flat_structure[i] == 'u':
            conjunction_mask = [False] * len(conjunction_mask)
        elif flat_structure[i] == 'n':
            negation_mask[-1] = True

    atoms = torch.stack(query_triples, dim=1)
    num_variables = variable * -1
    conjunction_mask = torch.tensor(conjunction_mask).unsqueeze(0).expand(batch_size, -1)
    negation_mask = torch.tensor(negation_mask).unsqueeze(0).expand(batch_size, -1)

    return atoms, num_variables, conjunction_mask, negation_mask


def create_instructions(chains):
    instructions = []

    prev_start = None
    prev_end = None

    path_stack = []
    start_flag = True
    for chain_ind, chain in enumerate(chains):
        if start_flag:
            prev_end = chain[-1]
            start_flag = False
            continue

        if prev_end == chain[0]:
            instructions.append(f"hop_{chain_ind-1}_{chain_ind}")
            prev_end = chain[-1]
            prev_start = chain[0]

        elif prev_end == chain[-1]:

            prev_start = chain[0]
            prev_end = chain[-1]

            instructions.append(f"intersect_{chain_ind-1}_{chain_ind}")
        else:
            path_stack.append(([prev_start, prev_end],chain_ind-1))
            prev_start = chain[0]
            prev_end = chain[-1]
            start_flag = False
            continue

        if len(path_stack) > 0:

            path_prev_start = path_stack[-1][0][0]
            path_prev_end = path_stack[-1][0][-1]

            if path_prev_end == chain[-1]:

                prev_start = chain[0]
                prev_end = chain[-1]

                instructions.append(f"intersect_{path_stack[-1][1]}_{chain_ind}")
                path_stack.pop()
                continue

    ans = []
    for inst in instructions:
        if ans:

            if 'inter' in inst and ('inter' in ans[-1]):
                    last_ind = inst.split("_")[-1]
                    ans[-1] = ans[-1]+f"_{last_ind}"
            else:
                ans.append(inst)

        else:
            ans.append(inst)

    instructions = ans
    return instructions


def t_norm_fn(tens_1: Tensor, tens_2: Tensor, t_norm: str = 'min') -> Tensor:
    if 'min' in t_norm:
        return torch.min(tens_1, tens_2)
    elif 'prod' in t_norm:
        return tens_1 * tens_2


def t_conorm_fn(tens_1: Tensor, tens_2: Tensor, t_norm: str = 'min') -> Tensor:
    if 'min' in t_norm:
        return torch.max(tens_1, tens_2)
    elif 'prod' in t_norm:
        return (tens_1 + tens_2) - (tens_1 * tens_2)


def make_batches(size: int, batch_size: int) -> List[Tuple[int, int]]:
    max_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, max_batch)]
    return res


def get_best_candidates(rel: Tensor,
                        arg1: Tensor,
                        forward_emb: Callable[[Tensor, Tensor], Tensor],
                        entity_embeddings: Callable[[Tensor], Tensor],
                        candidates: int = 5,
                        last_step: bool = False) -> Tuple[Tensor, Tensor]:
    batch_size, embedding_size = rel.shape[0], rel.shape[1]

    # [B, N]
    scores = forward_emb(arg1, rel)

    if not last_step:
        # [B, K], [B, K]
        k = min(candidates, scores.shape[1])
        z_scores, z_indices = torch.topk(scores, k=k, dim=1)
        # [B, K, E]
        z_emb = entity_embeddings(z_indices)

        # XXX: move before return
        assert z_emb.shape[0] == batch_size
        assert z_emb.shape[2] == embedding_size
    else:
        z_scores = scores

        z_indices = torch.arange(z_scores.shape[1]).view(1, -1).repeat(z_scores.shape[0], 1).to(rel.device)
        z_emb = entity_embeddings(z_indices)

    return z_scores, z_emb
