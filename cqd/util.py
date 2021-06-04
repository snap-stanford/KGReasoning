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


def top_k_selection(chains,
                    chain_instructions,
                    graph_type,
                    scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                    forward_emb: Callable[[Tensor, Tensor], Tensor],
                    entity_embeddings: Callable[[Tensor], Tensor],
                    candidates: int = 5,
                    t_norm: str = 'min',
                    batch_size: int = 1,
                    scores_normalize: str = 'default'):
    res = None

    if 'disj' in graph_type:
        objective = t_conorm_fn
    else:
        objective = t_norm_fn

    nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]

    scores = None

    batches = make_batches(nb_queries, batch_size)

    for batch in tqdm(batches):

        nb_branches = 1
        nb_ent = 0
        batch_scores = None
        candidate_cache = {}

        batch_size = batch[1] - batch[0]
        dnf_flag = False
        if 'disj' in graph_type:
            dnf_flag = True

        for inst_ind, inst in enumerate(chain_instructions):
            with torch.no_grad():
                if 'hop' in inst:

                    ind_1 = int(inst.split("_")[-2])
                    ind_2 = int(inst.split("_")[-1])

                    indices = [ind_1, ind_2]

                    if objective == t_conorm_fn and dnf_flag:
                        objective = t_norm_fn

                    last_hop = False
                    for hop_num, ind in enumerate(indices):
                        last_step = (inst_ind == len(chain_instructions) - 1) and last_hop

                        lhs, rel, rhs = chains[ind]

                        if lhs is not None:
                            lhs = lhs[batch[0]:batch[1]]
                        else:
                            # print("MTA BRAT")
                            batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
                            lhs = lhs_3d.view(-1, embedding_size)

                        rel = rel[batch[0]:batch[1]]
                        rel = rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                        rel = rel.view(-1, embedding_size)

                        if f"rhs_{ind}" not in candidate_cache:

                            # print("STTEEE MTA")
                            z_scores, rhs_3d = get_best_candidates(rel, lhs, forward_emb, entity_embeddings, candidates, last_step)

                            # [Num_queries * Candidates^K]
                            z_scores_1d = z_scores.view(-1)
                            if 'disj' in graph_type or scores_normalize:
                                z_scores_1d = torch.sigmoid(z_scores_1d)

                            # B * S
                            nb_sources = rhs_3d.shape[0] * rhs_3d.shape[1]
                            nb_branches = nb_sources // batch_size
                            if not last_step:
                                batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
                            else:
                                nb_ent = rhs_3d.shape[1]
                                batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)

                            candidate_cache[f"rhs_{ind}"] = (batch_scores, rhs_3d)

                            if not last_hop:
                                candidate_cache[f"lhs_{indices[hop_num + 1]}"] = (batch_scores, rhs_3d)

                        else:
                            batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                            candidate_cache[f"lhs_{ind + 1}"] = (batch_scores, rhs_3d)
                            last_hop = True
                            continue

                        last_hop = True

                elif 'inter' in inst:
                    ind_1 = int(inst.split("_")[-2])
                    ind_2 = int(inst.split("_")[-1])

                    indices = [ind_1, ind_2]

                    if objective == t_norm_fn and dnf_flag:
                        objective = t_conorm_fn

                    if len(inst.split("_")) > 3:
                        ind_1 = int(inst.split("_")[-3])
                        ind_2 = int(inst.split("_")[-2])
                        ind_3 = int(inst.split("_")[-1])

                        indices = [ind_1, ind_2, ind_3]

                    for intersection_num, ind in enumerate(indices):
                        last_step = (inst_ind == len(chain_instructions) - 1)  # and ind == indices[0]

                        lhs, rel, rhs = chains[ind]

                        if lhs is not None:
                            lhs = lhs[batch[0]:batch[1]]
                            lhs = lhs.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                            lhs = lhs.view(-1, embedding_size)

                        else:
                            batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
                            lhs = lhs_3d.view(-1, embedding_size)
                            nb_sources = lhs_3d.shape[0] * lhs_3d.shape[1]
                            nb_branches = nb_sources // batch_size

                        rel = rel[batch[0]:batch[1]]
                        rel = rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                        rel = rel.view(-1, embedding_size)

                        if intersection_num > 0 and 'disj' in graph_type:
                            batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                            rhs = rhs_3d.view(-1, embedding_size)
                            z_scores = scoring_function(rel, lhs, rhs)

                            z_scores_1d = z_scores.view(-1)
                            if 'disj' in graph_type or scores_normalize:
                                z_scores_1d = torch.sigmoid(z_scores_1d)

                            batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores, t_norm)

                            continue

                        if f"rhs_{ind}" not in candidate_cache or last_step:
                            z_scores, rhs_3d = get_best_candidates(rel, lhs, forward_emb, entity_embeddings, candidates, last_step)

                            # [B * Candidates^K] or [B, S-1, N]
                            z_scores_1d = z_scores.view(-1)
                            if 'disj' in graph_type or scores_normalize:
                                z_scores_1d = torch.sigmoid(z_scores_1d)

                            if not last_step:
                                nb_sources = rhs_3d.shape[0] * rhs_3d.shape[1]
                                nb_branches = nb_sources // batch_size

                            if not last_step:
                                batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
                            else:
                                if ind == indices[0]:
                                    nb_ent = rhs_3d.shape[1]
                                else:
                                    nb_ent = 1

                                batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)
                                nb_ent = rhs_3d.shape[1]

                            candidate_cache[f"rhs_{ind}"] = (batch_scores, rhs_3d)

                            if ind == indices[0] and 'disj' in graph_type:
                                count = len(indices) - 1
                                iterator = 1
                                while count > 0:
                                    candidate_cache[f"rhs_{indices[intersection_num + iterator]}"] = (
                                    batch_scores, rhs_3d)
                                    iterator += 1
                                    count -= 1

                            if ind == indices[-1]:
                                candidate_cache[f"lhs_{ind + 1}"] = (batch_scores, rhs_3d)
                        else:
                            batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
                            candidate_cache[f"rhs_{ind + 1}"] = (batch_scores, rhs_3d)

                            last_hop = True
                            del lhs, rel
                            continue

                        del lhs, rel, rhs, rhs_3d, z_scores_1d, z_scores

        if batch_scores is not None:
            # [B * entites * S ]
            # S ==  K**(V-1)
            scores_2d = batch_scores.view(batch_size, -1, nb_ent)
            res, _ = torch.max(scores_2d, dim=1)
            scores = res if scores is None else torch.cat([scores, res])

            del batch_scores, scores_2d, res, candidate_cache

        else:
            assert False, "Batch Scores are empty: an error went uncaught."

        res = scores

    return res
