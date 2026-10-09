"""Deterministic auxiliary sampling from the existing processed training set.

No second dataset, worker prefetch, global RNG consumption, or epoch changes.
Disjointness is by processed row ID, not semantic deduplication.
"""
import random


class StreamPoolPlan:
    VERSION = 1

    def __init__(self, size, batch_size, world_size, control_batches, seed):
        self.signature = (size, batch_size, world_size, control_batches, seed)
        self.global_batch = batch_size * world_size
        count = control_batches * self.global_batch
        if size < count + self.global_batch * 2:
            raise ValueError("Training set is too small for disjoint NetPrefix pools")
        self.order = list(range(size))
        random.Random(seed).shuffle(self.order)
        self.control_ids = self.order[:count]
        self.control_set = set(self.control_ids)
        self.cursor = count

    def control_conflict(self, repair_ids):
        return not self.control_set.isdisjoint(repair_ids)

    def next_audit_ids(self, repair_ids):
        excluded = set(repair_ids)
        selected = []
        while self.cursor < len(self.order) and len(selected) < self.global_batch:
            index = self.order[self.cursor]
            self.cursor += 1
            if index not in excluded:
                selected.append(index)
        # Never rewind or recycle a partial/exhausted audit batch.
        return selected if len(selected) == self.global_batch else None

    def state_dict(self):
        return dict(version=self.VERSION, signature=self.signature, cursor=self.cursor)

    def load_state_dict(self, state):
        if state["version"] != self.VERSION or tuple(state["signature"]) != self.signature:
            raise ValueError("NetPrefix training stream layout changed on resume")
        if not len(self.control_ids) <= state["cursor"] <= len(self.order):
            raise ValueError("Invalid NetPrefix audit cursor")
        self.cursor = state["cursor"]


class TrainingStreamSource:
    def __init__(self, dataset, collate, batch_size, rank, world_size, control_batches, seed):
        self.dataset, self.collate = dataset, collate
        self.batch_size, self.rank = batch_size, rank
        self.plan = StreamPoolPlan(len(dataset), batch_size, world_size, control_batches, seed)
        self.repair_ids = []

    def local_batch(self, global_ids):
        start = self.rank * self.batch_size
        return self.collate([self.dataset[i] for i in global_ids[start:start + self.batch_size]])

    def control_batches(self):
        ids, width = self.plan.control_ids, self.plan.global_batch
        return [self.local_batch(ids[i:i + width]) for i in range(0, len(ids), width)]

    def set_repair_ids(self, ids):
        if any(i < 0 or i >= len(self.dataset) for i in ids):
            raise ValueError("NetPrefix requires stable processed sample IDs")
        self.repair_ids = ids
        return self.plan.control_conflict(ids)

    def __iter__(self):
        return self

    def __next__(self):
        ids = self.plan.next_audit_ids(self.repair_ids)
        if ids is None:
            raise StopIteration
        return self.local_batch(ids)

    def state_dict(self):
        return self.plan.state_dict()

    def load_state_dict(self, state):
        self.plan.load_state_dict(state)
