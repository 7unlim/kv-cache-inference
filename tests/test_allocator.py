import random
import pytest
from kv_cache import BlockPool

def test_partial_cow_and_release():
    p = BlockPool(4, 4, 16)
    p.create(0)
    for _ in range(3): p.append([0])
    p.fork(0, 1)
    old = p.table(0)[0]
    assert p.refcount(old) == 2
    a = p.append([1])
    assert a.copies == [(old, p.table(1)[0])]
    assert p.table(0) != p.table(1)
    assert p.length(0) == 3 and p.length(1) == 4
    p.release(0); p.release(1)
    assert p.free_blocks == 4

def test_full_prefix_does_not_copy():
    p = BlockPool(3, 2, 8)
    p.create(0); p.append([0]); p.append([0]); p.fork(0,1)
    a = p.append([0,1])
    assert a.copies == []
    assert p.table(0)[0] == p.table(1)[0]
    assert p.table(0)[1] != p.table(1)[1]

def test_errors_are_atomic():
    p = BlockPool(1, 2, 2)
    p.create(0); p.create(1)
    with pytest.raises(RuntimeError): p.append([0,1])
    assert p.free_blocks == 1 and p.length(0) == p.length(1) == 0
    p.append([0]); p.append([0])
    with pytest.raises(Exception): p.append([0])
    with pytest.raises(ValueError): p.append([1,1])
    with pytest.raises(ValueError): p.fork(0,1)
    assert p.length(0) == 2
    assert p.table(1) == []

@pytest.mark.parametrize("dims", [(0,1,1),(1,0,1),(1,1,0),(-1,1,1)])
def test_invalid_dimensions(dims):
    with pytest.raises(ValueError): BlockPool(*dims)

def test_randomized_ownership():
    rng = random.Random(42)
    p = BlockPool(128,4,64)
    states = {}; next_id = 0
    for _ in range(1500):
        choice = rng.choice(["create","fork","append","release"])
        if not states or choice == "create":
            p.create(next_id); states[next_id] = 0; next_id += 1
        elif choice == "fork":
            source = rng.choice(list(states))
            p.fork(source,next_id); states[next_id] = states[source]; next_id += 1
        elif choice == "release":
            seq = rng.choice(list(states)); p.release(seq); del states[seq]
        else:
            ids = rng.sample(list(states), min(len(states),rng.randint(1,5)))
            before = {i:(p.length(i),p.table(i)) for i in states}
            try:
                p.append(ids)
                for i in ids: states[i] += 1
            except (RuntimeError, ValueError):
                assert before == {i:(p.length(i),p.table(i)) for i in states}
        counts = [0]*128
        for seq,length in states.items():
            assert p.length(seq) == length
            assert len(p.table(seq)) == (length+3)//4
            for block in p.table(seq): counts[block] += 1
        assert sum(c == 0 for c in counts) == p.free_blocks
        assert counts == [p.refcount(i) for i in range(128)]
    for seq in states: p.release(seq)
    assert p.free_blocks == 128

def test_batched_cow_reuses_last_owner_without_false_oom():
    p=BlockPool(2,4,16)
    p.create(0); p.append([0]); p.fork(0,1)
    a=p.append([0,1])
    assert len(a.copies)==1
    assert p.free_blocks==0
    assert p.table(0)!=p.table(1)
