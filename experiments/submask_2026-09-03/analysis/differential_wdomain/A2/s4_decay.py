"""S4: how far does a GF(2)-linear (XOR) trail actually survive in the real
compression function?  This is the direct measurement of the "complementary
difficulty profile": the linear layer is free, the modular additions are not.

For random d in the zero-output kernel of the linearised 21-round map, run the
REAL compression on (W, W^d) for the archived witness and for random W, and
record the first round at which the true state difference departs from the
linearised prediction.
"""
import numpy as np, time
from core import (S0, S1, s0, s1, ch, maj, H0, K, popcnt32)
from s2_witness import (lin_compress_map, kernel_basis, check_kernel,
                        int_to_words, wparse, hparse, WITNESSES)

M = np.uint32(0xFFFFFFFF)


def real_states(W, rounds):
    """W:(n,16) -> list of per-round (a..h) tuples of arrays."""
    from core import expand
    Wx = expand(W, rounds)
    n = W.shape[0]
    a = np.full(n, H0[0], np.uint32); b = np.full(n, H0[1], np.uint32)
    c = np.full(n, H0[2], np.uint32); d = np.full(n, H0[3], np.uint32)
    e = np.full(n, H0[4], np.uint32); f = np.full(n, H0[5], np.uint32)
    g = np.full(n, H0[6], np.uint32); h = np.full(n, H0[7], np.uint32)
    out = []
    for t in range(rounds):
        t1 = (h + S1(e) + ch(e, f, g) + K[t] + Wx[:, t]).astype(np.uint32)
        t2 = (S0(a) + maj(a, b, c)).astype(np.uint32)
        h, g, f, e = g, f, e, (d + t1).astype(np.uint32)
        d, c, b, a = c, b, a, (t1 + t2).astype(np.uint32)
        out.append(np.stack([a, b, c, d, e, f, g, h], axis=1))
    return out


def lin_states(dw16, rounds):
    """linearised predicted (a..h) differences per round, python ints."""
    def rotr(x, n): return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF
    L0 = lambda x: rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
    L1 = lambda x: rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
    l0 = lambda x: rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
    l1 = lambda x: rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
    dw = [int(x) for x in dw16]
    for t in range(16, rounds):
        dw.append(l1(dw[t-2]) ^ dw[t-7] ^ l0(dw[t-15]) ^ dw[t-16])
    a = b = c = d = e = f = g = h = 0
    out = []
    for t in range(rounds):
        dt1 = h ^ L1(e) ^ g ^ dw[t]          # Ch ~ z
        dt2 = L0(a) ^ a                       # Maj ~ x
        h, g, f, e = g, f, e, d ^ dt1
        d, c, b, a = c, b, a, dt1 ^ dt2
        out.append((a, b, c, d, e, f, g, h))
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(11)
    R = 21
    cols, nout = lin_compress_map(R, "z", "x")
    kern = kernel_basis(cols, nout)
    check_kernel(cols, nout, kern)
    NS = 400
    ds = []
    for _ in range(NS):
        v = 0
        for _ in range(int(rng.integers(1, 4))):
            v ^= kern[int(rng.integers(0, len(kern)))]
        if v:
            ds.append(v)
    print(f"{len(ds)} zero-output linearised trails, R={R}")

    # random messages, 4096 per difference
    NM = 4096
    first_dev = []
    for v in ds[:120]:
        dw = int_to_words(v)
        pred = lin_states(dw, R)
        Wr = rng.integers(0, 1 << 32, (NM, 16), dtype=np.uint64).astype(np.uint32)
        A = real_states(Wr, R)
        B = real_states((Wr ^ dw[None, :]).astype(np.uint32), R)
        dev = R
        for t in range(R):
            dif = (A[t] ^ B[t]).astype(np.uint32)
            p = np.array(pred[t], dtype=np.uint32)
            match = np.all(dif == p[None, :], axis=1)
            if not match.any():
                dev = t
                break
        first_dev.append(dev)
    fd = np.array(first_dev)
    print(f"first round at which NO message out of {NM} still follows the "
          f"linear trail:")
    print(f"   min {fd.min()}  median {np.median(fd)}  max {fd.max()}  "
          f"mean {fd.mean():.2f}   (of {R} rounds)")
    hist = np.bincount(fd, minlength=R + 1)
    print("   histogram by round:", {i: int(hist[i]) for i in range(R + 1) if hist[i]})

    # survival probability per round for the best few
    print("\n  per-round survival fraction (fraction of 4096 random messages "
          "still on the linear trail):")
    for v in ds[:3]:
        dw = int_to_words(v)
        pred = lin_states(dw, R)
        Wr = rng.integers(0, 1 << 32, (NM, 16), dtype=np.uint64).astype(np.uint32)
        A = real_states(Wr, R)
        B = real_states((Wr ^ dw[None, :]).astype(np.uint32), R)
        fr = []
        for t in range(R):
            dif = (A[t] ^ B[t]).astype(np.uint32)
            p = np.array(pred[t], dtype=np.uint32)
            fr.append(float(np.all(dif == p[None, :], axis=1).mean()))
        print("   ", [f"{x:.3f}" for x in fr])
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
