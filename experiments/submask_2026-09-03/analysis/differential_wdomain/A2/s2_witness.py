"""S2: does any XOR difference d map a known preimage to another preimage
of the SAME digest?   Tested directly on the archived witnesses.

Families of d:
  F1 weight-1 over 512 message bits                          (512)
  F2 weight-2                                                (130,816)
  F3 MSB-only subsets of W0..W15  -- the ONLY family that is
     free through every modular addition (law (ii): 2^-hw(d&0x7fffffff)) (65,535)
  F4 random weight 3..8                                      (2,000,000)
  F5 kernel of the XOR-linearised 21-round message expansion (sampled)
  F6 kernel of the fully XOR-linearised 21-round compression (sampled)
"""
import sys, time, itertools
import numpy as np
from core import compress, expand, popcnt32, S0, S1, s0, s1, ch, maj, H0, K

R20 = ("d962ca30635f9b74ac6c8c1243a1a9cf800e81bc05f1d2e40764c68c795f7388",
       "a36f4238 f2c9204e b5b6653b 070401f7 5928e4f3 fe766be2 52026907 f7ee1812 "
       "01344603 ea505012 86cbe6cb 6ce73d0b 5c91de6a 43355e3b ff3d5e88 c1ad0b54", 20)
R20A = ("f" * 64,
        "0b0187bf b9aea692 b66effa5 087dca3a 0caea827 1d2f9916 739a224e a87c0eae "
        "7f9ef4a7 b318a7de a8848c61 a7a141a4 11ac114b 952299be 97aaa67f c6acfe57", 20)
R19_1 = ("1e65261c54255188604f5375091839733de63e966b5e4715658226bf03588447",
         "22f091af ec52d67b 74c33819 a280dc6a b001ff1a 1f2356a5 3eccf108 bd9a2333 "
         "abe611d1 6d1e5a20 8041df25 e43d31af aa895a2e 69106ad2 7479fa3a 2a9abb91", 19)
R19_2 = ("fb52f81baed24f8728faf5bbce82c67d510761172fb9876d9e3a72dda351b7ca",
         "37e6702f bc20efea 2dd42a3e 501dfbe9 3cacc578 ea2de1c1 11c0f066 0f22be47 "
         "2a447d2d 13f0080f 1f33df6b d655d8e6 15730eaa 9bf64950 9f129973 5a964edf", 19)
R19_3 = ("1bd7ebbdc4d938fb26d19b5dd5caf333de397bd1c745727bd5556baf38ccf977",
         "3ce8fba4 e2fb9661 44730c59 e1cf4bc0 e1a18d93 97658983 67efe2a7 ef260ecb "
         "d4c6dbe0 13e9388e 95664a59 4d9e248b 74137862 664815ac 89eae95a cd7dbef5", 19)
WITNESSES = [("R20-solver", R20), ("R20-allones", R20A),
             ("R19-P1", R19_1), ("R19-P2", R19_2), ("R19-P3", R19_3)]


def wparse(s):
    return np.array([int(t, 16) for t in s.split()], dtype=np.uint32)


def hparse(s):
    return np.array([int(s[i:i+8], 16) for i in range(0, 64, 8)], dtype=np.uint32)


# ---------------------------------------------------------------- GF(2) tools
def lin_expand_map(rounds):
    """XOR-linearised message expansion: 512 msg-diff bits -> 32*(rounds-16) bits."""
    cols = []
    for i in range(512):
        d = np.zeros((1, 16), dtype=np.uint32)
        d[0, i // 32] = np.uint32(1 << (i % 32))
        w = list(d[0])
        for t in range(16, rounds):
            w.append(s1(w[t-2]) ^ w[t-7] ^ s0(w[t-15]) ^ w[t-16])
        v = 0
        for j, t in enumerate(range(16, rounds)):
            v |= int(w[t]) << (32 * j)
        cols.append(v)
    return cols, 32 * (rounds - 16)


def lin_compress_map(rounds, ch_lin, maj_lin):
    """Fully XOR-linearised compression: 512 msg bits -> 256 output-diff bits.
    ch_lin/maj_lin: which inputs the linear approximation keeps."""
    cols = []
    for i in range(512):
        w = [np.uint32(0)] * 16
        w[i // 32] = np.uint32(1 << (i % 32))
        for t in range(16, rounds):
            w.append(s1(w[t-2]) ^ w[t-7] ^ s0(w[t-15]) ^ w[t-16])
        a = b = c = d = e = f = g = h = np.uint32(0)
        for t in range(rounds):
            chv = np.uint32(0)
            for src, val in zip("xyz", (e, f, g)):
                if src in ch_lin:
                    chv ^= val
            mjv = np.uint32(0)
            for src, val in zip("xyz", (a, b, c)):
                if src in maj_lin:
                    mjv ^= val
            t1 = h ^ S1(e) ^ chv ^ w[t]
            t2 = S0(a) ^ mjv
            h, g, f, e = g, f, e, (d ^ t1)
            d, c, b, a = c, b, a, (t1 ^ t2)
        v = 0
        for j, val in enumerate([a, b, c, d, e, f, g, h]):
            v |= int(val) << (32 * j)
        cols.append(v)
    return cols, 256


def kernel_basis(cols, nout):
    """cols[i] = image of unit vector e_i (as int bitmask).  Return kernel basis
    as list of 512-bit ints."""
    rows = []            # (image, preimage)
    piv = {}
    kern = []
    for i, cv in enumerate(cols):
        img, pre = cv, 1 << i
        newpiv = False
        for b in range(nout - 1, -1, -1):
            if img >> b & 1:
                if b in piv:
                    pi, pp = piv[b]
                    img ^= pi
                    pre ^= pp
                else:
                    piv[b] = (img, pre)
                    newpiv = True
                    break
        if not newpiv and img == 0:
            kern.append(pre)
    # sanity: basis vectors must be nonzero and independent
    assert all(k != 0 for k in kern)
    return kern


def int_to_words(v):
    return np.array([(v >> (32 * j)) & 0xFFFFFFFF for j in range(16)], dtype=np.uint32)


# ---------------------------------------------------------------- families
def fam_weight1():
    return [1 << i for i in range(512)]


def fam_weight2():
    return [(1 << i) | (1 << j) for i in range(512) for j in range(i + 1, 512)]


def fam_msb():
    base = [1 << (32 * w + 31) for w in range(16)]
    out = []
    for m in range(1, 1 << 16):
        v = 0
        for w in range(16):
            if m >> w & 1:
                v |= base[w]
        out.append(v)
    return out


def fam_random_lowweight(rng, n, wlo=3, whi=8):
    out = np.zeros((n, 16), dtype=np.uint32)
    ws = rng.integers(wlo, whi + 1, n)
    for k in range(whi):
        pos = rng.integers(0, 512, n)
        act = (ws > k)
        out[np.arange(n)[act], pos[act] // 32] ^= (np.uint32(1) << (pos[act] % 32).astype(np.uint32))
    return out


def fam_from_kernel(kern, rng, n, maxcomb=4):
    if not kern:
        return np.zeros((0, 16), dtype=np.uint32)
    out = np.zeros((n, 16), dtype=np.uint32)
    kb = len(kern)
    for i in range(n):
        v = 0
        for _ in range(int(rng.integers(1, maxcomb + 1))):
            v ^= kern[int(rng.integers(0, kb))]
        out[i] = int_to_words(v)
    return out


def to_word_array(int_list):
    n = len(int_list)
    a = np.zeros((n, 16), dtype=np.uint32)
    for i, v in enumerate(int_list):
        for j in range(16):
            a[i, j] = (v >> (32 * j)) & 0xFFFFFFFF
    return a


# ---------------------------------------------------------------- test driver
def drop_zero(D):
    """Remove the trivial d = 0 rows (they are not differences)."""
    keep = D.any(axis=1)
    return D[keep]


def check_kernel(cols, nout, kern, k=8):
    """Confirm each basis vector really maps to 0 under the linear map."""
    for v in kern[:k]:
        img = 0
        for i in range(512):
            if v >> i & 1:
                img ^= cols[i]
        assert img == 0, "kernel basis vector does not map to zero"


def test_family(name, W, rounds, target, D, chunk=200000):
    """D: (n,16) uint32 differences. Return hits, min out weight, best d."""
    n = D.shape[0]
    hits = 0
    best = 257
    bestd = None
    for s in range(0, n, chunk):
        blk = D[s:s+chunk]
        Wp = (W[None, :] ^ blk).astype(np.uint32)
        dig = compress(Wp, rounds)
        dif = dig ^ target[None, :]
        wt = popcnt32(dif.reshape(-1)).reshape(-1, 8).sum(axis=1)
        h = int(np.count_nonzero(wt == 0))
        hits += h
        i = int(np.argmin(wt))
        if wt[i] < best:
            best = int(wt[i])
            bestd = blk[i].copy()
    return hits, best, bestd


def main():
    rng = np.random.default_rng(4242)
    t00 = time.time()

    # --- build structured families once (they do not depend on the witness)
    print("building families ...", flush=True)
    F1 = to_word_array(fam_weight1())
    F2 = to_word_array(fam_weight2())
    F3 = to_word_array(fam_msb())
    F4 = drop_zero(fam_random_lowweight(rng, 2_000_000))

    kern_sched = {}
    for R in (19, 20, 21):
        cols, nout = lin_expand_map(R)
        kb = kernel_basis(cols, nout)
        check_kernel(cols, nout, kb)
        kern_sched[R] = kb
        print(f"   linearised expansion kernel dim, R={R}: {len(kb)} "
              f"(generic 512 - {nout} = {512-nout})", flush=True)

    kern_comp = {}
    for R in (19, 20, 21):
        best = None
        for chl, mjl in [("z", "x"), ("y", "x"), ("xyz", "xyz"), ("z", "xyz"), ("y", "z")]:
            cols, nout = lin_compress_map(R, chl, mjl)
            kb = kernel_basis(cols, nout)
            check_kernel(cols, nout, kb)
            if best is None or len(kb) > len(best[1]):
                best = ((chl, mjl), kb)
        kern_comp[R] = best
        print(f"   linearised compression kernel dim, R={R}: {len(best[1])} "
              f"(generic 256; best linearisation Ch~{best[0][0]}, Maj~{best[0][1]})",
              flush=True)

    grand_tested = 0
    grand_hits = 0
    rows = []
    for label, (hexh, wtxt, R) in WITNESSES:
        W = wparse(wtxt)
        T = hparse(hexh)
        got = compress(W[None, :], R)[0]
        assert np.array_equal(got, T), f"{label} does not verify!"
        print(f"\n### {label}  ({R} rounds)  verified OK", flush=True)

        F5 = drop_zero(fam_from_kernel(kern_sched[R], rng, 300_000))
        F6 = drop_zero(fam_from_kernel(kern_comp[R][1], rng, 300_000))
        fams = [("F1 weight-1", F1), ("F2 weight-2", F2), ("F3 MSB-only", F3),
                ("F4 rand wt3-8", F4), ("F5 sched-kernel", F5),
                ("F6 compress-kernel", F6)]
        for fname, D in fams:
            if D.shape[0] == 0:
                print(f"   {fname:<20} empty")
                continue
            hits, best, bd = test_family(label, W, R, T, D)
            grand_tested += D.shape[0]
            grand_hits += hits
            rows.append((label, fname, D.shape[0], hits, best))
            print(f"   {fname:<20} n={D.shape[0]:>9,}  hits={hits:<3d} "
                  f"min |dDigest| = {best:3d} / 256   ({time.time()-t00:6.1f}s)",
                  flush=True)

    print("\n" + "=" * 70)
    print(f"TOTAL differences tested: {grand_tested:,}   TOTAL hits: {grand_hits}")
    if grand_hits == 0:
        import math
        print(f"95% upper bound on P[d maps a preimage to a preimage] "
              f"<= 3/{grand_tested} = 2^{math.log2(3/grand_tested):.1f}")
    print(f"elapsed {time.time()-t00:.1f}s")


if __name__ == "__main__":
    main()
