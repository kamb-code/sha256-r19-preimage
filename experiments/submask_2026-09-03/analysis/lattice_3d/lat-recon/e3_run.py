"""E3: how much of the unknown must be GIVEN AWAY before the lattice works?

`known_high` top bits of u are handed to the solver as constants.  This is the
only knob that lowers the density; measuring where LLL starts to succeed says
exactly how far SHA-256 sits from the lattice-friendly regime.
"""
import e2_sigma0, json

out = []
with open('e3_density.log', 'w') as fh:
    for w, trials, step in ((8, 40, 1), (12, 40, 1), (16, 30, 1), (32, 10, 2)):
        for kh in range(0, w, step):
            a = e2_sigma0.run(w, trials=trials, known_high=kh, enc='B')
            a['unknown_bits'] = w - kh
            fh.write(json.dumps(a) + "\n"); fh.flush()
            print(json.dumps(a), flush=True)
            out.append(a)
            if a['rate'] >= 1.0:
                break
json.dump(out, open('e3_density.json', 'w'), indent=1)
