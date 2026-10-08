"""Render the paper's display equations to paper/figures/eq_<name>.png.

    python paper/make_equations.py

Each equation is matplotlib mathtext at 600 dpi, cropped tight. The keys here
are what build_master.js references; the maths follows chromgraph/ exactly
(data.detrend, graph._topk_distal, model.EdgeBias/ContactBiasedAttention/Block,
train.*_loss), so change the code and this together.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"

EQUATIONS = {
    "expected": r"$E(s)=\frac{1}{|V_s|}\sum_{(i,\,i+s)\in V_s} M_{i,\,i+s},\qquad"
                r"\mathrm{O/E}_{ij}=\frac{M_{ij}}{E(|i-j|)},\qquad"
                r"c_{ij}=\frac{\mathrm{O/E}_{ij}}{1+\mathrm{O/E}_{ij}}\in(0,1)$",
    "ssm": r"$y=\mathcal{F}^{-1}\!\left(\mathcal{F}(u)\cdot\mathcal{F}(K)\right)+D\odot u,\qquad"
           r"K[t]=\sum_{n=1}^{N} c_n\,b_n\,a_n^{\,t},\qquad a_n=\exp(-\exp(\rho_n))\in(0,1)$",
    "bias": r"$b_{ij}=\mathrm{MLP}_{\mathrm{HiC}}(c_{ij})+\mathrm{MLP}_{\mathrm{dist}}"
            r"\left(\log_2(1+d_{ij})\right)\in\mathbb{R}^{H}$",
    "attention": r"$\alpha^{(h)}_{ij}=\frac{\exp\left(q^{(h)\top}_i k^{(h)}_j/\sqrt{d_h}+b^{(h)}_{ij}\right)}"
                 r"{\sum_{l\in\mathcal{N}(i)}\exp\left(q^{(h)\top}_i k^{(h)}_l/\sqrt{d_h}+b^{(h)}_{il}\right)},"
                 r"\qquad j\in\mathcal{N}(i)$",
    "block": r"$u_i=\mathrm{LN}\left(z_i+W_O\left[\sum_{j\in\mathcal{N}(i)}\alpha^{(h)}_{ij}"
             r"v^{(h)}_j\right]_{h=1}^{H}\right),\qquad z_i'=\mathrm{LN}\left(u_i+\mathrm{FFN}(u_i)\right)$",
    "dropout": r"$\mathcal{E}=\mathcal{E}_{\mathrm{local}}\cup\left\{e\in\mathcal{E}_{\mathrm{distal}}"
               r":\ \xi=0\right\},\qquad \xi\sim\mathrm{Bernoulli}(p)\ \mathrm{per\ sample}$",
    "l_mask": r"$\mathcal{L}_{\mathrm{mask}}=\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}"
              r"\left\|g(z_i)-\mathrm{sg}(h_i)\right\|_2^2$",
    "l_con": r"$\mathcal{L}_{\mathrm{con}}=-\frac{1}{|\mathcal{T}|}\sum_{(i,j)\in\mathcal{T}}\log"
             r"\frac{\exp(\tilde z_i^\top\tilde z_j/\tau)}{\exp(\tilde z_i^\top\tilde z_j/\tau)"
             r"+\sum_{(k,l)\in\mathcal{D}_{ij}}\exp(\tilde z_k^\top\tilde z_l/\tau)}$",
    "l_contact": r"$\mathcal{L}_{\mathrm{contact}}=\frac{1}{|\mathcal{T}|}\sum_{(i,j)\in\mathcal{T}}"
                 r"\left(f\left(z_i,z_j,d_{ij}\right)-\log\mathrm{O/E}_{ij}\right)^2,\qquad"
                 r"f=\mathrm{MLP}\left([\,z_i\odot z_j,\ |z_i-z_j|,\ \log_2(1+d_{ij})\,]\right)$",
    "l_distill": r"$\mathcal{L}_{\mathrm{distill}}=\frac{1}{|V|}\sum_{i\in V}"
                 r"\left\|z^{\mathrm{free}}_i-\mathrm{sg}(z_i)\right\|_2^2$",
    "l_total": r"$\mathcal{L}=\lambda_{\mathrm{mask}}\mathcal{L}_{\mathrm{mask}}"
               r"+\lambda_{\mathrm{con}}\mathcal{L}_{\mathrm{con}}"
               r"+\lambda_{\mathrm{contact}}\mathcal{L}_{\mathrm{contact}}"
               r"+\lambda_{\mathrm{distill}}\mathcal{L}_{\mathrm{distill}}$",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 11})
    for name, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=11, color="black")
        fig.savefig(OUT / f"eq_{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02,
                    transparent=False, facecolor="white")
        plt.close(fig)
        print(f"eq_{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
