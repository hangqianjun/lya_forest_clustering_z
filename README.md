# Repo info:

Here are a list of the notebooks and notes on what they compute:

- `compute_mean_flux.ipynb`: computes the mean flux of the Lya skewers
- `convert_cats_to_treecorr.ipynb`: combine and save the photometric catalogue to the LSSTxDESI footprint, also including a cut at $z<1$ to limit the size of the catalogue.
- `correlation_function.ipynb`: uses `treecorr` to compute angular correlation functions between the photometric sample and the Lya flux ($\delta_F$). 
- `gal_tests.ipynb`: some tests on the photometric sample, for example, full sky maps and angular correlation functions $C_{\ell}^{gg}$
- `looking_at_CoLoRe.ipynb`: Laura's notebook on accessing and viewing the catalogues
- `lya_tests.ipynb`: computing the $\delta_F$ of the Lya forest skewers, split them into redshift bins, and save them into catalogues.
- `theory.ipynb`: Computing the theory $C_{\ell}^{gg}$ using CCL.
