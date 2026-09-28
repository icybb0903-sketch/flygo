# Third-party notices

## MaleCNS v1.0 data

Source: [MaleCNS download page](https://male-cns.janelia.org/download/).
Attribution: FlyEM (HHMI Janelia), University of Cambridge (Department of
Zoology), MRC Laboratory of Molecular Biology, and Google Research.
License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Derived assets in `data/malecns/runtime/` and the connection subgraph in
`data/p1/` use this source data, not the project's MIT license. Changes:
selected located/classified nodes, weight-threshold filtering, deterministic
CSR arrays, engineering neurotransmitter codes and engineered input/output
mappings. Original source and artifact hashes are recorded in the manifests.
This work is neither the complete original dataset nor an officially
validated physiological model. The data providers do not endorse flygo.

## three.js

The vendored `app/vendor/three.core.js` and `three.module.js` are licensed
under MIT by the three.js authors. The original copyright and license are
retained in [THREE_LICENSE.txt](app/vendor/THREE_LICENSE.txt).

## Engineering and visualization references

- [Housefly](https://github.com/sandbornm/housefly), Mark Sandborn:
  engineering reference for connectome-driven task interfaces, frozen graph
  dynamics and explicit model limitations. Its upstream code is MIT.
- [fly-blackjack](https://github.com/WilliamJones/fly-blackjack):
  reference for connectome visualization presentation.

The current flygo application is the Python/JavaScript Go project, not the
Housefly blackjack application. Reference links are not affiliations or
endorsements. No original blackjack gameplay, NeuroMechFly body asset or
blackjack demonstration media is part of this release tree.
