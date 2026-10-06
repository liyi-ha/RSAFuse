# Local Evaluation Subsets

This private snapshot includes the author's existing paired image subsets:

| Dataset | Infrared | Visible | Pairs |
| --- | ---: | ---: | ---: |
| LLVIP | 50 | 50 | 50 |
| M3FD | 33 | 33 | 33 |
| RoadScene | 42 | 42 | 42 |
| TNO | 25 | 25 | 25 |

Each `Inf` image is paired with the `Vis` image having the same filename stem.
The original local filenames and image bytes are preserved. They may differ
from official dataset identifiers. `manifest.csv` records relative paths,
dimensions and SHA-256 hashes for this snapshot, not an official-ID mapping.

These folders are local fusion evaluation subsets, not the complete datasets.
They do not include object-detection labels, training data, or an assertion
that every image belongs to an official test partition. In particular, local
LLVIP names must not be treated as official image IDs.

The local `RoadScene` folder contains filenames beginning with `FLIR_`. This
snapshot preserves the existing folder label; it does not independently verify
that those files belong to the official RoadScene benchmark. Confirm source
provenance before using that dataset name in public experimental claims.

Source datasets and imagery belong to their respective providers. Their terms
are not replaced by this repository. This inclusion is for the owner's private
research repository; verify permissions, attribution and redistribution terms
before any public release. Obtain full datasets from their original providers.
