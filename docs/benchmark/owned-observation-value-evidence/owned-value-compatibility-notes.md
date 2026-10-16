# Reviewed internal observation contract

The independent review cleared one exact-source installation with this limitation preserved. `OwnedObservation` is an internal value-storage implementation, not a general drop-in replacement for every behavior of the mutable `Observation` dataclass.

Pixel reads return independent readonly NumPy array snapshots. Array object identity is not stable; metadata edits on one snapshot do not affect the committed value. Explicit `observation.pixel = value` replaces the committed value and cached identity. Existing mutable observations, subclasses and custom physical-key hooks retain their original paths.

Inherited dataclass equality is not supported for the internal owned type: with a two-element NumPy pixel, even `o == o` raises `ValueError`, whereas the original mutable observation returns `True`. The independent reviewer found no affected equality caller in the inspected repository. The mapper, bundle solver and export paths consume the pixel values and physical identities rather than generic observation equality. External consumers must not assume general dataclass equality compatibility.

No source repair is made in this installation: changing equality would change the frozen source and invalidate its exact-source approval. A future repair requires separately scoped validation. See [the authentic review](owned-value-independent-review33.md).

Full09's 12.95% lower observed case time is qualified timing evidence from arms separated by resource stops, not an independently established causal speedup. Source/output identity proves correctness equivalence, not equivalent resource conditions. Full01 raw ATE remains worse than VO; aligned ATE has only about0.444% margin. Candidate runtimes remain above VO.
