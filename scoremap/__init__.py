"""Score mapping: place the source-score passages heard in a generated piece into one MusicXML score.

mxl      - read a source MusicXML part into timed elements (absolute positions in quarters) and spanner groups
align    - audio-to-score alignment (chroma/onset DTW, repeat jumps) -> audio seconds <-> score position
extract  - cut points: where a passage can start/end without cutting notes, beams, tuplets (hard) or ties/slurs (soft)
layout   - place the passages on the output timeline (quarter = 1 s), barlines only where nothing is cut
emit     - write the output MusicXML (copied elements verbatim; only durations rescaled, default-x dropped)
verify   - check that every copied element is identical to its source
"""
