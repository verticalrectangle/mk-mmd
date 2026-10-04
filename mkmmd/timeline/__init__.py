"""Timeline analysis of a song clip: stems, beats, downbeats, vocal loudness and word timings (docs/design.md:
Timeline). Heavy dependencies (torch, demucs, faster-whisper) are imported only by the functions that need them
(`pip install mk-mmd[timeline]`)."""
