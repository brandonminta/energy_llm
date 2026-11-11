"""How it runs: beta* calibration, sharded/resumable extraction, status and
merge, the single-process smoke run, and the SIGUSR1/SIGTERM stop. Everything
that reads or writes ``<output_dir>`` lives here."""
