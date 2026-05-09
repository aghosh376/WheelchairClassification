"""
LSL.py — Diagnostic tool for OpenBCI/Ganglion stream.
Run this first to confirm your stream is live and check sampling rate.
"""
import time
from pylsl import StreamInlet, resolve_byprop, resolve_streams

DURATION = 10  # seconds to test


def find_stream(target_name='obci_eeg1', target_type='EEG', timeout=5.0):
    print(f"Scanning for stream (name='{target_name}' or type='{target_type}')...")
    streams = resolve_byprop('name', target_name, timeout=timeout)
    if not streams:
        streams = resolve_byprop('type', target_type, timeout=timeout)
    if not streams:
        print("\n[ERROR] No streams found. All visible streams:")
        for s in resolve_streams(wait_time=2.0):
            print(f"  - Name: {s.name()} | Type: {s.type()} | "
                  f"Channels: {s.channel_count()} | Rate: {s.nominal_srate()} Hz")
        return None
    info = streams[0]
    print(f"[OK] Found: '{info.name()}' | Type: {info.type()} | "
          f"Channels: {info.channel_count()} | Rate: {info.nominal_srate()} Hz")
    return StreamInlet(info)


def test_sampling_rate(inlet):
    print(f"\nTesting for {DURATION}s — reading all channels...")
    start = time.time()
    total_samples = 0
    num_chunks = 0

    while time.time() < start + DURATION:
        chunk, timestamps = inlet.pull_chunk()
        if chunk:
            num_chunks += 1
            total_samples += len(chunk)
            # Print first sample of each chunk to show channel values
            sample = chunk[0]
            ch_str = " | ".join([f"Ch{i}: {v:.6f}" for i, v in enumerate(sample)])
            print(f"  [chunk {num_chunks:3d}] {len(chunk):3d} samples | First: {ch_str}")

    elapsed = time.time() - start
    avg_rate = total_samples / elapsed
    print(f"\n{'='*60}")
    print(f"Chunks received : {num_chunks}")
    print(f"Total samples   : {total_samples}")
    print(f"Elapsed time    : {elapsed:.2f}s")
    print(f"Avg sample rate : {avg_rate:.1f} Hz")
    if avg_rate < 180 or avg_rate > 220:
        print("[WARNING] Rate is far from 200 Hz — check your OpenBCI settings.")
    else:
        print("[OK] Sampling rate looks good.")


if __name__ == '__main__':
    inlet = find_stream()
    if inlet:
        time.sleep(1)  # let buffer fill
        test_sampling_rate(inlet)