from pathlib import Path
import torchaudio

def cut_wav(input_wav_path, output_wav_path, start_time_sec=0.0, max_duration_sec=25.0):
    """
    Cuts a segment from the input WAV file starting at start_time_sec and lasting up to max_duration_sec seconds.
    The segment is saved to output_wav_path.

    Args:
        input_wav_path (str or Path): Path to the input WAV file.
        output_wav_path (str or Path): Path to save the cut WAV file.
        start_time_sec (float): Start time in seconds from where to begin the cut.
        max_duration_sec (float): Maximum duration of the output segment in seconds (default 25.0).
    """
    input_wav_path = str(input_wav_path)
    output_wav_path = str(output_wav_path)
    waveform, sample_rate = torchaudio.load(input_wav_path)
    total_duration = waveform.shape[1] / sample_rate

    if start_time_sec < 0 or start_time_sec >= total_duration:
        raise ValueError(f"start_time_sec ({start_time_sec}) is out of bounds for audio length {total_duration:.2f}s")

    # Calculate start and end sample indices
    start_sample = int(start_time_sec * sample_rate)
    end_sample = min(start_sample + int(max_duration_sec * sample_rate), waveform.shape[1])

    cut_waveform = waveform[:, start_sample:end_sample]
    torchaudio.save(output_wav_path, cut_waveform, sample_rate)
    print(f"Saved cut WAV: {output_wav_path} ({(end_sample-start_sample)/sample_rate:.2f}s)")

# Example usage:
# cut_wav("input.wav", "output_cut.wav", start_time_sec=5.0, max_duration_sec=25.0)
