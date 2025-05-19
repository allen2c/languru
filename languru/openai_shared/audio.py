import pathlib
import wave


def save_pcm_as_wav(pcm_data: bytes, wav_filepath: pathlib.Path | str) -> None:
    with wave.open(str(wav_filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24_000)
        wf.writeframes(pcm_data)
