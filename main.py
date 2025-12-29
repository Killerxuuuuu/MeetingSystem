import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):

    def _list_audio_backends():
        return ["soundfile"]

    torchaudio.list_audio_backends = _list_audio_backends

from modules.system import StreamingMeetingSystem


def main():
    system = StreamingMeetingSystem()
    system.start()


if __name__ == "__main__":
    main()
