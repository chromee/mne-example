import pylsl
from pylsl import StreamInfo, StreamInlet, StreamOutlet

print(pylsl.protocol_version())
print(pylsl.library_version())
print(pylsl.local_clock())

stream_infos = pylsl.resolve_streams()
print(stream_infos)
